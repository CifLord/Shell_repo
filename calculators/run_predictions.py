import sys, random, os, json, threading, lmdb, pickle, torch, argparse, time
import vaspjob
f = vaspjob.__file__
repo_dir = f.replace(os.path.join(f.split('/')[-2], f.split('/')[-1]), '')
sys.path.append(repo_dir)

from prediction_tools import MyThread
from ocpmodels.datasets import LmdbDataset
import logging
from structure_generation.bare_slabs import slab_generator
from structure_generation.lmdb_generator import generate_lmdb, lmdb_size
from structure_generation.oxide_adsorption import surface_adsorption
from structure_generation.lmdb_generator import convert_atoms_data

def read_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_lmdb", dest="input_lmdb", type=str, 
                        help="Name of lmdb file to write slab/adslabs to (prior to predictions)")
    parser.add_argument("-j", "--json_list", dest="json_path", type=str,default=None, 
                        help="Name of json file to of mp-ids")
    parser.add_argument("-m", "--mpids", dest="list_of_mpids", type=str, default=None,
                        help="List of mpids to run slab/adslab predictions on")
    parser.add_argument("-d", "--nthreads", dest="number_of_threads", type=int, default=4,
                        help="Number of threads to distribute predictions to")
    parser.add_argument("-w", "--mmi", dest="mmi", type=int, default=1,
                        help="Max miller index")
    parser.add_argument("-s", "--slab_size", dest="slab_size", type=float, default=12.5,
                        help="Slab size")
    parser.add_argument("-v", "--vac_size", dest="vac_size", type=float, default=12.5,
                        help="Vacuum size")
    parser.add_argument("-k", "--mapikey", dest="MAPIKEY", type=str, 
                        default='11nq6MypZgP6PxZwugyPGW9w6UFVF5Ja',
                        help="Materials Project API KEY")
    parser.add_argument("-b", "--debug", dest="debug", type=str, default=False, 
                        help="Run in debug mode, ie don't run the ASE calculator but do everything else")
    parser.add_argument("-g", "--gpus", dest="gpus", type=int, default=1, 
                        help="Number of GPUs available")

    args = parser.parse_args()

    return args


if __name__=="__main__":
    
    initT = time.time()
    
    args = read_options()  
    if args.list_of_mpids is None:
        if args.json_path is None:
            raise TypeError("Need input mp-ids,either in json or list")
        else:        
            with open(args.json_path) as f:
                mpid_list=json.load(f)
    else:    
        mpid_list = args.list_of_mpids.split(' ')

    p=0
    log_fname=str(args.input_lmdb).replace('.lmdb','.log')
    logging.basicConfig(filename=log_fname, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s')
    os.makedirs('prediction',exist_ok=True)
    for mpid in mpid_list:
        all_atoms_slabs = []        
        # Generate all bare slabs
        slab_atoms = slab_generator(mpid, args.mmi, args.slab_size, args.vac_size, 
                                    MAPIKEY=args.MAPIKEY, height_tol=2, min_lw=8, tol=0.1, 
                                    functional='GemNet-OC', count_undercoordination=False)
        if len(slab_atoms)==0:
            continue
        all_atoms_slabs.extend(slab_atoms)        
        # Generate all adslabs
        for slab in slab_atoms:
            adslabs = surface_adsorption(convert_atoms_data(slab))
            all_atoms_slabs.extend(adslabs)
        input_pathname=args.input_lmdb.rstrip('.lmdb')+'{:04d}'.format(p)+'.lmdb' 
        logging.info('Total number of predictions: %s' %(len(all_atoms_slabs)))       
        #print('Total number of predictions: %s' %(len(all_atoms_slabs)))
        
        if lmdb_size(input_pathname) >=10000:
            p+=1
            input_pathname=args.input_lmdb.rstrip('.lmdb')+'{:04d}'.format(p)+'.lmdb'                  
        generate_lmdb(all_atoms_slabs, input_pathname)
        #print('finished slab generation: %s' %(mpid))
        logging.info('finished slab generation: %s' %(mpid)) 
        
    if p==0:
        input_lmdbs=[args.input_lmdb.rstrip('.lmdb')+'{:04d}'.format(p)+'.lmdb']
    else:        
        input_lmdbs=[args.input_lmdb.rstrip('.lmdb')+'{:04d}'.format(p)+'.lmdb' for p in range(p)]   
    logging.info('finished all slab generation: %s' %(mpid)) 
    for i in input_lmdbs:
        input_lmdb = LmdbDataset({'src': i})
        output_lmdb = i.rstrip('.lmdb')+'_ads.lmdb'
        # equally distribute dataset to multiple threads
        for j in range(args.number_of_threads): 
            lp = range(int(len(input_lmdb)/args.number_of_threads)*j, 
                       int(len(input_lmdb)/args.number_of_threads)*(1+j))
            thread = MyThread([input_lmdb[ii] for ii in lp], output_lmdb, 
                              args.gpus, debug=args.debug)
            thread.start()
            sys.stdout = open(os.devnull, "w")
    
    logging.info('Finished all OC22 predictions in %s' %(time.time() - initT))
