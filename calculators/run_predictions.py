import sys, random, os, json, threading, lmdb, pickle, torch, argparse
sys.path.append('/shareddata/shell/Shell_repo/')
from prediction_tools import MyThread
from ocpmodels.datasets import LmdbDataset
import logging
from structure_generation.bare_slabs import slab_generator
from structure_generation.lmdb_generator import generate_lmdb,lmdb_size
from structure_generation.oxide_adsorption import surface_adsorption
from structure_generation.lmdb_generator import convert_atoms_data

def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mpids", dest="list_of_mpids", type=str, 
                        help="List of mpids to run slab/adslab predictions on")
    parser.add_argument("-i", "--input_lmdb", dest="input_lmdb", type=str, 
                        help="Name of lmdb file to write slab/adslabs to (prior to predictions)")
    #parser.add_argument("-o", "--output_lmdb", dest="output_lmdb", type=str, 
                        #help="Name of lmdb file to write predictions of slab/adslab to")
    parser.add_argument("-j", "--batch", dest="batch", type=int, default=5, 
                        help="number of batch size in one GPU")    
    parser.add_argument("-q", "--ngpus", dest="number_of_gpus", type=int, default=3, 
                        help="number of GPUs")                 
    parser.add_argument("-d", "--nthreads", dest="number_of_threads", type=int, default=5,
                        help="Number of threads to distribute predictions to")
    parser.add_argument("-w", "--mmi", dest="mmi", type=int, default=1,
                        help="Max miller index")
    parser.add_argument("-s", "--slab_size", dest="slab_size", type=float, default=12.5,
                        help="Slab size")
    parser.add_argument("-v", "--vac_size", dest="vac_size", type=float, default=12.5,
                        help="Vacuum size")
    parser.add_argument("-k", "--mapikey", dest="MAPIKEY", type=str, 
                        default='HO6BA47aEuOPR8Puc0qshrc6G9596FNa',
                        help="Materials Project API KEY")
    parser.add_argument("-b", "--debug", dest="debug", type=str, default=False, 
                        help="Run in debug mode, ie don't run the ASE calculator but do everything else")

    args = parser.parse_args()

    return args


if __name__=="__main__":
    
    args = read_options()    
    mpid_list = args.list_of_mpids.split(' ')  
    p=0
    logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
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
        input_pathname=args.input_lmdb.rstrip('.lmdb')+'{:05d}'.format(p)+'.lmdb' 
        logging.info('Total number of predictions: %s' %(len(all_atoms_slabs)))       
        #print('Total number of predictions: %s' %(len(all_atoms_slabs)))
        
        if lmdb_size(input_pathname) >=10000:
            p+=1
            input_pathname=args.input_lmdb.rstrip('.lmdb')+'{:05d}'.format(p)+'.lmdb'                  
        generate_lmdb(all_atoms_slabs, input_pathname)
        #print('finished slab generation: %s' %(mpid))
        logging.info('finished slab generation: %s' %(mpid)) 
    input_lmdbs=[args.input_lmdb.rstrip('.lmdb')+'{:05d}'.format(p)+'.lmdb' for p in range(p)]    
    for i in input_lmdbs:
        input_lmdb = LmdbDataset({'src': i})
        output_lmdb = i.rstrip('.lmdb')+'_ads.lmdb'
        # equally distribute dataset to multiple threads

        for j in range(args.number_of_threads):
            # allocate GPUs based on the index of the thread
            gpus = min(j // args.batch, args.number_of_gpus)
            # calculate the subset of the input dataset to process in this thread
            start_idx = j * len(input_lmdb) // args.number_of_threads
            end_idx = (j + 1) * len(input_lmdb) // args.number_of_threads
            # create a thread with the subset of the input dataset
            thread = MyThread(input_lmdb[start_idx:end_idx], output_lmdb, gpus, debug=args.debug)
            thread.start()
            # suppress stdout to reduce clutter
            sys.stdout = open(os.devnull, "w")
        if i%2000 == 0:
            logging.info('finished slab generation: %s' %(i))        
            #print('Finished the ads-slab energy prediction task')
