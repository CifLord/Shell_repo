import sys, random, os, json, threading, lmdb, pickle, torch, argparse
sys.path.append('/shareddata/shell/Shell_repo/')
from prediction_tools import MyTread
from ocpmodels.datasets import LmdbDataset

from structure_generation.bare_slabs import slab_generator
from structure_generation.lmdb_generator import generate_lmdb
from structure_generation.oxide_adsorption import surface_adsorption
from structure_generation.lmdb_generator import convert_atoms_data

def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mpids", dest="list_of_mpids", type=str, 
                        help="List of mpids to run slab/adslab predictions on")
    parser.add_argument("-i", "--input_lmdb", dest="input_lmdb", type=str, 
                        help="Name of lmdb file to write slab/adslabs to (prior to predictions)")
    parser.add_argument("-o", "--output_lmdb", dest="output_lmdb", type=str, 
                        help="Name of lmdb file to write predictions of slab/adslab to")
    parser.add_argument("-d", "--nthreads", dest="number_of_threads", type=int, default=3,
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

    args = parser.parse_args()

    return args


if __name__=="__main__":
    
    args = read_options()    
    
    mpid_list = args.list_of_mpids.split(' ')  
    
    # all_atoms_slabs = []
    # for mpid in mpid_list:
        
    #     # Generate all bare slabs
    #     slab_atoms = slab_generator(mpid, args.mmi, args.slab_size, args.vac_size, 
    #                                 MAPIKEY=args.MAPIKEY, height_tol=2, min_lw=8, tol=0.1, 
    #                                 functional='GemNet-OC', count_undercoordination=False)
    #     all_atoms_slabs.extend(slab_atoms)
        
    #     # Generate all adslabs
    #     for slab in slab_atoms:
    #         adslabs = surface_adsorption(convert_atoms_data(slab))
    #         all_atoms_slabs.extend(adslabs)

    #print('Total number of predictions: %s' %(len(all_atoms_slabs)))
    #generate_lmdb(all_atoms_slabs, args.input_lmdb)
    input_lmdb = LmdbDataset({'src': args.input_lmdb})
    
    # equally distribute dataset to multiple threads
    for i in range(args.number_of_threads):
        lp = range(int(len(input_lmdb)/args.number_of_threads)*i, int(len(input_lmdb)/args.number_of_threads)*(1+i))         
        thread = MyTread([input_lmdb[ii] for ii in lp], args.output_lmdb)
        thread.start()
        # sys.stdout = open(os.devnull, "w") # what is this?
    
    print('Finished the ads-slab energy prediction task')