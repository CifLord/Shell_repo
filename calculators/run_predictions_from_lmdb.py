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

    parser.add_argument("-i", "--lmdb_list", dest="lmdb_list", type=str, 
                        help="Name of lmdb file to write slab/adslabs to (prior to predictions)")
    parser.add_argument("-d", "--nthreads", dest="number_of_threads", type=int, default=4,
                        help="Number of threads to distribute predictions to")
    parser.add_argument("-b", "--debug", dest="debug", type=str, default=False, 
                        help="Run in debug mode, ie don't run the ASE calculator but do everything else")
    parser.add_argument("-g", "--gpus", dest="gpus", type=int, default=1, 
                        help="Number of GPUs available")
    parser.add_argument("-a", "--remove_ads", dest="remove_ads", type=str, default=None, 
                        help="Adsorbate to omit")
    parser.add_argument("-s", "--add_spring", dest="add_spring", type=str, default=True, 
                        help="Whether or not to add Hookean constraints")

    args = parser.parse_args()

    return args


if __name__=="__main__":
    
    initT = time.time()
    
    args = read_options()  
    with open(args.lmdb_list) as f:
        lmdb_list = json.load(f)
    
        for n in lmdb_list:
            if not os.path.isfile(n):
                continue
            input_lmdb = LmdbDataset({'src': n})
            output_lmdb = n.rstrip('.lmdb')+'_ads.lmdb'
            # equally distribute dataset to multiple threads
            for j in range(args.number_of_threads): 
                lp = range(int(len(input_lmdb)/args.number_of_threads)*j, 
                           int(len(input_lmdb)/args.number_of_threads)*(1+j))
                thread = MyThread([input_lmdb[ii] for ii in lp], output_lmdb, 
                                  args.gpus, debug=args.debug, skip_ads=args.remove_ads, add_spring=args.add_spring)
                thread.start()
                sys.stdout = open(os.devnull, "w")

        logging.info('Finished all OC22 predictions in %s' %(time.time() - initT))
