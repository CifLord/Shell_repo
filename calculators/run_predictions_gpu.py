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
    
    parser.add_argument("-i", "--in_lmdb", dest="input_lmdbs", type=str, 
                        help="input lmdb files")    
    parser.add_argument("-b", "--find_intersection", dest="if_predicted", type=bool, default=False,
                        help="if True, then no intersections")   
    parser.add_argument("-d", "--nthreads", dest="number_of_threads", type=int, default=4,
                        help="Number of threads to distribute predictions to")
                        

    args = parser.parse_args()

    return args
    
def find_inter(s1,s2):
    dict1={}
    dict2={}
    for i in range(len(s1)):
        dict1[i]=s1[i].rid
    for i in range(len(s2)):
        dict2[i]=s2[i].rid
  
    set1=set([dict1.values()][0])
    set2=set([dict2.values()][0])   
    set3=list(set1-set2)
    dict11={v:k for k,v in dict1.items()}
    result = [dict11[i] for i in set3]
    return result

if __name__=="__main__":
    
    args = read_options()
    lmdbs = args.input_lmdbs.split(' ') #['prediction/phase10001.lmdb']    
    

    for i in lmdbs:
        print(i)
        if args.if_predicted == True:
            predicted=i.rstrip('.lmdb')+'_ads.lmdb'
            seeit1=LmdbDataset({"src":i})
            seeit2=LmdbDataset({"src":predicted})
            need_rerun=find_inter(seeit1,seeit2)
        
            input_lmdb = LmdbDataset({'src': i})
            output_lmdb = i.rstrip('.lmdb')+'_ads2.lmdb'
        else:
            input_lmdb = LmdbDataset({'src': i})
            output_lmdb = i.rstrip('.lmdb')+'_ads.lmdb'
        # equally distribute dataset to multiple threads
        #logging.info('start prediction: %s' %(i))
        print('start prediction:%s' %(i))
        print('start prediction:',args.number_of_threads)
        for j in range(args.number_of_threads): 
            gpus=0            
            lp = range(int(len(input_lmdb)/args.number_of_threads)*j, int(len(input_lmdb)/args.number_of_threads)*(1+j))
            if args.if_predicted == True:
                thread = MyThread([input_lmdb[ii] for ii in lp if ii in need_rerun], output_lmdb, gpus, debug=False)
            else:
                thread = MyThread([input_lmdb[ii] for ii in lp], output_lmdb, gpus, debug=False)
            thread.start()
            sys.stdout = open(os.devnull, "w")