import sys, random, os, json, threading, lmdb, pickle, torch, argparse, time
import re
from pathlib import Path
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
                        help="input lmdb path")    
    parser.add_argument("-b", "--find_intersection", dest="if_predicted", type=bool, default=False,
                        help="if True, find intersections")   
    parser.add_argument("-d", "--nthreads", dest="number_of_threads", type=int, default=8,
                        help="Number of threads to distribute predictions to")
    parser.add_argument("-j", "--fixedslab", dest="slabs_fix", type=bool, default=False,
                        help="If set to True, the code will search the slab structures and set the fixed layer into correct type and predict ")
    parser.add_argument("-m", "--checktype", dest="check_type", type=str, default='all',choices=('all','slabs'),
                        help="re-run type: all means rerun slabs and ads; slabs means only rerun slabs.")
                        

    args = parser.parse_args()

    return args
    
def find_inter(s1,s2,slabs_only=False):
    dict1={}
    dict2={}
    
    if slabs_only is True:
        for i in range(len(s1)):
            if hasattr(s1[i],'adsorbate'):
                pass
            else:
                dict1[i]=s1[i].rid
        for i in range(len(s2)):
            dict2[i]=s2[i].rid

    else:
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
    
def get_slab_ids(lmdb_path:str):

    slab_idx=[]
    seeit1 = LmdbDataset({"src":lmdb_path})    
    for i in range(len(seeit1)):
        if hasattr(seeit1[i],'adsorbate'):
            pass
        else:
            slab_idx.append(i)
            
    return slab_idx 
    
def process_file(in_lmdb, args):
    i=str(in_lmdb)    
    thread_list=[]
    if args.if_predicted == True:
        if args.check_type == 'all':
            predicted=i.rstrip('.lmdb')+'_ads.lmdb'
            
            seeit1=LmdbDataset({"src":i})
            seeit2=LmdbDataset({"src":predicted})
            need_rerun=find_inter(seeit1,seeit2)
        
            input_lmdb = LmdbDataset({'src': i})
            output_lmdb = i.rstrip('.lmdb')+'_ads2.lmdb'
        else:            
            predicted=i.rstrip('.lmdb')+'_slabs.lmdb'                
            seeit1=LmdbDataset({"src":i})
            if os.path.exists(predicted):
                seeit2=LmdbDataset({"src":predicted})
                need_rerun=find_inter(seeit1,seeit2,True)
            else:
                need_rerun=get_slab_ids(i)                
            
            if len(need_rerun)==0:
                print(i, "no need to rerun")
                return None
            print(len(need_rerun))
            input_lmdb = LmdbDataset({'src': i})
            output_lmdb = i.rstrip('.lmdb')+'_slabs_plus.lmdb'
    else:        
        if args.slabs_fix ==True:
    
            slabs_idx=get_slab_ids(i) 
            input_lmdb = LmdbDataset({'src': i})
            output_lmdb = i.rstrip('.lmdb')+'_slabs.lmdb'      
    
        else:
            input_lmdb = LmdbDataset({'src': i})
            output_lmdb = i.rstrip('.lmdb')+'_ads.lmdb'
    # equally distribute dataset to multiple threads
    #logging.info('start prediction: %s' %(i))
    print('start prediction:%s' %(i))
    print('start prediction:',args.number_of_threads)
    for j in range(args.number_of_threads): 
        gpus=0 
        if args.slabs_fix ==True:                    
            # Calculate the start and end indices for each chunk
            chunk_size = len(slabs_idx) // args.number_of_threads
            start_idx = j * chunk_size
            # For the last thread, include any remaining elements
            end_idx = start_idx + chunk_size if j != args.number_of_threads - 1 else len(slabs_idx)
            # Create the lp for this thread
            lp = slabs_idx[start_idx:end_idx]
        else:        
            lp = range(int(len(input_lmdb)/args.number_of_threads)*j, int(len(input_lmdb)/args.number_of_threads)*(1+j))
        if args.if_predicted == True:
            thread = MyThread([input_lmdb[ii] for ii in lp if ii in need_rerun], output_lmdb, gpus, debug=False,refixed=args.slabs_fix)
        if args.check_type == 'slabs':
            need_rerun = slabs_idx
            chunk_size = len(slabs_idx) // args.number_of_threads
            start_idx = j * chunk_size
            end_idx = start_idx + chunk_size if j != args.number_of_threads - 1 else len(slabs_idx)
            lp = slabs_idx[start_idx:end_idx]                
            thread = MyThread([input_lmdb[ii] for ii in lp if ii in need_rerun], output_lmdb, gpus, debug=False,refixed=True)
        else:
            thread = MyThread([input_lmdb[ii] for ii in lp], output_lmdb, gpus, debug=False,refixed=args.slabs_fix)
        thread_list.append(thread)
        
    return thread_list

if __name__=="__main__":
    
    
    args = read_options()
    ppath=Path(args.input_lmdbs)
    #pattern = re.compile(r'.*\d\.lmdb$')
    # Use a list comprehension with the re.match function to filter the files
    #lmdbs = [path for path in sorted(ppath.glob("*.lmdb")) if pattern.match(str(path))]
    lmdbs=sorted(ppath.glob("*.lmdb"))
    all_threads = []
    for i in lmdbs:
        thread_list = process_file(i, args)
        if thread_list is None:
            continue
        else:
            all_threads.extend(thread_list)
    sys.stdout = open(os.devnull, "w")
    for thread in all_threads:
        thread.start()
    for thread in all_threads:
        thread.join()
