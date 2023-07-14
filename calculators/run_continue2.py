import sys
import argparse
import os
import vaspjob
f = vaspjob.__file__
repo_dir = f.replace(os.path.join(f.split('/')[-2], f.split('/')[-1]), '')
sys.path.append(repo_dir)
from prediction_tools import CalculationThread
from tqdm import tqdm
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import json





def read_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_json", dest="input_json", type=str, 
                        help="Name of json file with trajs")

    parser.add_argument("-d", "--nthreads", dest="number_of_threads", type=int, default=4,
                        help="Number of threads to distribute predictions to")


    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = read_options() 

    config_yml=os.path.join(repo_dir, 'ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml')
    checkpoint=os.path.join(repo_dir, "ocp/prediction/gemnet_oc_base_oc20_oc22.pt")
    
    num_threads=args.number_of_threads
    #traj_list='trajs0.json'
    traj_list=args.input_json
    os.makedirs('./trajs_s100',exist_ok=True)
    calc = OCPCalculator(config_yml, checkpoint, cpu=False)

    threads = []
    with open ('./Candidates/traj_paths/'+traj_list) as f:
        traj_paths=json.load(f)
        

    traj_out_path='./trajs_s100/'
    for j in range(num_threads):
        lp = range(int(len(traj_paths)/num_threads)*j,int(len(traj_paths)/num_threads)*(1+j))
        thread = CalculationThread(calc=calc,traj_in=[traj_paths[ii] for ii in lp],traj_out_path=traj_out_path)    
        thread.start()
        threads.append(thread)
        sys.stdout = open(os.devnull, "w")

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    
    
    

