from pathlib import Path
from prediction_tools import CalculationThread
import os
from tqdm import tqdm
import sys
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from calculators import vaspjob
import json
sys.path.append(repo_dir)
f = vaspjob.__file__
repo_dir = f.replace(os.path.join(f.split('/')[-2], f.split('/')[-1]), '')
config_yml=os.path.join(repo_dir, 'ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml')
checkpoint=os.path.join(repo_dir, "ocp/prediction/gemnet_oc_base_oc20_oc22.pt")
calc = OCPCalculator(config_yml, checkpoint, cpu=False)

traj_in='./trajs'
traj_out_path='./trajs_s100'
num_gpus=8
num_threads=8

ppath=Path(traj_in)
traj_paths = sorted(ppath.glob("*.traj"))
num_paths=len(traj_paths)/num_gpus
for i in range(len(num_gpus)):
    with open(f'Candidates/trajs{i}.json', 'w') as f:
        json.dump(traj_paths[i:(i+1)*num_paths], f)

threads = []
for i in num_gpus:
    traj_paths=json.load(f'Candidates/trajs{i}.json')
    for traj_in in traj_paths:
        thread = CalculationThread(calc=calc, traj_in=traj_in, traj_out_path=traj_out_path)
        thread.start()
        threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
    sys.stdout = open(os.devnull, "w")