from pathlib import Path
from prediction_tools import CalculationThread
import os
from tqdm import tqdm
import sys
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from calculators import vaspjob


f = vaspjob.__file__
repo_dir = f.replace(os.path.join(f.split('/')[-2], f.split('/')[-1]), '')
sys.path.append(repo_dir)

config_yml=os.path.join(repo_dir, 'ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml')
checkpoint=os.path.join(repo_dir, "ocp/prediction/gemnet_oc_base_oc20_oc22.pt")

traj_in='./trajs'
traj_out_path='./trajs_s100'
ppath=Path(traj_in)
traj_paths = sorted(ppath.glob("*.traj"))
calc = OCPCalculator(config_yml, checkpoint, cpu=False)


threads = []
for traj_in in traj_paths:
    thread = CalculationThread(calc=calc, traj_in=traj_in, traj_out_path=traj_out_path)
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()