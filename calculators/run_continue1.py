import sys
import os
import vaspjob
f = vaspjob.__file__
repo_dir = f.replace(os.path.join(f.split('/')[-2], f.split('/')[-1]), '')
sys.path.append(repo_dir)
import json

traj_in='./trajs'
num_gpus=4

traj_paths = os.listdir('./trajs')
num_paths=len(traj_paths)//num_gpus
os.makedirs('./Candidates/traj_paths',exist_ok=True)
for i in range(num_gpus):
    with open(f'Candidates/traj_paths/trajs{i}.json', 'w') as f:
        json.dump(traj_paths[i:(i+1)*num_paths], f)

