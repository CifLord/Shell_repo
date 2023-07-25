import sys
import os
import vaspjob
f = vaspjob.__file__
repo_dir = f.replace(os.path.join(f.split('/')[-2], f.split('/')[-1]), '')
sys.path.append(repo_dir)
import json
import argparse

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--traj_dir", dest="traj_dir", type=str, 
                        help="Location of trajs")
    parser.add_argument("-d", "--gpus", dest="gpus", type=int, default=4,
                        help="Number of GPUs to distribute jobs across")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_options()
    traj_in = args.traj_dir
    num_gpus = args.gpus

    traj_paths = [os.path.join(args.traj_dir, f) for f in os.listdir(args.traj_dir)]
    num_paths_per_gpu = len(traj_paths) // num_gpus
    remainder_paths = len(traj_paths) % num_gpus

    os.makedirs('./Candidates/traj_paths', exist_ok=True)

    start_idx = 0
    for i in range(num_gpus):
        end_idx = start_idx + num_paths_per_gpu + (1 if i < remainder_paths else 0)
        with open(f'Candidates/traj_paths/trajs{i}.json', 'w') as f:
            json.dump(traj_paths[start_idx:end_idx], f)
        start_idx = end_idx
