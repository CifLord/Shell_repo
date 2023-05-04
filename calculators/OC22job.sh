#!/usr/bin/env python

#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -N 1 -n 1
#SBATCH -t 200:00:00
#SBATCH --mem=180G 
#SBATCH --gpus=1

