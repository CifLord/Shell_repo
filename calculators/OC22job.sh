#!/usr/bin/env python

#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -N 1 -n 1
#SBATCH -t 200:00:00
#SBATCH --mem=180G 
#SBATCH --gpus=1


CUDA_VISIBLE_DEVICES=0 python run_predictions.py -j "Candidates/hpi_test1.json" -i "prediction/hpi_1.lmdb" &
CUDA_VISIBLE_DEVICES=1 python run_predictions.py -j "Candidates/hpi_test2.json" -i "prediction/hpi_2.lmdb" &
CUDA_VISIBLE_DEVICES=2 python run_predictions.py -j "Candidates/hpi_test3.json" -i "prediction/hpi_3.lmdb" &
CUDA_VISIBLE_DEVICES=3 python run_predictions.py -j "Candidates/hpi_test4.json" -i "prediction/hpi_4.lmdb" &
CUDA_VISIBLE_DEVICES=4 python run_predictions.py -j "Candidates/hpi_test5.json" -i "prediction/hpi_5.lmdb" &
CUDA_VISIBLE_DEVICES=5 python run_predictions.py -j "Candidates/hpi_test6.json" -i "prediction/hpi_6.lmdb" &
CUDA_VISIBLE_DEVICES=6 python run_predictions.py -j "Candidates/hpi_test7.json" -i "prediction/hpi_7.lmdb" &
CUDA_VISIBLE_DEVICES=7 python run_predictions.py -j "Candidates/hpi_test8.json" -i "prediction/hpi_8.lmdb" &
