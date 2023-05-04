#!/bin/bash

echo starting 

CUDA_VISIBLE_DEVICES=1,2,3 python run_predictions_gpu.py -m "mp-505009 mp-21908 mp-505271 mp-674490 mp-1224786" -i "prediction/test_mp2.lmdb" -j 5 -q 2 -d 10 


echo finished
