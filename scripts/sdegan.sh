#!/bin/bash

cuda_no=1

# model parameters
batch_size=128
epochs=400

# Synthetic Data
for i in {1..5}
do
    echo "CUDA_VISIBLE_DEVICES=$cuda_no python sdegan/train.py --batch_size $batch_size --epochs $epochs --exp_no $i"
    CUDA_VISIBLE_DEVICES=$cuda_no python sdegan/train.py --batch_size $batch_size --epochs $epochs --exp_no $i
done

# HSI Data
for i in {1..5}
do
    echo "CUDA_VISIBLE_DEVICES=$cuda_no python sdegan/train.py --batch_size $batch_size --epochs $epochs --exp_no $i --real_data"
    CUDA_VISIBLE_DEVICES=$cuda_no python sdegan/train.py --batch_size $batch_size --epochs $epochs --exp_no $i --real_data
done




