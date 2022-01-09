#!/bin/bash

cuda_no=1

# model parameters
batch_size=128
t_size=22

generator_lr=0.0002
discriminator_lr=0.0001
weight_decay=0.0

epochs=100
swa_step_start=80


CUDA_VISIBLE_DEVICES=$cuda_no python sdegan/train.py --batch_size $batch_size --t_size $t_size --generator_lr $generator_lr --discriminator_lr $discriminator_lr --weight_decay $weight_decay --epochs $epochs --swa_step_start $swa_step_start --val_not_included






