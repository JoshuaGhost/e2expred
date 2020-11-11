#!/bin/bash

w_exps=( 0.0001 )
GPU_NUM=$1

RUN_CLEAN=$2
split=dev
cd ../../

dataset=movies
for w_exp in "${w_exps[@]}"
do
	CUDA_VISIBLE_DEVICES=$GPU_NUM PYTHONPATH=$PYTHONPATH:./ python latent_rationale/e2expred/train.py --conf_fname ./parameters/${dataset}_e2expred.json --dataset_name ${dataset} --w_aux 1.0 --w_exp $w_exp --data_dir ../data --save_path ./results/e2expred/separate_encoder_cold_start --batch_size 13 --decode_split ${split} 
done
