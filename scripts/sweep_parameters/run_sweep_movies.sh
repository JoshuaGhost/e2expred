#!/bin/bash

w_exps=( 0.0001 0.001 0.01 0.1 1.0 10 100 1000 10000 )

GPU_NUM=$1

RUN_CLEAN=$2

cd ../../

if [ clean == "$RUN_CLEAN" ];
then
	rm logs/movies_fever.*
fi

for w_exp in "${w_exps[@]}"
do
	date >> logs/sweep_movies.stdout
	date >> logs/sweep_movies.stderr
	CUDA_VISIBLE_DEVICES=$GPU_NUM  PYTHONPATH=$PYTHONPATH:./ python latent_rationale/mtl_e2e/train.py --conf_fname ./parameters/movies_e2expred.json --dataset_name movies --w_aux 1.0 --w_exp $w_exp --data_dir ../data --save_path ./results/mtl_e2e/separate_encoder_cold_start --batch_size 13  1>logs/sweep_movies.stdout 2>logs/sweep_movies.stderr
	date >> logs/sweep_movies.stdout
	date >> logs/sweep_movies.stderr
done

