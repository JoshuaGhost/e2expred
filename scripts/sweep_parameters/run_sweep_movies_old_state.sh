#!/bin/bash

#w_exps=( 0.0001 0.001 0.01 0.1 1.0 10 100 1000 10000 )
w_exps=( 1.0 )
#datasets = ( movies fever multirc )
GPU_NUM=$1

RUN_CLEAN=$2
split=test
cd ../../

if [ clean == "$RUN_CLEAN" ];
then
	rm logs/sweep_movies.${split}.std*
fi

dataset=movies
for w_exp in "${w_exps[@]}"
do
	echo ${w_exp} >> logs/sweep_movies.${split}.stdout
	echo ${w_exp} >> logs/sweep_movies.${split}.stderr
	date >> logs/sweep_movies.${split}.stdout
	date >> logs/sweep_movies.${split}.stderr
	CUDA_VISIBLE_DEVICES=$GPU_NUM PYTHONPATH=$PYTHONPATH:./ python latent_rationale/mtl_e2e/train.py --conf_fname ./parameters/${dataset}_e2expred.json --dataset_name ${dataset} --w_aux 1.0 --w_exp $w_exp --data_dir ../data --save_path ./results/e2expred/separate_encoder_cold_start --batch_size 13 1>>logs/sweep_${dataset}.${split}.stdout 2>>logs/sweep_${dataset}.${split}.stderr
	date >> logs/sweep_${dataset}.${split}.stdout
	date >> logs/sweep_${dataset}.${split}.stderr
done

