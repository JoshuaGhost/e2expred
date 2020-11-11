#!/bin/bash


GPU_NUM=$1

RUN_CLEAN=$2

pushd ../

mkdir -p logs

if [ clean == "$RUN_CLEAN" ];
then
	rm ./logs/sweep_fever_0.2*
fi

for w_exp in 0.{2..9};
do
	date >> logs/sweep_fever_0.2_0.9.stdout
	date >> logs/sweep_fever_0.2_0.9.stderr
	CUDA_VISIBLE_DEVICES=$GPU_NUM PYTHONPATH=$PYTHONPATH:./ python latent_rationale/e2expred/train.py --conf_fname ./parameters/fever_e2expred.json --dataset_name fever --w_aux 1.0 --w_exp $w_exp --data_dir ../data --save_path ./results/e2expred/separate_encoder_cold_start --batch_size 13 --train_on_part 0.1 1>>logs/sweep_fever_0.2_0.9.stdout 2>>logs/sweep_fever_0.2_0.9.stderr
	date >> logs/sweep_fever_0.2_0.9.stdout
	date >> logs/sweep_fever_0.2_0.9.stderr
done

popd
