#!/bin/bash
pushd ../

conda init bash
conda activate pytorch
RUN_CLEAN=clean

mkdir -pv logs

if [ clean == "$RUN_CLEAN" ];
then
	rm logs/sweep_fever_0.2_0.9.*
fi

for w_exp in 0.{2..9};
do
	date
	PYTHONPATH=$PYTHONPATH:./ python3 latent_rationale/e2expred/train.py --conf_fname parameters/fever_e2expred.json --dataset_name fever --w_aux 1.0 --w_exp $w_exp --data_dir ./data --save_path results/e2expred/separate_encoder_cold_start --batch_size 13 --train_on_part 0.1
	date
done

popd
