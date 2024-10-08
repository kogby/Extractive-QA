#!/bin/bash

python preprocess.py \
       --inference \
       --preprocess mc \
       --context_data "${1}" \
       --test_data "${2}"

wait

python mc/test_mc.py \
	--test_file data/test_mc.json \
	--ckpt_dir checkpoint/mc \
	--test_batch_size 32 \
	--out_file mc_test_pred.json \

wait

python qa/run_qa.py \
	--do_predict \
	--model_name_or_path checkpoint/qa \
	--output_dir qa/ \
	--test_file mc_test_pred.json \
	--pad_to_max_length \
	--max_seq_length 512 \
	--doc_stride 128 \
	--per_gpu_eval_batch_size 10 \

python submission.py \
       --output_path "${3}"

