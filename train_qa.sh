python preprocess.py --preprocess qa

wait

python qa/run_qa.py \
  --do_train \
  --do_eval \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --output_dir  checkpoint/qa \
  --train_file data/train_qa.json \
  --validation_file data/valid_qa.json \
  --cache_dir ./cache/qa \
  --per_gpu_train_batch_size 10 \
  --gradient_accumulation_steps 8 \
  --per_gpu_eval_batch_size 10 \
  --eval_accumulation_steps  8 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --warmup_ratio 0.1 \
  --overwrite_output_dir
