python preprocess.py --preprocess mc

wait

python mc/run_swag_no_trainer.py \
    --tokenizer_name bert-base-chinese \
    --model_name_or_path hfl/chinese-bert-wwm-ext \
    --output_dir  checkpoint/mc \
    --train_file data/train_mc.json \
    --validation_file data/valid_mc.json \
    --pad_to_max_length \
    --max_seq_length 512 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --report_to wandb