#!/bin/bash

# if ! command -v gdown &> /dev/null
# then
#     echo "gdown 尚未安裝，開始安裝..."
#     pip3 install gdown
# fi

FILE_ID_DATA="1hmzh6OPY_6rdALhwxH0x_NakXeAG5p68"
FILE_ID_CHECKPOINTS="1eUNZpQD7Q1iJhk7m-G0X7magecacixkC"

# 定義下載檔案的輸出路徑
OUTPUT_DATA="data.zip"
OUTPUT_CHECKPOINTS="checkpoint.zip"

# 下載 data.zip
echo "正在下載 data.zip..."
gdown "https://drive.google.com/uc?id=$FILE_ID_DATA" -O $OUTPUT_DATA

# 下載 best_model_checkpoints.zip
echo "正在下載 best_model_checkpoints.zip..."
gdown "https://drive.google.com/uc?id=$FILE_ID_CHECKPOINTS" -O $OUTPUT_CHECKPOINTS

echo "下載完成！"

if [ ! -d data ]; then
	echo "正在解壓 data.zip..."
	unzip data.zip
fi

if [ ! -d best_model_checkpoints ]; then
	echo "正在解壓 checkpoint.zip..."
    unzip checkpoint.zip
fi