# Extractive-QA
2024 Applied Deep Learning Homework 1

This repository is implementation of HW1 for Applied Deep Learning course in 2024 at National Taiwan University.

## Download dataset and model checkpoint

To download the datasets and checkpoints of Multiple Choice & Question Answering models, run the following command:

```
bash ./download.sh
```

## Reproducing best result

To reproduce best result, run the following command:

Note: you can define your own argument paths if you have your own data.

```
bash ./run.sh data/context.json data/test.json pred/prediction.csv
```

## Training

### Multiple Choice

To train the multiple choice model, run the following command:

```
bash ./train_mc.sh
```

### Question Answering

To train the question answering model, run the following command:

```
bash ./train_qa.sh
```

## Implementation
Due to the lack of training resources, the model is trained using google colab PRO, with A100 GPU equipped.
