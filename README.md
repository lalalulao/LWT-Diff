# Title:Diffusion Model with Multi-layer Wavelet Transform for Low-Light Image Enhancement

## Pipline
![image](https://github.com/lalalulao/MWT-Diff/blob/origin/image/Figure1.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, You need to modify datasets/dataset.py slightly for your environment, and then run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on LOL Dataset, run:

```eval
python evaluate.py --model-file mymodel.pth --benchmark imagenet
```
## Visual Comporison

![image](https://github.com/lalalulao/MWT-Diff/blob/origin/image/Figure2.png)

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth). 
