# Title:Diffusion Model with Multi-layer Wavelet Transform for Low-Light Image Enhancement

## Pipline
![image]([image\Pipeline.png](https://github.com/lalalulao/LWT-Diff/blob/master/image/Compare.png))
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

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth). 
