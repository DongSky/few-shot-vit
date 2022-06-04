# SUN with DeepEMD as Meta-Tuning (SUN-D)
This code is based on [DeepEMD](https://github.com/icoz69/DeepEMD) repository, sincerely thanks for the contribution. 

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch >= version 1.1](https://pytorch.org)

- [QPTH](https://github.com/locuslab/qpth)

- [CVXPY](https://www.cvxpy.org/)

- [OpenCV-python](https://pypi.org/project/opencv-python/)

- [tensorboard](https://www.tensorflow.org/tensorboard)
## Dataset
Please click the Google Drive [link](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing) or [Baidu Drive (uk3o)](https://pan.baidu.com/s/17hbnrRhM1acpcjR41P3J0A) for downloading the 
following datasets, or running the downloading bash scripts in folder `datasets/` to download.


### MiniImageNet Dataset

It contains 100 classes with 600 images in each class, which are built upon the ImageNet dataset. The 100 classes are divided into 64, 16, 20 for meta-training, meta-validation and meta-testing, respectively.

## Pretrained Models
To reproduce the accuracy and follow this work easier, we recommend that using SUN-M to warmup your meta-training model, then conduct SUN-D for further meta-tuning. Thus you can download [this file](https://drive.google.com/file/d/1WFk_xHBCiaVLLYXPSXDKfClqjz_qOJFn/view?usp=sharing) as pretrained model, and store it in this directory.

## Training
We use miniImageNet and Visformer as example. 
```shell
python train_meta.py -deepemd grid -patch_list 2,3 -shot 1 -way 5 -solver opencv -gpu 0,1,2,3 -save_all
```

## Evaluation
Since the 5-shot evaluation is too slow, we use 2000 episodes in 1-shot evaluation and 600 episodes in 5-shot evaluation.
```shell
python eval.py -deepemd grid -patch_list 2,3 -gpu 0,1,2,3 -test_episode 2000 -shot 1
python eval.py -deepemd grid -patch_list 2,3 -gpu 0,1,2,3 -test_episode 600 -shot 5
```
