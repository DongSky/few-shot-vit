# Few-Shot ViT

## Requirements
- PyTorch (>= 1.9)
- TorchVision
- timm (latest)
- einops
- tqdm
- numpy
- scikit-learn
- scipy
- argparse
- tensorboardx

## Update
June 4th, 2022: we upload the meta-tuning phase of SUN-D.

June 4th, 2022: we upload the teacher training code in the meta-training phase of SUN.

June 3rd, 2022: we upload the meta-tuning phase of SUN-M. 

## Pretrained Checkpoints
Currently we provide SUN-M (Visformer) trained on miniImageNet (5-way 1-shot and 5-way 5-shot), see [Google Drive](https://drive.google.com/drive/folders/1Ynf45BQqMz8XUMuVkDaj3JmoRM7jGFaJ?usp=sharing) for details.

More pretrained checkpoints coming soon. 

## Evaluate the Pretrained Checkpoints

#### Prepare data
For example, miniImageNet:

cd test\_phase

Download miniImageNet dataset from [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))

unzip the package to materials/mini-imagenet, then obtain materials/mini-imagenet with pickle files.

#### Prapare pretrained checkpoints

Download corresponding checkpoints from [Google Drive](https://drive.google.com/drive/folders/1Ynf45BQqMz8XUMuVkDaj3JmoRM7jGFaJ?usp=sharing) and store the checkpoints in test\_phase/ directory.

#### Evaluation
```shell
cd test_phase
python test_few_shot.py --config configs/test_1_shot.yaml --shot 1 --gpu 1 # for 1-shot
python test_few_shot.py --config configs/test_5_shot.yaml --shot 5 --gpu 1 # for 5-shot
```
For 1-shot, you can obtain: test epoch 1: acc=67.80 +- 0.45 (%)

For 5-shot, you can obtain: test epoch 1: acc=83.25 +- 0.28 (%)

Test accuracy may slightly vary with different pytorch/cuda versions or different hardwares

# TODO
- more checkpoints
- training code of meta-training phase and SUN-D meta tuning phase
