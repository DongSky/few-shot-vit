# Meta-Baseline based meta-tuning code
Training code of SUN-M, some training configs are borrowed from the open-source mmfewshot, sincerely thanks for the contribution.

note that we provide both vanilla meta-baseline meta-tuning code as well as meta-tuning with warmup learning rate. In the readme we use warmup version as example. 

## Training
We use the CNN-enhanced ViT Visformer as example to conduct SUN-M on miniImageNet. Suppose that we have obtained Visformer checkpoint from meta-training phase.

#### Dataset
Follow the description in the test phase to setup the dataset, e.g., miniImageNet.

#### Meta-Training checkpoint

[SUN Visformer Meta-Training](https://drive.google.com/drive/folders/1qrsq2BzQlc3_gj8Of_k5zss6OSUGX6AR?usp=sharing)

Move this checkpoint into pretrain directory (or any position you like, but you need edit the load encoder path in the config files below).

#### Train
```shell
python train_meta_warmup.py --name train1shot --gpu 0,1 --config configs/train_meta_mini_visformer_1shot.yaml # 1-shot
python train_meta_warmup.py --name train5shot --gpu 0,1 --config configs/train_meta_mini_visformer_5shot.yaml # 5-shot
```

We will update more training configs gradually. 
