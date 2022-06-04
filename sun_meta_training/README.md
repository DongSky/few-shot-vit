# Meta-Training Phase

## Dataset Preparation

Follow the teacher training procedure of meta-training phase to setup the datasets, e.g., miniImageNet. 

## Teacher Model
Download the teacher model (e.g., Visformer pretrained on miniImageNet train set) from [this link](https://drive.google.com/drive/folders/1v4_R1HCMIUyLb81DaJxSwAFcpHKfdEQH?usp=sharing), and store it into the pretrain directory. 

## Training
We use Visformer and miniImageNet as example. 
```shell
python offline.py --gpu 0,1 --name metatraining --config configs/offline_tl_visformer_k5_800epoch.yaml
```
You can obtain the checkpoints in the save/metatraining directory. 