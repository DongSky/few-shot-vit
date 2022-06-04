# SUN teacher Training Code

## Data prepration
Same with the test phase, we also setup the miniImageNet directory with gitkeep file.

## Training
We use Visformer and miniImageNet as example. 
```shell
python train_classifier.py --name trainteacher --gpu 0,1 --config configs/train_classifier_mini_visformer_300epoch.yaml
```
The corresponding checkpoints will be stored in the save/trainteacher.

Note that if you use docker-based training platform (e.g., Kubernetes), edit the ensure path function in utils, such that the training code can be executed automatically without manually check existing save path. 