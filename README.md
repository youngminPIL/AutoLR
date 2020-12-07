## AutoLR: Layer-wise Pruning and Auto-tuning of Learning Rates in Fine-tuning of Deep Networks

Official Pytorch implementation of paper:

[AutoLR: Layer-wise Pruning and Auto-tuning of Learning Rates in Fine-tuning of Deep Networks](https://arxiv.org/abs/2002.06048) (AAAI 2021).

[Project page](https://sites.google.com/view/youngmin-ro-vision/home/acfn-1?authuser=0)



## Environment
Python 3.6, Pytorch 0.4.1, Torchvision, tensorboard


## Train 
Default setting:
- Architecture: ResNet-50
- Dataset: CUB2011 or Cars-196 retrieval
- Batch size: 40
- Image size: 224X224


### prepare
The dataset path should be changed to your own path.

CUB2011-200 dataset are available on https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view

Cars-196 dataset are available on https://ai.stanford.edu/~jkrause/cars/car_dataset.html

```
prepare_cub.py 
```

### train network on the each periods. 

Train model in period 1. This is a baseline of our algorithm. 

The dataset path(data_dir='/home/ro/FG/CUB_200_2011/pytorch') should be changed to your own path.


```
train_CUB.py --dataset CUB-200
```

In the case of Cars-196 retrieval dataset training, 

```
train_CUB.py --dataset Cars-196
```





## Citation

```
@inproceedings{AutoLR_ro,
	title = {Layer-wise Pruning and Auto-tuning of Layer-wise Learning Rates in Fine-tuning of Deep Networks
},
	author = {Youngmin Ro, Jin Young Choi},
	booktitle = {Arxiv},
	year = {2019}
}
```
Youngmin Ro, Jin Young Choi, 
"Layer-wise Pruning and Auto-tuning of Layer-wise Learning Rates in Fine-tuning of Deep Networks", CoRR, 2020.



