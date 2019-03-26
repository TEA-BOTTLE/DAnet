#!/bin/sh

cd ../exper/

CUDA_VISIBLE_DEVICES=0 python val_hierarchy.py \
	--arch=vgg_HDA \
    --num_gpu=1 \
    --threshod=0.2 \
    --dataset=cub \
    --img_dir=../../dataset/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_HDA \
    --onehot=False \

