#!/bin/bash
source activate MultiDomainLearning
python main.py --dataset vgg-flowers --net resnet --pretrained pretrained_57.pth --prefix tmp --old 1 --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset cifar100 --net resnet --pretrained tmp_checkpoint.pth --prefix tmp --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset daimlerpedcls --net resnet --pretrained tmp_checkpoint.pth --prefix tmp --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset dtd --net resnet --pretrained tmp_checkpoint.pth --prefix tmp --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset gtsrb --net resnet --pretrained tmp_checkpoint.pth --prefix tmp --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset omniglot --net resnet --pretrained tmp_checkpoint.pth --prefix tmp --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset svhn --net resnet --pretrained tmp_checkpoint.pth --prefix tmp --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset ucf101 --net resnet --pretrained tmp_checkpoint.pth --prefix tmp --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
python main.py --dataset aircraft --net resnet --pretrained tmp_checkpoint.pth --prefix final_fe --output "outfe.txt" --frozen 1 --mirror 0 --scaling 0
