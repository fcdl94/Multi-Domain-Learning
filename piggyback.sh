#!/bin/bash
source activate MultiDomainLearning
python main.py --dataset vgg-flowers --net piggyback --pretrained pretrained_57.pth --prefix tmp --old 1 --output "out.txt"
python main.py --dataset cifar100 --net piggyback --pretrained tmp_checkpoint.pth --prefix tmp --output "out.txt"
python main.py --dataset daimlerpedcls --net piggyback --pretrained tmp_checkpoint.pth --prefix tmp --output "out.txt"
python main.py --dataset dtd --net piggyback --pretrained tmp_checkpoint.pth --prefix tmp --output "out.txt"
python main.py --dataset gtsrb --net piggyback --pretrained tmp_checkpoint.pth --prefix tmp --output "out.txt"
python main.py --dataset omniglot --net piggyback --pretrained tmp_checkpoint.pth --prefix tmp --output "out.txt"
python main.py --dataset svhn --net piggyback --pretrained tmp_checkpoint.pth --prefix tmp --output "out.txt"
python main.py --dataset ucf101 --net piggyback --pretrained tmp_checkpoint.pth --prefix tmp --output "out.txt"
python main.py --dataset aircraft --net piggyback --pretrained tmp_checkpoint.pth --prefix final_piggyback --output "out.txt"
