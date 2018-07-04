#!/bin/bash
source activate MultiDomainLearning
#python main.py --dataset vgg-flowers --net piggyback --pretrained models/pretrained_57.pth --prefix piggy/vgg-tmp --old 1 --output "out.txt" >>train.txt
#python main.py --dataset cifar100 --net piggyback --pretrained piggy/vgg-tmp_checkpoint.pth --prefix piggy/cifar-tmp --output "out.txt" >>train.txt
#python main.py --dataset daimlerpedcls --net piggyback --pretrained piggy/cifar-tmp_checkpoint.pth --prefix piggy/dlp-tmp --output "out.txt" >>train.txt
#python main.py --dataset dtd --net piggyback --pretrained piggy/dlp-tmp_checkpoint.pth --prefix piggy/dtd-tmp --output "out.txt" >>train.txt
#python main.py --dataset gtsrb --net piggyback --pretrained piggy/dtd-tmp_checkpoint.pth --prefix piggy/gts-tmp --output "out.txt" >>train.txt
#python main.py --dataset omniglot --net piggyback --pretrained piggy/gts-tmp_checkpoint.pth --prefix piggy/omni-tmp --output "out.txt" >>train.txt
#python main.py --dataset svhn --net piggyback --pretrained piggy/omni-tmp_checkpoint.pth --prefix piggy/svhn-tmp --output "out.txt" >>train.txt
#python main.py --dataset ucf101 --net piggyback --pretrained piggy/svhn-tmp_checkpoint.pth --prefix piggy/ucf-tmp --output "out.txt" >>train.txt
#python main.py --dataset aircraft --net piggyback --pretrained piggy/ucf-tmp_checkpoint.pth --prefix piggy/final_piggyback --output "out.txt" >>train.txt

#cp piggy/final_piggyback_checkpoint.pth models/piggyback_v1.pth

echo "" > test.txt
python main.py --dataset imagenet12 --net piggyback --pretrained models/piggyback_v1.pth --test 1   >>test.txt
python main.py --dataset vgg-flowers --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset cifar100 --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset daimlerpedcls --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset dtd --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset gtsrb --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset omniglot --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset svhn --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset ucf101 --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
python main.py --dataset aircraft --net piggyback --pretrained models/piggyback_v1.pth --test 1  >>test.txt
