#!/usr/bin/env bash
source activate MultiDomainLearning
python main.py --dataset vgg-flowers --net piggyback --pretrained models/pretrained_57.pth --prefix models/tmpvgg --old 1 --output "out.txt" --visdom="check"
python main.py --dataset dtd --net piggyback --pretrained models/tmpvgg_checkpoint.pth --prefix models/tmpdtd --output "out.txt" --visdom="check"
python main.py --dataset ucf101 --net piggyback --pretrained models/tmpdtd_checkpoint.pth --prefix models/tmpufc --output "out.txt" --visdom="check"
python main.py --dataset vgg-flowers --net piggyback --pretrained models/tmpvgg_checkpoint.pth --test 1
python main.py --dataset vgg-flowers --net piggyback --pretrained models/tmpdtd_checkpoint.pth --test 1
python main.py --dataset vgg-flowers --net piggyback --pretrained models/tmpufc_checkpoint.pth --test 1
