# Multi-Domain-Learning

This code provide two type of neural nets able to classify different type of dataset.
The accepted data-set are those included in the [decathlon challenge](https://www.robots.ox.ac.uk/~vgg/decathlon/).

There are mainly two model:
1. 28 Wide Residual network that can be used both as Feature extractor or with fine tuning.
2. A [piggyback](https://arxiv.org/abs/1801.06519) network that is similar to a 28 resnet but applies binary mask that are dataset-specific.
3. A [quantized](http://arxiv.org/abs/1805.11119v2) network that is similar in spirit to Piggyback, but adds an affine transformation to the binary mask, increasing the network capacity.

Many parts of the code were made by [Massimiliano Mancini](https://scholar.google.it/citations?user=bqTPA8kAAAAJ&hl=it) that helped me a lot in rearraning the code.

___

This code require to have installed:
* Python3
* [Pytorch](https://pytorch.org/) (at least 0.4.0)
* Numpy (I'm using 1.16.2, maybe it works also with older versions)
* [Visdom](https://github.com/facebookresearch/visdom) (I'm using 0.1.8.8)

And it's mandatory to have CUDA configured if you'd run it on NVIDIA GPUs

___

Commands to run the code:

<code>
python main.py [Arguments]
</code>

The most important arguments are:
- net <The method to use, options: resnet (fine-tuning), [piggyback](http://arxiv.org/abs/1801.06519v2), [quantized](http://arxiv.org/abs/1805.11119v2)>
- pretrained <The path to pretrained model, if not added, it will start from scratch>
- dataset <The dataset that you want to train, options: d_names = 'imagenet12','cifar100','daimlerpedcls','dtd','gtsrb','omniglot','svhn','ucf101','vgg-flowers'>

To see the other parameter run:
'python main.py -h'
Remeber to modify the path to the dataset in training.py with the location of the dataset in your file system.

___

Check out the project [RobotChallenge](https://github.com/fcdl94/RobotChallenge) to see the implementation of the parallel and residual adapters of Rebuffi [1](https://arxiv.org/abs/1705.08045) [2](https://arxiv.org/pdf/1803.10082).

If you have any question or suggestion, please write to me an [email](f.cermelli94@gmail.com)


