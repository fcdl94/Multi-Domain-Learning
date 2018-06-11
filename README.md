# Multi-Domain-Learning

This code provide two type of neural nets able to classify different type of dataset.
The accepted data-set are those included in the [decathlon challenge](https://www.robots.ox.ac.uk/~vgg/decathlon/).

There are mainly two model:
1. 28 Wide Residual network that can be used both as Feature extractor or with fine tuning.
2. A [piggyback](https://arxiv.org/abs/1801.06519) network that is similar to a 28 resnet but applies binary mask that are dataset-specific.

Many parts of the code were made by [Massimiliano Mancini](https://scholar.google.it/citations?user=bqTPA8kAAAAJ&hl=it) that helped me a lot in rearraning the code.

___

This code require to have installed:
* [Pytorch](https://pytorch.org/)
* Numpy
* [Visdom](https://github.com/facebookresearch/visdom)

And it's mandatory to have CUDA configured if you'd run it on NVIDIA GPUs