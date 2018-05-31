
# Training settings
PATH_TO_DATASETS='....'
BATCH_SIZE =32
TEST_BATCH_SIZE=100
EPOCHS=60
STEP=45
NO_CUDA=False
IMAGE_CROP=64
LOG_INTERVAL=10
WORKERS=8

# image normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


#Training steps:
    # Preprocessing (cropping, hor-flipping, resizing)
    # set optimizer
    # set loss function
    # perform training epochs time


# Multi Optimizer
#https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/6