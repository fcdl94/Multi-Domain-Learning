import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from datetime import datetime
import numpy as np

import visdom


# Training settings
PATH_TO_DATASETS='/home/lab2atpolito/FabioDatiSSD/'
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

# Dataset to index dict
DICT_NAMES = {'imagenet12': 0,
            'aircraft': 1, 'cifar100': 2,
            'daimlerpedcls': 3, 'dtd': 4,
            'gtsrb': 5, 'omniglot': 6,
            'svhn': 7, 'ucf101': 8,
            'vgg-flowers': 9}


# Initialize visualization tool
vis = visdom.Visdom()

# Define the visualization environment
vis.env = "training"

# Check for CUDA usage
cuda = not NO_CUDA and torch.cuda.is_available()


def train(model, dataset_name, prefix, bn=False, mirror=True, scaling=True,
          decay=0.0, adamlr=0.0001, lr=0.001, momentum=0.9, epochs=EPOCHS):
    # Training steps:
    # Preprocessing (cropping, hor-flipping, resizing) and get data
    # Initialize data processing threads
    workers = WORKERS if cuda else 0

    data_transform = get_data_transform(mirror, scaling)

    dataset = datasets.ImageFolder(root=PATH_TO_DATASETS + '/' + dataset_name + '/train', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)

    # Build the test loader
    # (note that more complex data transforms can be used to provide better performances e.g. 10 crops)
    data_transform = get_data_transform(False, False)

    dataset = datasets.ImageFolder(root=PATH_TO_DATASETS + '/' + dataset_name + '/val', transform=data_transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)

    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.parameters())
    # set optimizer
    optimizerA = optim.Adam(base_params, lr=adamlr, weight_decay=decay)
    optimizerB = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    schedulerA = optim.lr_scheduler.StepLR(optimizerA, STEP)
    schedulerB = optim.lr_scheduler.StepLR(optimizerB, STEP)
    scheduler = MultipleOptimizer(schedulerA, schedulerB)
    optimizer = MultipleOptimizer(optimizerA, optimizerB)

    # set loss function
    cost_function = nn.CrossEntropyLoss()
    #Set the model index
    model.set_index(DICT_NAMES[dataset_name])

    print("--------PARAMETERS----------")
    for name, param in model.named_parameters():
        print(name + " requires grad " + str(param.requires_grad))
    print("--------          ----------")
    print("cuda:" + str(cuda))

    if cuda:
        model = model.cuda()

    # Initialize the lists needed for visualization, plus window offset for the graphs
    iters = []
    losses_training = []
    losses_test = []
    accuracies_test = []
    win_offset = DICT_NAMES[dataset_name] * 3

    # perform training epochs time
    loss_epoch_min = -1
    for epoch in range(1, epochs + 1):
        scheduler.step()
        loss_epoch = train_epoch(model, epoch, train_loader, optimizer, cost_function, bn)
        if loss_epoch_min == -1:
            loss_epoch_min = loss_epoch
        result = test_epoch(model, test_loader, dataset_name, cost_function)

        accuracies_test.append(result[0])
        losses_test.append(result[1])
        losses_training.append(loss_epoch)
        iters.append(epoch)

        print('Train Epoch: {} \tTrainLoss: {:.6f} \tTestLoss: {:.6f}\tAccuracyTest: {:.6f}'.format(
            epoch, loss_epoch, result[1], result[0]))

        # Print results
        vis.line(
            X=np.array(iters),
            Y=np.array(losses_training),
            opts={
                'title': ' Training Loss ' + dataset_name,
                'xlabel': 'iterations',
                'ylabel': 'loss'},
            name='Training Loss ' + dataset_name,
            win=0 + win_offset)
        vis.line(
            X=np.array(iters),
            Y=np.array(losses_test),
            opts={
                'title': ' Validation Loss ' + dataset_name,
                'xlabel': 'iterations',
                'ylabel': 'loss'},
            name='Validation Loss ' + dataset_name,
            win=1 + win_offset)
        vis.line(
            X=np.array(iters),
            Y=np.array(accuracies_test),
            opts={
                'title': ' Accuracy ' + dataset_name,
                'xlabel': 'iterations',
                'ylabel': 'accuracy'},
            name='Validation Accuracy ' + dataset_name,
            win=2 + win_offset)

        # Save the model
        if loss_epoch < loss_epoch_min:
            loss_epoch_min = loss_epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizerA': optimizerA.state_dict(),
                'optimizerB': optimizerB.state_dict()
            }, prefix + "_checkpoint.pth")


# Perform a single training epoch
def train_epoch(model, epoch, train_loader, optimizers, cost_function, bn=False):
    # Set the model in training mode
    model.train()

    print("Starting time of Epoch " + str(epoch) + ": " + str(datetime.now().time()))

    # If BN parameters must be frozen, freeze them
    if not bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()
    # Init holders
    losses = 0
    current = 0

    # Perform the training procedure
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move the variables to GPU
        if cuda:
            data, target = data.cuda(), target.cuda()

        # Reset the optimizers
        optimizers.zero_grad()

        # Process input
        output = model(data)

        # Compute loss and gradients
        loss = cost_function(output, target)
        loss.backward()

        # Update parameters
        optimizers.step()

        # Check for log and update holders
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{:4d}/{:4d} ({:2.0f}%)]\tAvgLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()/BATCH_SIZE))

        losses += loss.item()
        current += 1

    return losses / current


def test_epoch(model, test_loader, dataset, cost_function):
    # Put the model in eval mode
    model.eval()
    torch.set_grad_enabled(False)
    # Init holders
    test_loss = 0
    correct = 0

    # Perform the evaluation procedure
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        # Update holders
        test_loss += cost_function(output, target).item()  # sum up batch loss
        pred = torch.max(output, 1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # Check if the prediction is correct

    # Compute accuracy and loss
    total_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Dataset ' + dataset + ' : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        total_loss, correct, len(test_loader.dataset),
        accuracy))

    results = [accuracy, total_loss]

    torch.set_grad_enabled(True)
    return results


# Save the current checkpoint
def save_checkpoint(state, checkpoint):
    torch.save(state, checkpoint)


def get_data_transform(mirror, scaling):
    # Create Dataloader w.r.t. chosen transformations
    if mirror:
        if scaling:
            data_transform = transforms.Compose([
                transforms.RandomResizedCrop(IMAGE_CROP),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            data_transform = transforms.Compose([
                transforms.RandomCrop(IMAGE_CROP),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    else:
        if scaling:
            data_transform = transforms.Compose([
                transforms.RandomResizedCrop(IMAGE_CROP),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            data_transform = transforms.Compose([
                transforms.RandomCrop(IMAGE_CROP),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    return data_transform


class MultipleOptimizer(object):
    # Multi Optimizer
    #   https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/6 '''

    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()