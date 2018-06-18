# Manager of the training procedure
from __future__ import print_function
import argparse
import torch
import network
from torchvision import transforms
from torch.utils.data.dataset import ConcatDataset
import torch.utils.data as data
from PIL import Image
import os
import os.path

################## DATA LOADER PART #####################################

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir):
    images = []
    #dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


def make_dataset_with_class(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class PathImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, dataset=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        if dataset:
            self.dataset = dataset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

    def __len__(self):
        return len(self.imgs)


################## ANNOTATION FUNCTION PART #####################################

DATASETS = ['aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb', 'imagenet12', 'omniglot', 'svhn',
            'ucf101','vgg-flowers']
OFFSETS = [206667, 250000, 229400, 203760, 239209, 1481167, 224345, 273257, 209537, 202040]
CLASSES = {'imagenet12': 1000, 'aircraft': 100, 'cifar100': 100, 'daimlerpedcls': 2, 'dtd': 47, 'gtsrb': 43,
           'omniglot': 1623, 'svhn': 10, 'ucf101': 101, 'vgg-flowers': 102}
# Dataset to index dict
DICT_NAMES = {'imagenet12': 0, 'aircraft': 1, 'cifar100': 2, 'daimlerpedcls': 3, 'dtd': 4, 'gtsrb': 5, 'omniglot': 6,
              'svhn': 7, 'ucf101': 8, 'vgg-flowers': 9}

DICT_OFFSETS = {}
for D, O in zip(DATASETS, OFFSETS):
    DICT_OFFSETS[D] = O


def annotate(model, loaders, conversion, output_text):
    results = "["
    model.cuda()
    model.eval()

    for i, loader in enumerate(loaders):
        annotated = 0

        offset = (10 ** 7) * (i + 1)

        for data, target, path in loader:
            data, target = data.cuda(), target.cuda()
            bs, ncrops, c, h, w = data.size()
            # Qui va aggiunto il tuo modo di fare forward per dataset
            model.set_index(DICT_NAMES[loader.dataset])
            output_10 = model(data.view(-1, c, h, w))
            output = output_10.view(bs, ncrops, -1).mean(1)

            pred = output.data.max(1, keepdim=True)[1]
            annotated += (len(path))

            for j in range(len(path)):
                results += "{\"image_id\":" + str(DICT_OFFSETS[DATASETS[i]] + offset + int(
                    path[j].split('/')[-1].split('.')[0])) + ", \"category_id\": " + str(
                    offset + int(conversion[i][int(pred[j])])) + "},"

        print(annotated)

        if output_text:
            with open(output_text.split('.json')[0] + "_" + str(DATASETS[i - 1]) + '.json', 'w') as file_out:
                file_out.write(results)

    results = results[:-1] + "]"

    return results


################################### ANNOTATION SET UP ###############################################
parser = argparse.ArgumentParser(description='TASER model for VDA challenge')
parser.add_argument('--pretrained', type=str, default="pretrained_new_last.pth.tar",
                    help='Whether to use a pretrained model.')
parser.add_argument('--output', type=str, default="results_2_final.json",
                    help='Where to store the results')
args = parser.parse_args()

mirroring = {'imagenet12': 0, 'aircraft': 1, 'cifar100': 1, 'daimlerpedcls': 1, 'dtd': 0, 'gtsrb': 0, 'omniglot': 0,
             'svhn': 0, 'ucf101': 1, 'vgg-flowers': 1}
sorting = {'imagenet12': 0, 'aircraft': 0, 'cifar100': 0, 'daimlerpedcls': 0, 'dtd': 0, 'gtsrb': 1, 'omniglot': 1,
           'svhn': 0, 'ucf101': 0, 'vgg-flowers': 1}
conversion = [[] for i in range(10)]

loaders = []
model = network.piggyback_net(CLASSES.values(), args.pretrained)

# Order as strings, when needed also alphabetically
for i, d in enumerate(DATASETS):
    print(d)
    if d == "svhn":
        conversion[i] = [str(j) for j in range(2, CLASSES[d] + 2)]
        conversion[i][9] = '1'
        print(conversion[i])
    else:
        conversion[i] = [str(j) for j in range(1, CLASSES[d] + 1)]

    if sorting[d]:
        print('sorting....')
        conversion[i].sort()
        order = [int(a) for a in conversion[i]]
        for j in range(len(conversion[i])):
            conversion[i][order[j] - 1] = str(j + 1)

for d in DATASETS:
    print(d)

    # If mirroring use 10 crops
    if mirroring[d]:
        data_transform = transforms.Compose([
            transforms.TenCrop(64),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(
                crop) for crop in crops]))
        ])
    else:
        data_transform = transforms.Compose([
            transforms.FiveCrop(64),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(
                crop) for crop in crops]))
        ])

    dataset = PathImageFolder(root='/home/lab2atpolito/FabioDatiSSD/' + d + '/test', transform=data_transform, dataset=d)

    loaders.append(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False))

results = annotate(model, loaders, conversion, args.output)

if args.output:
    with open(args.output, 'w') as file:
        file.write(results)
