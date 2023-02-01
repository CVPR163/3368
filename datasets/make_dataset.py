from torchvision import datasets, transforms
import os
def make_dataset(args):
    if args.data.NAME == "CIFAR10":    
        from CDARTS.datasets.transformers.cifar import data_transforms_cifar
        train_transform, valid_transform = data_transforms_cifar(args)
        train_dataset = datasets.CIFAR10(root=args.data.DATASET_PATH, train=True, download=True, transform=train_transform)
        valid_dataset =  datasets.CIFAR10(root=args.data.DATASET_PATH, train=False, download=True, transform=valid_transform)
    elif args.data.NAME == "imagenet":
        from CDARTS.datasets.transformers.imagenet import data_transforms_imagenet
        traindir = os.path.join(args.data.DATASET_PATH, 'train')
        valdir = os.path.join(args.data.DATASET_PATH, 'val')
        train_transform, valid_transform = data_transforms_imagenet(args)
        train_dataset = datasets.ImageFolder(root=traindir,transform=train_transform)
        valid_dataset = datasets.ImageFolder(valdir, transform=valid_transform)
    return train_dataset, valid_dataset