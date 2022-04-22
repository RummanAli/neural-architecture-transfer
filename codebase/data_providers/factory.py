import torch
from tkinter import image_names
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms

def get_dataloader(**kwargs):
    if kwargs['dataset'] == 'imagenet':
        from codebase.data_providers.imagenet import ImagenetDataProvider
        loader_class = ImagenetDataProvider
    elif kwargs['dataset'] == 'cifar10':
        from codebase.data_providers.cifar import CIFAR10DataProvider
        loader_class = CIFAR10DataProvider
    elif kwargs['dataset'] == 'cifar100':
        from codebase.data_providers.cifar import CIFAR100DataProvider
        loader_class = CIFAR100DataProvider
    elif kwargs['dataset'] == 'cinic10':
        from codebase.data_providers.cifar import CINIC10DataProvider
        loader_class = CINIC10DataProvider
    elif kwargs['dataset'] == 'aircraft':
        from codebase.data_providers.aircraft import FGVCAircraftDataProvider
        loader_class = FGVCAircraftDataProvider
    elif kwargs['dataset'] == 'cars':
        from codebase.data_providers.cars import StanfordCarsDataProvider
        loader_class = StanfordCarsDataProvider
    elif kwargs['dataset'] == 'dtd':
        from codebase.data_providers.dtd import DTDDataProvider
        loader_class = DTDDataProvider
    elif kwargs['dataset'] == 'flowers102':
        from codebase.data_providers.flowers102 import Flowers102DataProvider
        loader_class = Flowers102DataProvider
    elif kwargs['dataset'] == 'food101':
        from codebase.data_providers.food import Food101DataProvider
        loader_class = Food101DataProvider
    elif kwargs['dataset'] == 'pets':
        from codebase.data_providers.pets import OxfordIIITPetsDataProvider
        loader_class = OxfordIIITPetsDataProvider
    elif kwargs['dataset'] == 'stl10':
        from codebase.data_providers.stl10 import STL10DataProvider
        loader_class = STL10DataProvider
    else:
        raise NotImplementedError

    norm_mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
    norm_std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]

    test_dataset = loader_class(image_list_file=kwargs['image_list_file'],
                                    transform=transforms.Compose([
                                        transforms.Resize((kwargs['image_size'], kwargs['image_size']), interpolation=3),
                                        transforms.CenterCrop(kwargs['image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std),
                                        ]))

    loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=kwargs['test_batch_size'],
                             shuffle=False, num_workers=8, pin_memory=True)

    return loader
