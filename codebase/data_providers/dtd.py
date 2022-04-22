import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
The data is available from https://www.robots.ox.ac.uk/~vgg/data/dtd/
"""

'''
class DTDDataProvider:

    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=200, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):

        norm_mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
        norm_std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]

        valid_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = datasets.ImageFolder(os.path.join(save_path, 'valid'), valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)

'''
from tkinter import image_names
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms


class DTDDataProvider:
    def __init__(self,image_list_file,transform,save_path=None, train_batch_size=32, test_batch_size=200, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        data_dir = "/content/dtd/images"
        image_names = []
        labels = []
        img_names = []
        with open(image_list_file, "r") as f:
            counter = 0
            for i,line in enumerate(f,start=1):
                image_name= line[:-1]
                img_names.append(image_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(counter)
                #print(i,counter)
                if (i%120 == 0):
                  #print('hello')
                  counter  = counter + 1
        norm_mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
        norm_std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]
        self.image_names = image_names
        self.img_names = img_names
        self.labels = torch.Tensor(labels)#torch.nn.functional.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=47)
        self.transform = transform
        
    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image,label ,self.img_names[index]

    def __len__(self):
        return len(self.image_names)

