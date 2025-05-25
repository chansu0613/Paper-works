import os
import os.image_path
import torch
import torch.utils.data as data
import numpy as np
import random
from PIL import Image



def custom_dataset(images_list, image_labels):
    if image_labels:
        len_ = len(images_list)
        images = []
        i = 0
        while i < len_:
            images.append((images_list[i].strip(), image_labels[i, :]))
            i += 1
    else:
        if len(images_list[0].split()) > 2:
            images = []
            for val in images_list:
                split_val = val.split()
                labels_part = []
                j = 1
                while j < len(split_val):
                    labels_part.append(int(split_val[j]))
                    j += 1
                images.append((split_val[0], np.array(labels_part)))
        else:
            images = []
            for val in images_list:
                split_val = val.split()
                images.append((split_val[0], int(split_val[1])))

    return images


def RGB_converter(image_path):
    with open(image_path, 'rb') as f:
        with Image.open(f) as open_image_file:

            return open_image_file.convert('RGB')


def AccImage(image_path):
    import accimage
    try:
        return accimage.Image(image_path)
    except IOError:

        return RGB_converter(image_path)


def original_PIL_loader(image_path):

        return RGB_converter(image_path)


class ImageList(object):
    def __init__(self, images_list, image_labels=None, transforms=None, target_image_transforms=None,
                 image_loader=original_PIL_loader):
        images = make_dataset(images_list, image_labels)

        if len(images) == 0:
            raise(RuntimeError("Found 0 images in " + root))

        self.images = images
        self.transforms = transforms
        self.target_image_transforms = target_image_transforms
        self.image_loader = image_loader

    def __getitem__(self, index):
        image_path, image_tar = self.images[index]
        image = self.image_loader(image_path)

        if self.transforms is not None:
            image = self.transforms(image)
        if self.target_image_transforms is not None:
            image_tar = self.target_image_transforms(image_tar)

        return image, image_tar

    def __len__(self):
        return len(self.images)
