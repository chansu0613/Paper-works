import numpy as np
import torch.utils.data as utils
from .custom_dataset_loader import ImageList
from torchvision import transforms
from PIL import Image, ImageOps


class Image_resize():
    def __init__(self, input_size):
        if isinstance(size, int):
            self.input_size = (int(input_size), int(input_size))
        else:
            self.input_size = input_size

    def __call__(self, input_image):
        image_height, image_width = self.input_size

        return input_image.resize((image_height, image_width))


class Image_crop(object):

    def __init__(self, input_size, size_x, size_y):
        if isinstance(input_size, int):
            self.input_size = (int(input_size), int(input_size))
        else:
            self.input_size = input_size
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, input_image):
        image_height, image_width = self.input_size

        return input_image.crop((self.size_x, self.size_y, self.size_x + image_width, self.size_y + image_height))


def image_loader(Images_path, batch_size, resize_image=256, crop_image=224, trained=True, aligned_center=False):
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if not trained:
        start_center = (resize_image - crop_image - 1) / 2
        transformer = transforms.Compose([
            Image_resize(resize_image),
            Image_crop(crop_image, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        images = ImageList(open(Images_path).readlines(), transform=transformer)
        images_loader = utils.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        if aligned_center:
            transformer = transforms.Compose([Image_resize(resize_image),
                transforms.Scale(resize_image),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(crop_image),
                transforms.ToTensor(),
                normalize])
        else:
            transformer = transforms.Compose([Image_resize(resize_image),
                  transforms.RandomResizedCrop(crop_image),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize])
        images = ImageList(open(Images_path).readlines(), transform=transformer)
        images_loader = utils.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4)

    return images_loader

