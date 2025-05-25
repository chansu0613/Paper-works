import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable


# Using traditional and verified networks for input stage

class trad_ResNet50(nn.Module):
    def __init__(self):
        super(trad_ResNet50, self).__init__()
        traditional_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = traditional_resnet50.conv1
        self.bn1 = traditional_resnet50.bn1
        self.relu = traditional_resnet50.relu
        self.maxpool = traditional_resnet50.maxpool
        self.layer1 = traditional_resnet50.layer1
        self.layer2 = traditional_resnet50.layer2
        self.layer3 = traditional_resnet50.layer3
        self.layer4 = traditional_resnet50.layer4
        self.avgpool = traditional_resnet50.avgpool
        self.__in_features = traditional_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def output_num(self):

        return self.__in_features


dict_network = {"ResNet50": trad_ResNet50}