import model.resnet50 as resnet50
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GRL_Layer(torch.autograd.Function):
    def __init__(self, num_iter=0, max_iter=10000.0, alpha=1, high_value=0.1, low_value=0):
        self.num_iter = num_iter
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.num_iter += 1.0
        output = 1.0 * input

        return output

    def backward(self, grad_out):
        value_range = self.high_value - self.low_value
        exp_term = np.exp(-self.alpha * self.num_iter / self.max_iter)

        Sigmoid_Factor = (2.0 * value_range) / (1.0 + exp_term)

        self.Sigmoid_Scailing_Factor = np.float64(Sigmoid_Factor - value_range + self.low_value)

        return -self.Sigmoid_Scailing_Factor * grad_out

class UADAnet(nn.Module):

    def __init__(self, use_FE_net = True, FE_net = 'trad_ResNet50', num_class = 31 ,bn_dim = 1024, width =1024):
        super(UADAnet, self).__init__()
        self.use_FE_net = use_FE_net
        self.bn_dim = bn_dim
        self.num_class = num_class

        if use_FE_net:
            self.FE_network = resnet50.dict_network[FE_net]()

        self.class_Cls = nn.Sequential(
            nn.Linear(self.FE_network.output_num(), bn_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bn_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_class)
        )

        self.dis_Cls = nn.Sequential(
            nn.Linear(self.FE_network.output_num(), bn_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bn_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_class * 2)
        )

        self.Adv_Cls = nn.Sequential(
            nn.Linear(self.FE_network.output_num(), bn_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bn_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_class)
        )


        ## initialize weight

        for dep in range(3):
            self.class_Cls[dep*3.0].weight.data.normal_(0, 0.015)
            self.class_Cls[dep*3.0].bias.data.fill_(0.0)

            self.dis_Cls[dep*3.0].weight.data.normal_(0, 0.015)
            self.dis_Cls[dep*3.0].bias.data.fill_(0.0)

            self.Adv_Cls[dep*3.0].weight.data.normal_(0, 0.015)
            self.Adv_Cls[dep*3.0].bias.data.fill_(0.0)

        self.GRL = GRL_Layer()

        self.parameter_list = [{'params':self.FE_network.parameters(), 'lr':0.1},
                               {'params':self.class_Cls.parameters(), 'lr':1},
                               {'params':self.dis_Cls.parameters(), 'lr':1},
                               {'params':self.Adv_Cls.parameters(),'lr':1}]

    def forward(self, x):
        if use_FE_net:
            feature = self.FE_network(x)

        class_Cls_Feature = self.class_Cls(feature)
        dis_Cls_feature = self.dis_Cls(feature)
        Adv_Cls_feature = self.Adv_Cls(feature)

        return feature, class_Cls_Feature, dis_Cls_feature, Adv_Cls_feature


class UADA(object):
    def __init__(self, use_FE_net = False, FE_net = 'trad_ResNet50', num_class = 31, use_gpu = True):
        self.cls_net = UADAnet(use_FE_net= use_FE_net, FE_net = FE_net, num_class = num_class)
        self.use_gpu = use_gpu
        self.train_mode =False
        self.num_iter = 0
        self.num_class = num_class

        if self.use_gpu:
            self.cls_net = self.cls_net.cuda()

    def domain_gap(self, src_logits, tgt_logits):
        src_probs = torch.nn.functional.softmax(src_logits, dim=1)
        tgt_probs = torch.nn.functional.softmax(tgt_logits, dim=1)

        prob_diff = torch.sub(src_probs, tgt_probs)
        abs_diff = torch.abs(prob_diff)

        return torch.mean(abs_diff)

    def VirtualAdversarialTraining(self, input_data, rad):
        perturbation = Variable(torch.randn(input_data.data.size()).cuda())
        norm_perturbation = 1e-6 * (perturbation / torch.norm(perturbation, dim=(2,3), keepdim=True, dtype=None))
        perturbation = Variable(norm_perturbation.cuda(), requires_grad=True)

        _, class_Cls_Feature_input, _, _ = self.cls_net(input_data)
        _, class_Cls_Feature_perturb, _, _ = self.cls_net(input_data + perturbation)

        perturbation_loss = self.domain_gap(class_Cls_Feature_input, class_Cls_Feature_perturb)
        perturbation_loss.backward(retain_graph=True)

        adv_perturbation = perturbation.grad
        adv_perturbation = adv_perturbation / torch.norm(adv_perturbation)
        image_adv = input_data + rad * adv_perturbation

        return image_adv

    def compute_vitual_loss(self, input_data, input_data_vat):
        _, class_Cls_Feature_input, _, _ = self.cls_net(input_data)
        _, class_Cls_Feature_vat, _, _ = self.cls_net(input_data_vat)

        virtual_adversarial_loss = self.domain_gap(class_Cls_Feature_input, class_Cls_Feature_vat)

        return virtual_adversarial_loss

    def compute_entropy_loss(self, input_data):
        feature, class_Cls_Feature, dis_Cls_feature, Adv_Cls_feature = self.cls_net(input_data)

        class_prediction = F.softmax(class_Cls_Feature, dim=1)
        Predictive_Entropy_Minimization_Loss = - torch.mean(class_prediction * torch.log(class_prediction + 1e-6))

        return Predictive_Entropy_Minimization_Loss

    def compute_classification_loss(self, input_data, source_label):
        CE_loss = nn.CrossEntropyLoss().cuda()
        feature, class_Cls_Feature, dis_Cls_feature, Adv_Cls_feature = self.cls_net(input_data)

        classification_loss = CE_loss(class_Cls_Feature, source_label)

        return classification_loss

    def compute_source_loss(self, input_data_src, source_label):
        CE_loss = nn.CrossEntropyLoss().cuda()
        _, _, dis_Cls_feature_src, Adv_Cls_feature_src = self.cls_net(input_data_src)

        dis_Cls_loss_s = CE_loss(dis_Cls_feature_src, source_label)
        Adv_Cls_loss_s = CE_loss(Adv_Cls_feature_src, source_label)

        return dis_Cls_loss_s, Adv_Cls_loss_s

    def compute_target_loss(self, input_data_tar):
        CE_loss = nn.CrossEntropyLoss().cuda()

        _, class_Cls_Feature_tar, dis_Cls_feature_tar, Adv_Cls_feature_tar = self.cls_net(input_data_tar)

        target_label = class_Cls_Feature_tar.data.max(1)[1] + self.num_class

        dis_Cls_loss_t = CE_loss(dis_Cls_feature_tar, target_label)
        Adv_Cls_loss_t = CE_loss(Adv_Cls_feature_tar, target_label)

        return dis_Cls_loss_t, Adv_Cls_loss_t

    def compute_adversarial_loss(self, input_data_src, input_data_tar, source_label):
        CE_loss = nn.CrossEntropyLoss().cuda()
        _, _, dis_Cls_feature_src, Adv_Cls_feature_src = self.cls_net(input_data_src)

        source_label0 = source_label + self.num_class
        dis_Cls_loss_s = CE_loss(dis_Cls_feature_src, source_label0)
        Adv_Cls_loss_s = CE_loss(Adv_Cls_feature_src, source_label0)

        _, class_Cls_Feature_tar, dis_Cls_feature_tar, Adv_Cls_feature_tar = self.cls_net(input_data_tar)

        target_label = class_Cls_Feature_tar.data.max(1)[1]
        dis_Cls_loss_t = CE_loss(dis_Cls_feature_tar, target_label)
        Adv_Cls_loss_t = CE_loss(Adv_Cls_feature_tar, target_label)

        return  dis_Cls_loss_s, Adv_Cls_loss_s, dis_Cls_loss_t, Adv_Cls_loss_t

    def compute_discrepancy_loss(self, input_data_src, input_data_tar):
        _, _, dis_Cls_feature_src, Adv_Cls_feature_src = self.cls_net(input_data_src)
        _, _, dis_Cls_feature_tar, Adv_Cls_feature_tar = self.cls_net(input_data_tar)

        dis_loss_src = self.discrepancy(dis_Cls_feature_src, Adv_Cls_feature_src)
        dis_loss_tar = self.discrepancy(dis_Cls_feature_tar, Adv_Cls_feature_tar)

        return dis_loss_src, dis_loss_tar

    def class_prediction(self, input_data):
        _, class_Cls_Feature, dis_Cls_feature, Adv_Cls_feature = self.cls_net(input_data)

        return F.softmax(class_Cls_Feature, dim = 1), F.softmax(dis_Cls_feature, dim = 1), F.softmax(Adv_Cls_feature, dim = 1)

    def training_mode(self, mode):
        self.cls_net.train(mode)
        self.train_mode = mode

    def extract_feature_src_tar(self,input_data):
        feature, class_Cls_Feature, dis_Cls_feature, Adv_Cls_feature = self.cls_net(input_data)

        return feature

    def all_parameter_list(self):

        return self.cls_net.all_parameter_list
