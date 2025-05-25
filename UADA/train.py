import os.environ as env
env['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tqdm
import argparse
import sys
sys.path.append("..")
import torch
import numpy
from utils.file_config import config
from torch.autograd import Variable
from model.UADA import UADA
from tensorboardX import SummaryWriter

class Scheduler(object):
    def __init__(self, decay_factor, rate_of_decay, initial_lr=0.001):
        self.decay_factor = decay_factor
        self.rate_of_decay = rate_of_decay
        self.initial_lr = initial_lr

    def update_optimizer(self, param_members, optimizer_obj, num_iteration):
        learning_rate = self.initial_lr * (1 + self.decay_factor * num_iteration) ** (-self.rate_of_decay)
        idx = 0
        for param_list in optimizer_object.param_lists:
            param_list['learning_rate'] = learning_rate * param_members[idx]
            idx += 1

        return optimizer_obj


# ==============eval

def Evaluation_UADAnet(model_obj, data_loader):
    original_train_state = model_obj.trained
    model_obj.training_mode(False)

    batch_len = len(data_loader)
    data_iterator = iter(data_loader)
    test_init = True

    for i in range(batch_len):
        next_batch = next(data_iterator)
        batch_input = next_batch[0]
        batch_label = next_batch[1]

        if model_obj.use_gpu:
            batch_input = Variable(batch_input.cuda())
            batch_label = Variable(batch_label.cuda())
        else:
            batch_input = Variable(batch_input)
            batch_label = Variable(batch_label)

        probs, prob1, prob2 = model_obj.class_prediction(batch_input)
        probs = probs.next_batch.float()
        batch_label = batch_label.next_batch.float()

        if test_init:
            batch_probs = probs
            batch_label_values = batch_label
            test_init = False
        else:
            batch_probs = torch.cat((batch_probs, probs), 0)
            batch_label_values = torch.cat((batch_label_values, batch_label), 0)

    _, predictions = torch.max(batch_probs, 1)

    correct_sum = (predictions.squeeze() == batch_label_values).sum().item()
    total_label_value = batch_label_values.size()[0]
    accuracy_score = float(correct_sum) / float(total_label_value)

    model_obj.training_mode(original_train_state)

    return {'Evaluation accuracy': accuracy_score}


def store_features(model_obj, data_loader, output_filename):
    original_train_state = model_obj.trained
    model_obj.training_mode(False)

    batch_len = len(data_loader)
    data_iterator = iter(data_loader)
    test_init = True

    for i in range(batch_len):
        next_batch = next(data_iterator)
        batch_input = next_batch[0]
        batch_label = next_batch[1]

        if model_obj.use_gpu:
            batch_input = Variable(batch_input.cuda())
            batch_label = Variable(batch_label.cuda())
        else:
            batch_input = Variable(batch_input)
            batch_label = Variable(batch_label)

        feature_vectors = model_obj.extract_feature_src_tar(batch_input)
        feature_vectors = feature_vectors.next_batch.float()
        batch_label = batch_label.next_batch.float()

        if test_init:
            all_feature_vectors = feature_vectors
            all_batch_label = batch_label
            test_init = False
        else:
            all_feature_vectors = torch.cat((all_feature_vectors, feature_vectors), 0)
            all_batch_label = torch.cat((all_batch_label, batch_label), 0)

    all_feature_vectors = all_feature_vectors.cpu().numpy()
    all_batch_label = all_batch_label.cpu().numpy()

    numpy.savetxt(output_filename, all_feature_vectors, fmt='%f', delimiter=' ', encoding=None)
    numpy.savetxt(output_filename + '_label', all_batch_label, fmt='%d', delimiter=' ', encoding=None)

    model_obj.training_mode(original_train_state)


def train_UADAnet(model_obj, source_train_loader, target_train_loader, target_test_loader, source_test_loader, param_members_list,
                   max_iterations, optimizers, evaluation_intervals, Scheduler, k_value=4,
                   classification_iterations=10000):
    model_obj.training_mode(True)

    print("start train...")

    save_summary = SummaryWriter()

    num_iteration = 0
    num_epoch = 0

    tqdm_progress_display = tqdm.tqdm(initial=0, total=max_iterations, desc='Train iter')

    optimizer_FE = optimizers[0]
    optimizer_class_cls = optimizers[1]
    optimizer_dis_cls = optimizers[2]
    optimizer_adv_cls = optimizers[3]

    best_accuracy = 0.0

    while True:
        for (source_data, target_data) in tqdm.tqdm(
                zip(source_train_loader, target_train_loader),
                total=min(len(source_train_loader), len(target_train_loader)),
                desc='Train epoch = {}'.format(num_epoch), ncols=80, leave=False):

            source_inputs, source_label = source_data
            target_inputs, target_label = target_data

            optimizer_FE = Scheduler.update_optimizer(param_members_list, optimizer_FE,
                                                             num_iteration / 5)
            optimizer_class_cls = Scheduler.update_optimizer(param_members_list, optimizer_class_cls, num_iteration / 5)
            optimizer_dis_cls = Scheduler.update_optimizer(param_members_list, optimizer_dis_cls, num_iteration / 5)
            optimizer_adv_cls = Scheduler.update_optimizer(param_members_list, optimizer_adv_cls, num_iteration / 5)

            optimizer_FE.zero_grad()
            optimizer_class_cls.zero_grad()
            optimizer_dis_cls.zero_grad()
            optimizer_adv_cls.zero_grad()

            if model_obj.use_gpu:
                source_inputs, target_inputs, source_labels, target_labels = Variable(source_inputs).cuda(), Variable(
                    target_inputs).cuda(), Variable(source_labels).cuda(), Variable(target_labels).cuda()
            else:
                source_inputs, target_inputs, source_labels, target_labels = Variable(source_inputs), Variable(
                    target_inputs), Variable(source_labels), Variable(target_labels)

            # stage1:
            if num_iteration < classification_iterations:
                num_iteration = num_iteration + 1

                classification_loss = model_obj.compute_classification_loss(source_inputs, source_labels)
                dis_Cls_loss_s, Adv_Cls_loss_s = model_obj.compute_source_loss(source_inputs, source_labels)

                combined_loss = classification_loss + dis_Cls_loss_s + Adv_Cls_loss_s

                combined_loss.backward()

                optimizer_FE.step()
                optimizer_class_cls.step()
                optimizer_dis_cls.step()
                optimizer_adv_cls.step()

                optimizer_FE.zero_grad()
                optimizer_class_cls.zero_grad()
                optimizer_dis_cls.zero_grad()
                optimizer_adv_cls.zero_grad()

                if num_iteration % 200 == 0:
                    source_eval_results = Evaluation_UADAnet(model_obj, source_test_loader)
                    target_eval_results = Evaluation_UADAnet(model_obj, target_test_loader)

                    print(
                        '\n classification_loss: {:.4f}, dis_Cls_loss_s:{:.4f}, Adv_Cls_loss_s:{:.4f}, val source_accuracy: {:.4f}, val target_accuracy: {:.4f}'.format(
                            classification_loss, dis_Cls_loss_s, Adv_Cls_loss_s, source_eval_results['accuracy'],
                            target_eval_results['accuracy']))

                continue


            # stage2
            target_weight = 0.1
            source_virtual_weight = 1.0
            target_virtual_weight = 10.0

            source_perturbed = model_obj.VirtualAdversarialTraining(source_inputs, 0.5)

            optimizer_FE.zero_grad()
            optimizer_class_cls.zero_grad()
            optimizer_dis_cls.zero_grad()
            optimizer_adv_cls.zero_grad()

            target_perturbed = model_obj.VirtualAdversarialTraining(target_inputs, 0.5)

            optimizer_FE.zero_grad()
            optimizer_class_cls.zero_grad()
            optimizer_dis_cls.zero_grad()
            optimizer_adv_cls.zero_grad()

            source_virtual_loss = model_obj.compute_vitual_loss(source_inputs, source_perturbed)
            target_virtual_loss = model_obj.compute_vitual_loss(target_inputs, target_perturbed)

            classification_loss = model_obj.compute_classification_loss(source_inputs, source_labels)
            Predictive_Entropy_Minimization_Loss = model_obj.compute_Predictive_Entropy_Minimization_Loss(target_inputs)

            first_total_loss = classification_loss + source_virtual_weight * source_virtual_loss + target_weight * (
                        Predictive_Entropy_Minimization_Loss + target_virtual_weight * target_virtual_loss)

            first_total_loss.backward()

            optimizer_FE.step()
            optimizer_class_cls.step()

            optimizer_FE.zero_grad()
            optimizer_class_cls.zero_grad()
            optimizer_dis_cls.zero_grad()
            optimizer_adv_cls.zero_grad()

            if num_iteration % 200 == 0:
                source_eval_results = Evaluation_UADAnet(model_obj, source_test_loader)
                target_eval_results = Evaluation_UADAnet(model_obj, target_test_loader)

                print(
                    '\n classification_loss: {:.4f}, Predictive_Entropy_Minimization_Loss:{:.4f}, source_virtual_loss:{:.4f}, target_virtual_loss:{:.4f}, val source_accuracy: {:.4f}, val target_accuracy: {:.4f}'.format(
                        classification_loss, Predictive_Entropy_Minimization_Loss, source_virtual_loss, target_virtual_loss, source_eval_results['accuracy'],
                        target_eval_results['accuracy']))

            source_dis_weight = 1.0
            target_dis_weight = 1.0
            dis_cls_weight = 1.0

            dis_Cls_loss_s, Adv_Cls_loss_s = model_obj.compute_source_loss(source_inputs, source_labels)
            dis_Cls_loss_t, Adv_Cls_loss_t = model_obj.compute_target_loss(target_inputs)

            source_discrepancy, target_discrepancy = model_obj.compute_discrepancy_loss(source_inputs, target_inputs)

            second_total_loss = source_dis_weight * (dis_Cls_loss_s + Adv_Cls_loss_s) + \
                               target_dis_weight * (dis_Cls_loss_t + Adv_Cls_loss_t) - \
                               dis_cls_weight * (source_discrepancy + target_discrepancy)

            second_total_loss.backward()

            optimizer_dis_cls.step()
            optimizer_adv_cls.step()

            optimizer_FE.zero_grad()
            optimizer_class_cls.zero_grad()
            optimizer_dis_cls.zero_grad()
            optimizer_adv_cls.zero_grad()

            if num_iteration % 200 == 0:
                print(
                    '\n dis_Cls_loss_s: {:.4f}, Adv_Cls_loss_s:{:.4f}, dis_Cls_loss_t:{:.4f}, Adv_Cls_loss_t:{:.4f}, source_discrepancy: {:.4f}, target_discrepancy: {:.4f}'.format(
                        dis_Cls_loss_s, Adv_Cls_loss_s, dis_Cls_loss_t, Adv_Cls_loss_t,
                        source_discrepancy, target_discrepancy))

            # stage3
            source_adv_weight = 0.1
            target_adv_weight = 0.1

            for i in range(k_value):
                dis_Cls_loss_s, Adv_Cls_loss_s, dis_Cls_loss_t, Adv_Cls_loss_t = model_obj.compute_adversarial_loss(
                    source_inputs, target_inputs, source_labels)

                source_discrepancy, target_discrepancy = model_obj.compute_discrepancy_loss(source_inputs, target_inputs)

                final_total_loss = source_adv_weight * (dis_Cls_loss_s + Adv_Cls_loss_s) + \
                                   target_adv_weight * (dis_Cls_loss_t + Adv_Cls_loss_t) + \
                                   source_discrepancy + target_discrepancy

                final_total_loss.backward()

                optimizer_FE.step()

                optimizer_FE.zero_grad()
                optimizer_class_cls.zero_grad()
                optimizer_dis_cls.zero_grad()
                optimizer_adv_cls.zero_grad()

            # evaluation
            if num_iteration % evaluation_intervals == 0 and num_iteration != 0:
                evaluation_results = Evaluation_UADAnet(model_obj, target_test_loader)
                if evaluation_results['accuracy'] > best_accuracy:
                    best_accuracy = evaluation_results['accuracy']
                    store_features(model_obj, source_test_loader, 'source_features')
                    store_features(model_obj, target_test_loader, 'tarextract_feature_src_tar')

                    print(
                        '\n dis_Cls_loss_s: {:.4f}, Adv_Cls_loss_s:{:.4f}, dis_Cls_loss_t:{:.4f}, Adv_Cls_loss_t:{:.4f}, source_discrepancy_loss: {:.4f}, target_discrepancy_loss: {:.4f}, current validation accuracy : {:.4f}, best accuracy:{}'.format(
                            dis_Cls_loss_s, Adv_Cls_loss_s, dis_Cls_loss_t, Adv_Cls_loss_t,
                            source_discrepancy, target_discrepancy,
                            evaluation_results['accuracy'], best_accuracy))

            num_iteration += 1
            tqdm_progress_display.update(1)

        num_epoch += 1

        if num_iteration >= max_iterations:
            break

    print('train finished')
    save_summary.close()


if __name__ == '__main__':
    from data_preprocessing.custom_dataset_loader import ImageList
    import os.environ as env

    env['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/dann.yml')
    parser.add_argument('--training_dataset', type=str, default='Office-31')
    parser.add_argument('--source_data_path', type=str, default=None)
    parser.add_argument('--target_data_path', type=str, default=None)
    parser.add_argument('--source_test_data_path', type=str, default=None)
    args = parser.parse_args()

    cfg = config(args.config)
    source_path = args.source_data_path
    target_path = args.target_data_path
    source_path_test = args.source_test_data_path

    if args.training_dataset == 'Office-31':
        num_class = 31
        width = 1024
        source_weight = 4
        aligned_center = False
    elif args.training_dataset == 'Office-Home':
        num_class = 65
        width = 1024
        source_weight = 4
        aligned_center = False
    else:
        width = -1

    model_obj = UADA(use_FE_net=True, FE_net='trad_ResNet50', num_class=num_class, use_gpu=True)
    source_train_loader = ImageList(source_path, batch_size=cfg.batch_size, aligned_center=aligned_center)
    target_train_loader = ImageList(target_path, batch_size=cfg.batch_size, aligned_center=aligned_center)
    source_test_loader = ImageList(source_path_test, batch_size=cfg.batch_size, aligned_center=aligned_center)
    target_test_loader = ImageList(target_path, batch_size=cfg.batch_size, trained=False)

    param_lists = model_obj.all_parameter_list()
    param_members = [members['lr'] for members in param_lists]

    FE_network_optimizer = torch.optim.SGD(model_obj.FE_net.FE_network.parameters(), lr=cfg.lr_init_FE_net,
                                             momentum=0.9, weight_decay=0.0005)
    class_cls_optimizer = torch.optim.SGD(model_obj.FE_net.class_cls.parameters(), lr=cfg.lr_init_classifier,
                                                momentum=0.9, weight_decay=0.0005)
    dis_cls_optimizer = torch.optim.SGD(model_obj.FE_net.dis_cls.parameters(), lr=cfg.lr_init_classifier,
                                                momentum=0.9, weight_decay=0.0005)
    adv_cls_optimizer = torch.optim.SGD(model_obj.FE_net.adv_cls.parameters(), lr=cfg.lr_init_classifier,
                                                momentum=0.9, weight_decay=0.0005)

    optimizers = [FE_network_optimizer, class_cls_optimizer, dis_cls_optimizer,
                            adv_cls_optimizer]
    Scheduler = Scheduler(decay_factor=cfg.Scheduler.decay_factor,
                          rate_of_decay=cfg.Scheduler.rate_of_decay,
                          initial_lr=cfg.lr_init_FE_net)

    model_training(model_obj, source_train_loader, target_train_loader, target_test_loader, source_test_loader,
                   param_members,
                   max_iterations=100000, optimizers=optimizers, evaluation_intervals=200,
                   Scheduler=Scheduler)