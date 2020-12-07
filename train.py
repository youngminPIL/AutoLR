# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,  transforms
from util.weight_variation_AutoLR import *
import time
import os
from model_AutoLR import ft_net, ft_net_4_1
from test_embedded import Get_test_results_single
from tensorboard_logger import configure, log_value
import json
import copy



######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--batchsize', default=40, type=int, help='batchsize')
parser.add_argument('--dataset', default='CUB-200', type=str, help='dataset')
parser.add_argument('--max_f', default=0.4, type=float, help='max_f')
parser.add_argument('--min_f', default=2, type=float, help='min_f')


opt = parser.parse_args()


str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

if opt.dataset == 'CUB-200':
    data_dir = '/home/ro/FG/CUB_RT/pytorch'
elif opt.dataset == 'Cars-196':
    data_dir = '/home/ro/FG/STCAR_RT/pytorch'

scale = 1000000
gamma = 0.2
cls_lr = 0.01
thr_score = 0.94

e_drop = 40
e_end = 50
mlast = 3

desired_weva_set = []
min_factor = 2
max_factor = 0.4

conv1_factor = 0.5
strict = False
save_few_epoch = False
dir_name = '/data/ymro/AAAI2021/reproduce'

configure(dir_name)
print(dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
# gpu_ids[0] = opt.gpu_ids
print(gpu_ids[0])
# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
init_resize = (256,256)
resize = (224, 224)
transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(resize, interpolation=3),
    #transforms.RandomCrop(resize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(init_resize, interpolation=3),  # Image.BICUBIC
    transforms.CenterCrop(resize),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'test': transforms.Compose(transform_val_list),
}


image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                               data_transforms['train'])
image_datasets['test'] = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                               data_transforms['test'])

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batchsize, shuffle=True, num_workers=8)

dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=opt.batchsize, shuffle=False, num_workers=8)



dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))



def train_1epoch(phase, modelB, optimizer, epoch, trial):
    modelA = copy.deepcopy(modelB)

    running_loss = 0.0
    running_corrects = 0

    for data in dataloaders[phase]:
        # get the inputs
        inputs, labels = data
        # print(inputs.shape)
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        f, outputs = modelB(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)

    weva = compute_weight_variation(modelA, modelB)

    return modelB, optimizer, weva, running_loss, running_corrects

def weva2index(weva):
    weva_index = [weva.index(x) for x in sorted(weva)]
    return weva_index

def isSort(weva):
    weva_index = weva2index(weva)
    score = get_score(weva_index)
    return score


def get_score(A):
    diff = 0.
    for index, element in enumerate(A):
        diff += abs(index - element)
    return 1.0 - diff / len(A) ** 2 * 2

def adjustLR(optimizer, weva_table, lr_table, score, n_epoch):
    now_weva = weva_table[-1][1:-3]
    now_lr = lr_table[-1][1:-1]
    if len(weva_table) <= 1:
        # Here we make desired weight variation(weva)
        if n_epoch == 0:
            max_weva = max(now_weva)*opt.max_f
            min_weva = min(now_weva)*opt.min_f
            print('Bound condition of weigh variation are Max: {:.6f} Min: {:.6f}'.format(max_weva,min_weva))
            bias = min_weva
            interval = (max_weva - min_weva) / (len(now_weva) - 1)
            desired_weva = []
            for i in range(len(now_weva)):
                desired_weva.append(bias + i * interval)
            desired_weva_set.append(desired_weva)


            target_lr = now_weva[:]
            Gvalue = []
            for i in range(len(now_lr)):
                Gvalue.append(now_weva[i]/now_lr[i])

            for i in range(len(target_lr)):
                target_lr[i] = (desired_weva[i] - now_weva[i]) /Gvalue[i] + now_lr[i]

            target_lr.append(cls_lr)
            adjust_lr = target_lr

        else:
            max_weva = max(now_weva)
            min_weva = min(now_weva)
            interval = (max_weva - min_weva) / (len(now_weva) - 1)

            desired_weva = now_weva[:]
            center = int(len(now_weva)/2)
            for i in range(center, 0, -1):
                if desired_weva[i] < desired_weva[i-1]:
                    desired_weva[i - 1] = desired_weva[i] - interval
            for i in range(center, len(now_weva)-1, 1):
                if desired_weva[i] > desired_weva[i + 1]:
                    desired_weva[i + 1] = desired_weva[i] + interval
            desired_weva_set.append(desired_weva)
            target_lr = now_weva[:]
            Gvalue = []
            for i in range(len(now_lr)):
                Gvalue.append(now_weva[i]/now_lr[i])

            for i in range(len(target_lr)):
                target_lr[i] = (desired_weva[i] - now_weva[i]) /Gvalue[i] + now_lr[i]

            target_lr.append(cls_lr)
            adjust_lr = target_lr


    else:
        desired_weva = desired_weva_set[-1]
        target_lr = now_weva[:]
        Gvalue = []
        for i in range(len(now_lr)):
            Gvalue.append(now_weva[i] / now_lr[i])

        for i in range(len(target_lr)):
            target_lr[i] = (desired_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]

        target_lr.append(cls_lr)
        adjust_lr = target_lr

    return adjust_lr

#hint
def get_lr(optimizer):
    lrs = []
    for i in range(len(optimizer.param_groups)):
        lrs.append(optimizer.param_groups[i]['lr'])
    return lrs


def optimizer_binding(optimizer, model, now_lr):

    ignored_params = list(map(id, model.model.layer2.parameters())) + list(map(id, model.model.layer3.parameters())) + \
                     list(map(id, model.model.layer4.parameters())) + list(map(id, model.model.fc.parameters())) \
                     + list(map(id, model.classifier.parameters())) + list(map(id, model.model.layer1.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer_try = optim.SGD([
        {'params': base_params, 'lr': now_lr[0]},
        {'params': model.model.layer1[0].parameters(), 'lr': now_lr[1]},
        {'params': model.model.layer1[1].parameters(), 'lr': now_lr[2]},
        {'params': model.model.layer1[2].parameters(), 'lr': now_lr[3]},
        {'params': model.model.layer2[0].parameters(), 'lr': now_lr[4]},
        {'params': model.model.layer2[1].parameters(), 'lr': now_lr[5]},
        {'params': model.model.layer2[2].parameters(), 'lr': now_lr[6]},
        {'params': model.model.layer2[3].parameters(), 'lr': now_lr[7]},
        {'params': model.model.layer3[0].parameters(), 'lr': now_lr[8]},
        {'params': model.model.layer3[1].parameters(), 'lr': now_lr[9]},
        {'params': model.model.layer3[2].parameters(), 'lr': now_lr[10]},
        {'params': model.model.layer3[3].parameters(), 'lr': now_lr[11]},
        {'params': model.model.layer3[4].parameters(), 'lr': now_lr[12]},
        {'params': model.model.layer3[5].parameters(), 'lr': now_lr[13]},
        {'params': model.model.layer4[0].parameters(), 'lr': now_lr[14]},
        # {'params': model.model.layer4[1].parameters(), 'lr': now_lr[15]},
        # {'params': model.model.layer4[2].parameters(), 'lr': now_lr[16]},
        {'params': model.classifier.parameters(), 'lr': now_lr[15]}
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)  # for CUB

    return optimizer_try



def train_model(model_pre, thr_score, criterion, optimizer_pre, drop_timing, num_epochs=25):
    since = time.time()

    phase = 'train'
    model_pre.train(True)
    weva_success = []
    lr_success = []
    ntrial_succes = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if not strict :
            if epoch >= 1 and thr_score > 0.8:
                thr_score = thr_score*0.99

        Trial_error = True

        trial = 0
        weva_table = []
        lr_table = []

        now_lr = get_lr(optimizer_pre)

        while Trial_error:
            trial = trial + 1
            model_try = copy.deepcopy(model_pre)
            optimizer_try = optimizer_binding(optimizer_pre, model_try, now_lr)
            model_try, optimizer_try, weva_try, running_loss, running_corrects = train_1epoch(phase, model_try, optimizer_try, epoch, trial)
            score = round(isSort(weva_try[1:-mlast]),3)
            if score >= thr_score:
                Trial_error = False
                model_pre = copy.deepcopy(model_try)
                if epoch == e_drop - 1:
                    for i in range(len(now_lr)):
                        now_lr[i] = now_lr[i] * gamma
                optimizer_pre = optimizer_binding(optimizer_try, model_pre, now_lr)
                weva_success.append(copy.deepcopy(weva_try))
                print_lr = get_lr(optimizer_try)
                print_lr = print_lr[1:]
                lr_success.append(get_lr(optimizer_try))
                ntrial_succes.append(trial)

                #save model
                if phase == 'train':
                    save_network(model_pre, epoch)

            else :
                Trial_error = True
                weva_table.append(weva_try)
                print_lr = get_lr(optimizer_try)
                print_lr = print_lr[1:]
                lr_table.append(get_lr(optimizer_try))

                now_lr = adjustLR(optimizer_try, weva_table, lr_table, score, epoch)
                now_lr.insert(0, now_lr[0]*conv1_factor)



            #Print current state
            running_corrects = running_corrects.float()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('trial{}, score:{}, {} Loss: {:.8f} Acc: {:.8f}'.format(trial, score,
                phase, epoch_loss, epoch_acc))

            results = Get_test_results_single(image_datasets['test'], dataloaders['test'], model_try)

            weva_try_print = weva_try[1:-3]
            weva_try_print.append(weva_try[-1])

            epoLfmt = ['{:.6f}']*(len(weva_try_print)-1)
            epoLfmt =' '.join(epoLfmt)
            values = []
            for i in range(len(weva_try_print)-1):
                values.append(weva_try_print[i])
            epoLfmt = '   WeVa :' + epoLfmt
            print(epoLfmt.format(*values))

            if Trial_error == True:
                de_weva = desired_weva_set[-1]
                epoLfmt = ['{:.6f}'] * len(de_weva)
                epoLfmt = ' '.join(epoLfmt)
                values = []
                for i in range(len(de_weva)):
                    values.append(de_weva[i])
                epoLfmt = 'desWeVa :' + epoLfmt
                print(epoLfmt.format(*values))

            epoLfmt = ['{:.6f}'] * (len(print_lr)-1)
            epoLfmt = ' '.join(epoLfmt)
            values = []
            for i in range(len(print_lr)-1):
                values.append(print_lr[i])
            epoLfmt = '     LR :' + epoLfmt
            print(epoLfmt.format(*values))

            print('test accuracy : top-1 {:.4f} top-2 {:.4f} top-4 {:.4f} top-8 {:.4f}'.format(results[0]*100,results[1]*100,results[2]*100,results[3]*100))

            if phase == 'train':
                log_value('train_loss', epoch_loss, epoch)
                log_value('train_acc', epoch_acc, epoch)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model



# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(dir_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

def save_network_trial(network, epoch_label, trial):
    save_filename = 'net_%s_%s.pth' % (epoch_label, trial)
    save_path = os.path.join(dir_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network


model = ft_net_4_1(len(class_names))


if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
ignored_params = list(map(id, model.model.layer2.parameters())) + list(map(id, model.model.layer3.parameters())) + \
                 list(map(id, model.model.layer4.parameters())) + list(map(id, model.model.fc.parameters())) \
                 + list(map(id, model.classifier.parameters())) + list(map(id, model.model.layer1.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
print(base_params)
# Observe that all parameters are being optimized



optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.001},
    {'params': model.model.layer1[0].parameters(), 'lr': 0.001},
    {'params': model.model.layer1[1].parameters(), 'lr': 0.001},
    {'params': model.model.layer1[2].parameters(), 'lr': 0.001},
    {'params': model.model.layer2[0].parameters(), 'lr': 0.001},
    {'params': model.model.layer2[1].parameters(), 'lr': 0.001},
    {'params': model.model.layer2[2].parameters(), 'lr': 0.001},
    {'params': model.model.layer2[3].parameters(), 'lr': 0.001},
    {'params': model.model.layer3[0].parameters(), 'lr': 0.001},
    {'params': model.model.layer3[1].parameters(), 'lr': 0.001},
    {'params': model.model.layer3[2].parameters(), 'lr': 0.001},
    {'params': model.model.layer3[3].parameters(), 'lr': 0.001},
    {'params': model.model.layer3[4].parameters(), 'lr': 0.001},
    {'params': model.model.layer3[5].parameters(), 'lr': 0.001},
    {'params': model.model.layer4[0].parameters(), 'lr': 0.001},
    # {'params': model.model.layer4[1].parameters(), 'lr': 0.001},
    # {'params': model.model.layer4[2].parameters(), 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': cls_lr}
], momentum=0.9, weight_decay=5e-4, nesterov=True) #for CUB


if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, thr_score, criterion, optimizer_ft, e_drop, num_epochs=e_end)



