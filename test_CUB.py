# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_stride, PCB, PCB_test, ft_net, ft_net_4_1, ft_net_stride_4_1, ft_net_4_2, \
    ft_inception_v3_net, ft_net_CUB, ft_net18_3_2, ft_net18, ft_inception_v3_7a, ft_inception_v3_7b, ft_net_4_1s, \
    ft_net_3_6, ft_net_eli3_6
from model_tri import ft_net_tricls
import random
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='/home/zzd/Market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
name = opt.name

test_dir = '/home/ro/FG/CUB_RT/pytorch'
num_class = 100
gallery_eq_query = True

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

target = '/data/ymro/CVPR2020_reproduce/RT_AUTO_CUB4_1_G2/SGD_thr94_lr3_min3_max04_ep3040_conv05_std'
efrom = 0
euntil = 39
stride_ver = 1
init_resize = (256,256) ## ESSENTIAL for cub!! #######################
resize = (224,224)
opt.batchsize = 60


feature_size = 2048 #resnet 50

gpu_ids[0] = 3
# set gpu id2
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
print(gpu_ids[0])
##################################t##################################
# Load Data
# ---------
# We will use torchvision and torch.utils.data packages for loading the
# data.
#

data_transforms = transforms.Compose([
    transforms.Resize(init_resize, interpolation=3),
    transforms.CenterCrop(resize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in ['test']}

use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        ff = torch.FloatTensor(n, feature_size).zero_()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features

def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def pairwise_similarity(x, y=None):
    if y is None:
        y = x
    # normalization
    y = normalize(y)
    x = normalize(x)
    # similarity
    similarity = torch.mm(x, y.t())
    return similarity


def get_id(img_path):
    labels = []
    for path, v in img_path:
        label = v
        labels.append(int(label))
    return labels

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def Recall_at_ks(sim_mat, data='cub', query_ids=None, gallery_ids=None):
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['jd'] = [1, 2, 4, 8]
    ks_dict['product'] = [1, 10, 100, 1000]
    ks_dict['shop'] = [1, 10, 20, 30, 40, 50]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e6)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    # Hope to be much faster  yes!!
    num_valid = np.zeros(len(k_s))
    neg_nums = np.zeros(m)
    for i in range(m):
        x = sim_mat[i]

        pos_max = np.max(x[gallery_ids == query_ids[i]])
        neg_num = np.sum(x > pos_max)
        neg_nums[i] = neg_num

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i-1]

    return num_valid / float(m)

query_path = image_datasets['test'].imgs

query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')

search = target + '/ft_ResNet50'
file_list = os.listdir(search)
print(search)
for tepoch in range(euntil+1):
    if tepoch < efrom:
        continue
    file_name = 'net_%d.pth' % tepoch
    path = search + '/' + file_name

    if stride_ver == True:
        model = ft_net_stride_4_1(num_class) #market
    else:
        model = ft_net_4_1(num_class) #market
    model = load_network_path(model, path)
    model.classifier = nn.Sequential()


    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    query_feature = extract_feature(model, dataloaders['test'])

    # Save to Matlab for check
    result = {'query_f': query_feature.numpy(), 'query_label': query_label}

    query_label = np.asarray(result['query_label'])
    gallery_label =  np.asarray(result['query_label'])

    sim_mat = pairwise_similarity(query_feature)

    if gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))

    recall_ks = Recall_at_ks(sim_mat, query_ids=query_label, gallery_ids=gallery_label, data='cub')

    result = '  '.join(['%.4f' % k for k in recall_ks])
    print('%s: %s'% (tepoch, result))



