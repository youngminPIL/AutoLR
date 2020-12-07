# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import random
######


def Get_test_results_single(image_datasets, dataloaders, model, f_size=2048):
    gallery_eq_query = True
    feature_size = f_size

    query_path = image_datasets.imgs
    query_label = get_id(query_path)
    query_label = np.asarray(query_label)
    gallery_label = np.asarray(query_label)

    # Change to test mode
    model = model.train(False)

    # Extract feature
    recall_ks = []
    query_feature = extract_feature(model, dataloaders, feature_size)

    sim_mat = pairwise_similarity(query_feature)
    # sim_mat = pairwise_similarity(query_feature, gallery_feature)

    if gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))

    recall_ks.append(Recall_at_ks(sim_mat, query_ids=query_label, gallery_ids=gallery_label, data='cub'))
    # print('{:.4f}'.format(recall_ks[0]))

    return recall_ks[0]





def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


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


def extract_feature(model, dataloaders, feature_size):
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
            f, _ = model(input_img)
            f = f.data.cpu()
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features

def extract_feature_multi(model, dataloaders, f_size):
    feature_size = f_size * len(model)
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, feature_size).zero_()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            out = torch.FloatTensor()
            for j in range(len(model)):
                out_, _ = model[j](input_img)
                #out = torch.cat((out, out_.data.cpu()), 1)
                out = torch.cat((out, normalize(out_.data.cpu())), 1)
            ff = ff + out
        # norm feature
        ff = normalize(ff)
        features = torch.cat((features, ff), 0)
    return features


def extract_feature_doublehead(model, dataloaders, f_size):
    n_param = 4000
    feature_cat = torch.FloatTensor()
    features = []
    for i in range(len(dataloaders.dataset)//n_param+1):
        features.append(torch.FloatTensor())

    feature_size = f_size * 2
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, feature_size).zero_()

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())

            out1, out2,_ ,_ = model(input_img)
            f = torch.cat((normalize(out1.data.cpu()), normalize(out2.data.cpu()),), 1)
            ff = ff + f

        ff = normalize(ff)
        # features = torch.cat((features, ff), 0)

        ii = count // n_param
        features[ii] = torch.cat((features[ii], ff), 0)
    for j in range(len(features)):
        feature_cat = torch.cat((feature_cat, features[j]), 0)
    return feature_cat

def extract_feature_triplehead(model, dataloaders, f_size):
    n_param = 4000
    feature_cat = torch.FloatTensor()
    features = []
    for i in range(len(dataloaders.dataset)//n_param+1):
        features.append(torch.FloatTensor())

    feature_size = f_size * 3
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, feature_size).zero_()

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())

            out1, out2, out3,_ ,_ ,_ = model(input_img)
            f = torch.cat((normalize(out1.data.cpu()), normalize(out2.data.cpu()), normalize(out3.data.cpu())), 1)
            ff = ff + f

        ff = normalize(ff)
        # features = torch.cat((features, ff), 0)

        ii = count // n_param
        features[ii] = torch.cat((features[ii], ff), 0)
    for j in range(len(features)):
        feature_cat = torch.cat((feature_cat, features[j]), 0)
    return feature_cat


def extract_feature_2head(model, dataloaders, outidx, f_size):
    if outidx == 2:
        feature_size = f_size *2
    else:
        feature_size = f_size

    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, feature_size).zero_()

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())

            out1, out2 ,_ ,_ = model(input_img)
            if outidx == 0:
                f = out1.data.cpu()
            if outidx == 1:
                f = out2.data.cpu()
            if outidx == 2:
                out = torch.cat((out1, out2), 1)
                f = out.data.cpu()
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

def normalize_single(x):
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


def get_id_reid(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def get_id(img_path):
    labels = []
    for path, v in img_path:
        label = v
        labels.append(int(label))
    return labels

def get_id_car(img_path):
    labels = []
    for path, v in img_path:
        label = path.split('/')[-2]
        # label = filename[0:4]
        # camera = filename.split('c')[1]
        # if label[0:2] == '-1':
        #     labels.append(-1)
        # else:
        labels.append(int(label))
        # camera_id.append(int(camera[0]))
    return labels

def get_id_product(img_path):
    labels = []
    for path, v in img_path:
        label = path.split('/')[-2]
        # label = filename[0:4]
        # camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        # camera_id.append(int(camera[0]))
    return labels


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def Recall_at_ks(sim_mat, data='cub', query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
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
    # t = time.time() - start_time
    # print(t)
    return num_valid / float(m)


