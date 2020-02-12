from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np

vlr = 1
vlr2 = 1
scale = 1000000
mom = 0.9
n_ = 3
def get_size_scalar(torch_tensor):
    a,b,c,d = torch_tensor.size()
    return a*b*c*d

def get_size_scalarFC(torch_tensor):
    a,b = torch_tensor.size()
    return a*b

def compute_weight_variation(modelA, modelB):

    L1_varation = []
    nweight = get_size_scalar(modelA.model.conv1.weight)
    variation = torch.norm(modelA.model.conv1.weight.cpu() - modelB.model.conv1.weight.cpu(), 2)
    L1_varation.append(variation.detach().numpy()/nweight*scale)

    for i in range(len(modelA.model.layer1)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer1[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv1.weight.cpu() - modelB.model.layer1[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv2.weight.cpu() - modelB.model.layer1[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv3.weight.cpu() - modelB.model.layer1[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    for i in range(len(modelA.model.layer2)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer2[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv1.weight.cpu() - modelB.model.layer2[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv2.weight.cpu() - modelB.model.layer2[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv3.weight.cpu() - modelB.model.layer2[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    for i in range(len(modelA.model.layer3)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer3[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv1.weight.cpu() - modelB.model.layer3[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv2.weight.cpu() - modelB.model.layer3[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv3.weight.cpu() - modelB.model.layer3[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    for i in range(len(modelA.model.layer4)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer4[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv1.weight.cpu() - modelB.model.layer4[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv2.weight.cpu() - modelB.model.layer4[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv3.weight.cpu() - modelB.model.layer4[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    variation = 0
    nweight = 0
    nweight += get_size_scalarFC(modelA.classifier.add_block[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.add_block[0].weight.cpu() - modelB.classifier.add_block[0].weight.cpu(), 2),2)

    nweight += get_size_scalarFC(modelA.classifier.classifier[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.classifier[0].weight.cpu() - modelB.classifier.classifier[0].weight.cpu(), 2),2)
    L1_varation.append(variation.detach().numpy()**0.5/nweight* scale)

    return L1_varation

