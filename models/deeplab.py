import torch
import torch.nn as nn
from torch.autograd.variable import Variable

from .densenet import *
from .resnet import *
from .vgg import *
from .mobilenetv2 import *
from .funcs import *

import numpy as np
import sys

thismodule = sys.modules[__name__]
import pdb


def proc_densenet(model):
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    model.features.transition2[-1].kernel_size = 1
    model.features.transition2[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock3)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)

    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock4)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (4, 4)
            m.padding = (4, 4)
    model.classifier = None
    return model


def proc_vgg(model):
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    model.features[3][-1].kernel_size = 1
    model.features[3][-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features[4])
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)
    model.classifier = None
    return model


def proc_mobilenet2(model):
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if isinstance(layer, InvertedResidual):
                remove_sequential(all_layers, layer.conv)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    model.features[7].conv[3].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features[8:14])
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)

    model.features[14].conv[3].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features[15:])
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (4, 4)
            m.padding = (4, 4)
    model.classifier = None
    return model


procs = {
    'densenet169': proc_densenet,
    'vgg16': proc_vgg,
    'mobilenet2': proc_mobilenet2,
}


class DeepLab(nn.Module):
    def __init__(self, pretrained=True, c_output=21, base='densenet169'):
        super(DeepLab, self).__init__()
        dims = dim_dict[base][::-1]
        self.preds = nn.ModuleList([nn.Conv2d(dims[0], c_output, kernel_size=3, dilation=dl, padding=dl)
                                    for dl in [6, 12, 18, 24]])
        self.upscale = nn.ConvTranspose2d(c_output, c_output, 16, 8, 4)
        self.apply(weight_init)
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature = procs[base](self.feature)
        self.apply(fraze_bn)

    def forward(self, x):
        x = self.feature(x)
        x = sum([f(x) for f in self.preds])
        x = self.upscale(x)
        return x

    # def forward_mscale(self, xs):
    #     outputs = []
    #     for x in xs:
    #         x = self.feature(x)
    #         # x = self.pred(x)
    #         x = sum([f(x) for f in self.preds])
    #         x = self.upscale(x)
    #         outputs += [x]
    #     merge = torch.max(outputs[0], F.upsample(outputs[1], size=img_size, mode='bilinear'))
    #     merge = torch.max(merge, F.upsample(outputs[2], size=img_size, mode='bilinear'))
    #     outputs += [merge]
    #     return outputs


if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
