import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd.variable import Variable
from .tools import fraze_bn, weight_init, dim_dict
from .base_network import BaseNetwork

from .densenet import *
from .resnet import *
from .vgg import *
from .parampool import ParamPool
import sys
thismodule = sys.modules[__name__]
import pdb


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition2[-2].register_forward_hook(hook)
    model.features.transition1[-2].register_forward_hook(hook)
    # dilation
    # def remove_sequential(all_layers, network):
    #     for layer in network.children():
    #         if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
    #             remove_sequential(all_layers, layer)
    #         if list(layer.children()) == []:  # if leaf node, add it to list
    #             all_layers.append(layer)
    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    # all_layers = []
    # remove_sequential(all_layers, model.features.denseblock4)
    # for m in all_layers:
    #     if isinstance(m, nn.Conv2d) and m.kernel_size==(3, 3):
    #         m.dilation = (2, 2)
    #         m.padding = (2, 2)
    return model


procs = {
    'densenet169': proc_densenet,
}

class Nothing(nn.Module):
    def forward(self, input):
        return input


class SalCls(nn.Module, BaseNetwork):
    def __init__(self, n_class=200, base='densenet169'):
        super(SalCls, self).__init__()
        dims = dim_dict[base][::-1]
        self.preds = nn.ModuleList([nn.Conv2d(d, 1, kernel_size=1) for d in dims[:3]])
        # self.upscales = nn.ModuleList([
        #     Nothing(),
        #     # nn.ConvTranspose2d(1, 1, 4, 2, 1),
        #     nn.ConvTranspose2d(1, 1, 4, 2, 1),
        #     nn.ConvTranspose2d(1, 1, 16, 8, 4),
        # ])
        self.reduce = nn.Conv2d(dims[0], 512, kernel_size=3, padding=1)
        self.classifier = nn.Linear(512, n_class)
        self.param_pool = ParamPool(512)
        self.apply(weight_init)
        self.feature = getattr(thismodule, base)(pretrained=True)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        self.apply(fraze_bn)

    def forward(self, x):
        self.feature.feats[x.device.index] = []
        x = self.feature(x)
        self.feature.feats[x.device.index][0] = F.upsample(self.feature.feats[x.device.index][0], scale_factor=4, mode='bilinear')
        self.feature.feats[x.device.index][1] = F.upsample(self.feature.feats[x.device.index][1], scale_factor=8, mode='bilinear')
        msk = self.preds[0](x)
        big_msk = F.upsample(msk, scale_factor=16, mode='bilinear')
        # big_msk = self.upscales[0](msk)
        # for i in range(1, len(self.preds)):
        #     big_msk = big_msk + self.preds[i](feats[i])
        #     big_msk = self.upscales[i](big_msk)
        # msk = F.sigmoid(msk)
        msk_feat = self.reduce(x)*F.sigmoid(msk)
        msk_feat = self.param_pool(msk_feat)
        cls = self.classifier(msk_feat)
        return big_msk, msk, cls



if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
