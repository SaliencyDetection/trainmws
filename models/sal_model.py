# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from .base_model import BaseModel
import networks
import pdb


class SalModel(BaseModel):
    def __init__(self, opt, **kwargs):
        super(SalModel, self).__init__(opt)

        self.name = 'SalSal_' + opt.base
        self.ws = 0.05

        net = networks.DeepLab(pretrained=True, c_output=1, base=opt.base)
        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                            lr=opt.lr)

    def save(self, label):
        self.save_network(self.net, self.name, label)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name, label)

    def set_input(self, data):
        self.input = data['img_sal'].cuda()
        self.targets = data['gt_sal'].cuda()

    def forward(self):
        # print("We are Forwarding !!")
        self.big_mask_logits = self.net.forward(self.input)

    def test(self, input, name, WW, HH):
        with torch.no_grad():
            big_mask_logits = self.net.forward(input.cuda())
            outputs = F.sigmoid(big_mask_logits.squeeze(1))
        outputs = outputs.detach().cpu().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')

    def backward(self):
        # Combined loss
        loss = self.criterion(self.big_mask_logits, self.targets) * (1-self.ws)
        gt_self = F.sigmoid(self.big_mask_logits).detach()
        gt_self[gt_self>0.5] = 1
        gt_self[gt_self<=0.5] = 0
        loss += self.criterion(self.big_mask_logits, gt_self) * self.ws
        loss.backward()
        self.loss['sal'] = loss.item()

    def optimize_parameters(self, it):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

