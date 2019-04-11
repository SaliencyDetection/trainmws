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


class ClsModel(BaseModel):
    def __init__(self, opt, **kwargs):
        super(ClsModel, self).__init__(opt)

        self.name = 'SalCls_' + opt.base
        self.wr = 5e-4  # regularization weight
        self.wst = 0.01

        net = networks.SalCls(n_class=kwargs['n_class'], base=opt.base)
        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()

        self.pred_classes = None

        if opt.is_train:
            msk_size = 16
            if self.opt.no_self_train:
                xx, yy = np.meshgrid(np.arange(msk_size), np.arange(msk_size))
                val = np.sqrt((yy.astype(np.float)-msk_size/2)**2 + (xx.astype(np.float)-msk_size/2)**2)
                val = np.exp(0.4*val)
            else:
                val = np.ones((msk_size, msk_size))
            self.val = torch.tensor(val).float().cuda()[None, None, ...]
            self.criterion = nn.BCEWithLogitsLoss()
            self.criterion_reg = nn.L1Loss(reduce=False)
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                                lr=opt.lr)
        if not opt.no_self_train:
            device = next(self.net.parameters()).device
            self.net.load_state_dict(torch.load('cls_init.pth', map_location={'cuda:%d' % device.index: 'cpu'}), )
            self.criterion_st = networks.SelfTrainLoss(self.v_mean, self.v_std)

    def save(self, label):
        self.save_network(self.net, self.name, label)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name, label)

    def set_input(self, data):
        self.input = data['img_cls'].cuda()
        self.targets = data['categories'].cuda()

    def forward(self):
        # print("We are Forwarding !!")
        self.big_mask_logits, self.mask, self.pred_classes = self.net.forward(self.input)

    def test(self, input, name, WW, HH):
        with torch.no_grad():
            big_mask_logits, _, _ = self.net.forward(input.cuda())
            outputs = F.sigmoid(big_mask_logits.squeeze(1))
        outputs = outputs.detach().cpu().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')

    def backward(self):
        # Combined loss
        loss_cls = self.criterion(self.pred_classes, self.targets)
        loss_reg = self.criterion_reg(F.sigmoid(self.mask), torch.zeros(self.mask.shape).cuda()) * self.val.expand_as(self.mask)
        loss_reg = loss_reg.mean()
        loss = loss_cls + self.wr * loss_reg #6, 6, 6
        if not self.opt.no_self_train:
            loss_self = self.criterion_st(self.input, F.sigmoid(self.big_mask_logits))
            loss += loss_self * self.wst
            self.loss['self'] = loss_self.item()

        loss.backward()
        self.loss['cls'] = loss_cls.item()
        self.loss['reg'] = loss_reg.item()

    def optimize_parameters(self, it):
        # if it!=0 and it % 500 == 0:
        #     self.wr *= 2
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

