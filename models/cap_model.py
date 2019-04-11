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


class CapModel(BaseModel):
    def __init__(self, opt, **kwargs):
        super(CapModel, self).__init__(opt)

        self.name = 'SalCap_' + opt.base
        self.wr = 5e-5  # regularization weight
        self.wst = 0.01

        net = networks.SalCap(vocab_size=kwargs['vocab_size'], base=opt.base)
        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()

        self.captions = None
        self.lengths = None
        self.pred_captions = None

        if opt.is_train:
            self.criterion = nn.CrossEntropyLoss()
            self.criterion_st = networks.SelfTrainLoss(self.v_mean, self.v_std)
            self.criterion_reg = nn.BCELoss()
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                                lr=1e-4)
        msk_size = 16
        # val = np.ones((msk_size, msk_size))
        if not opt.no_self_train:
            device = next(self.net.parameters()).device
            self.net.load_state_dict(torch.load('cap_init.pth', map_location={'cuda:%d' % device.index: 'cpu'}), )
            self.criterion_st = networks.SelfTrainLoss(self.v_mean, self.v_std)
            val = np.ones((msk_size, msk_size))
        else:
            xx, yy = np.meshgrid(np.arange(msk_size), np.arange(msk_size))
            val = np.sqrt((yy.astype(np.float)-msk_size/2)**2 + (xx.astype(np.float)-msk_size/2)**2)
            val = np.exp(0.4*val)
        self.val = torch.tensor(val).float().cuda()[None, None, ...]

    def save(self, label):
        self.save_network(self.net, self.name, label)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name, label)

    def set_input(self, data):
        self.input = data['img_cap'].cuda()
        self.captions = data['captions'].cuda()
        self.lengths = data['lengths']
        targets = pack_padded_sequence(self.captions, self.lengths, batch_first=True)[0]
        self.targets = targets.cuda()

    def forward(self):
        self.big_mask_logits, self.mask, self.pred_captions, self.full_captions = self.net.forward(self.input, self.captions, self.lengths)

    def test(self, input, name, WW, HH):
        with torch.no_grad():
            big_mask_logits = self.net.forward(input.cuda())
            outputs = F.sigmoid(big_mask_logits.squeeze(1))
        outputs = outputs.detach().cpu().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')

    def backward(self, it):
        # Combined loss
        loss_cap = self.criterion(self.pred_captions, self.targets)
        loss_reg = self.criterion_reg(F.sigmoid(self.mask), torch.zeros(self.mask.shape).cuda())
        loss = loss_cap + self.wr * loss_reg #6, 6, 6
        if not self.opt.no_self_train:
            loss_self = self.criterion_st(self.input, F.sigmoid(self.big_mask_logits))
            loss += loss_self * self.wst
            self.loss['self'] = loss_self.item()

        loss.backward()
        self.loss['cap'] = loss_cap.item()
        self.loss['reg'] = loss_reg.item()

    def optimize_parameters(self, it):
        self.forward()
        self.optimizer.zero_grad()
        self.backward(it)
        self.optimizer.step()

    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

