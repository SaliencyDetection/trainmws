# coding=utf-8
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from .base_model import BaseModel

import networks
import pdb


class TransLoss(nn.Module):
    def forward(self, x, target):
        target = target.clone()
        target[target>0.5] = 1
        target[target<=0.5] = 0
        loss = F.binary_cross_entropy(x, target)
        return loss


class ClsCapModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--cap_results_dir', type=str, default='',
                            help='path to save validation results of caption network.')
        parser.add_argument('--cls_results_dir', type=str, default='',
                            help='path to save validation results of classification network.')
        return parser

    def __init__(self, opt, **kwargs):
        super(ClsCapModel, self).__init__(opt)

        self.name = 'SalCap_' + opt.base
        self.wr = 5e-4  # regularization weight
        self.wt_cls = 0.01
        self.wt_cap = 0.01
        self.wc = 0.1

        net_cap = networks.SalCap(vocab_size=kwargs['vocab_size'], base=opt.base)
        net_cap = torch.nn.parallel.DataParallel(net_cap)
        self.net_cap = net_cap.cuda()
        net_cls = networks.SalCls(n_class=kwargs['n_class'], base=opt.base)
        net_cls = torch.nn.parallel.DataParallel(net_cls)
        self.net_cls = net_cls.cuda()

        # device = next(self.net_cap.parameters()).device
        # self.net_cap.load_state_dict(torch.load('cap_init.pth', map_location={'cuda:%d' % device.index: 'cpu'}))
        # device = next(self.net_cls.parameters()).device
        # self.net_cls.load_state_dict(torch.load('cls_init.pth', map_location={'cuda:%d' % device.index: 'cpu'}), )

        self.img_unl = None
        self.img_cap = None
        self.captions = None
        self.lengths = None
        self.pred_captions = None
        self.target_captions = None
        self.img_cls = None
        self.pred_classes = None
        self.target_classes = None
        self.big_mask_logits_cap = None
        self.mask_cap = None
        self.big_mask_logits_cls = None
        self.mask_cls = None

        if opt.start_it > 0:
            self.load(opt.start_it)

        if opt.is_train:
            # self.criterion_st = networks.SelfTrainLoss(self.v_mean, self.v_std)
            # self.criterion_tt = networks.TransTrainLoss(self.v_mean, self.v_std)
            self.criterion_co = networks.DeepCoTrainLoss(self.v_mean, self.v_std, net_cls=self.net_cls, net_cap=self.net_cap)
            # self.criterion_co = nn.L1Loss()
            self.criterion_cap = nn.CrossEntropyLoss()
            self.criterion_cls = nn.BCEWithLogitsLoss()
            self.criterion_reg = nn.BCELoss()
            # self.criterion_two = nn.BCELoss(reduce=False)
            self.criterion_tt = TransLoss()
            # self.criterion_tt = nn.L1Loss()
            self.optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, self.net_cls.parameters()), 'lr': opt.lr},
                {'params': filter(lambda p: p.requires_grad, self.net_cap.parameters()), 'lr': opt.lr}])

    def save(self, label):
        self.save_network(self.net_cls, self.name+'_netcls', label)
        self.save_network(self.net_cap, self.name+'_netcap', label)

    def load(self, label):
        print('loading %s' % label)
        self.load_network(self.net_cls, self.name+'_netcls', label)
        self.load_network(self.net_cap, self.name+'_netcap', label)

    def show_tensorboard(self, num_iter, num_show=4):
        loss = 0
        for k, v in self.loss.items():
            self.writer.add_scalar(k, v, num_iter)
            loss += v
        self.writer.add_scalar('total loss', loss, num_iter)
        num_show = min(self.img_cls.size(0), self.img_cap.size(0), num_show)

        pred = torch.sigmoid(self.big_mask_logits_cap[:num_show])
        self.writer.add_image('prediction cap_net', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)

        pred = torch.sigmoid(self.big_mask_logits_cls[:num_show])
        self.writer.add_image('prediction cls_net', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)

        if hasattr(self.criterion_co, 'gt_mr'):
            pred = self.criterion_co.gt_mr[:num_show]
            self.writer.add_image('gt mr', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)
        #
        # img = self.img_unl.cpu()[:num_show]*self.v_std + self.v_mean
        # self.writer.add_image('image unl', torchvision.utils.make_grid(img), num_iter)

        img = self.img_cls.cpu()[:num_show]*self.v_std + self.v_mean
        self.writer.add_image('image cls', torchvision.utils.make_grid(img), num_iter)

        img = self.img_cap.cpu()[:num_show]*self.v_std + self.v_mean
        self.writer.add_image('image cap', torchvision.utils.make_grid(img), num_iter)

    def set_input(self, data):
        self.img_unl = data['img_unl']
        self.img_cls = data['img_cls']
        self.target_classes = data['categories']
        self.img_cap = data['img_cap']
        self.captions = data['captions']
        self.lengths = data['lengths']
        target_captions = pack_padded_sequence(self.captions, self.lengths, batch_first=True)[0]
        self.target_captions = target_captions

    def forward_cap(self):
        self.big_mask_logits_cap, self.mask_cap, self.pred_captions = \
            self.net_cap.forward(self.img_cap.cuda(1), self.captions.cuda(1), self.lengths)

    def forward_cls(self):
        self.big_mask_logits_cls, self.mask_cls, self.pred_classes = \
            self.net_cls.forward(self.img_cls.cuda(0))

    def test(self, input, name, WW, HH):
        with torch.no_grad():
            big_mask_logits_cls, _, _ = self.net_cls.forward(input.cuda(0))
            big_mask_logits_cap = self.net_cap.forward(input.cuda(1))
        outputs = F.sigmoid(big_mask_logits_cls.squeeze(1))
        outputs = outputs.detach().cpu().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.cls_results_dir, name[ii]), 'PNG')
        outputs = F.sigmoid(big_mask_logits_cap.squeeze(1))
        outputs = outputs.detach().cpu().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.cap_results_dir, name[ii]), 'PNG')
        outputs = F.sigmoid(big_mask_logits_cls.squeeze(1).cpu())*0.5+F.sigmoid(big_mask_logits_cap.squeeze(1).cpu())*0.5
        outputs = outputs.detach().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')

    def backward_co(self, it):
        big_pred_cap, _ = \
            self.net_cap.forward(self.img_unl.cuda())
        big_pred_cls, _, _ = \
            self.net_cls.forward(self.img_unl.cuda())
        loss = self.criterion_co(self.img_unl, [F.sigmoid(big_pred_cap), F.sigmoid(big_pred_cls)])
        loss *= self.wc
        loss.backward()
        self.loss['co'] = loss

    def backward_cap(self, it):
        # Combined loss
        loss_cap = self.criterion_cap(self.pred_captions, self.target_captions.cuda())
        loss_reg = self.criterion_reg(F.sigmoid(self.mask_cap), torch.zeros(self.mask_cap.shape).cuda())
        loss = loss_cap + self.wr * loss_reg  # 6, 6, 6
        self.loss['cap'] = loss_cap.item()
        self.loss['reg'] = loss_reg.item()
        loss.backward()

        pred_big, pred = self.net_cap.forward(self.img_cls.cuda())
        loss_two = self.criterion_tt(F.sigmoid(pred), F.sigmoid(self.mask_cls.detach())) * self.wt_cap
        # gt = F.sigmoid(self.mask_cls.detach())

        # loss_two = self.criterion_tt(F.sigmoid(pred), gt) * self.wt_cap

        loss_two.backward()
        self.loss['cap_two'] = loss_two.item()

    def backward_cls(self, it):
        # Combined loss
        loss_cls = self.criterion_cls(self.pred_classes, self.target_classes.cuda())
        loss_reg = self.criterion_reg(F.sigmoid(self.mask_cls), torch.zeros(self.mask_cls.shape).cuda())
        # loss_reg = (loss_reg*self.cls_val).mean()
        loss = loss_cls + loss_reg * self.wr #6, 6, 6
        self.loss['cls'] = loss_cls.item()
        self.loss['reg'] = loss_reg.item()
        loss.backward()

        pred_big, pred, _ = self.net_cls.forward(self.img_cap.cuda())
        loss_two = self.criterion_tt(F.sigmoid(pred), F.sigmoid(self.mask_cap).detach()) * self.wt_cls
        # gt = F.sigmoid(self.mask_cap).detach()
        # loss_two = self.criterion_tt(F.sigmoid(pred), gt) * self.wt_cls
        loss_two.backward()
        self.loss['cls_two'] = loss_two.item()

    def optimize_parameters(self, it):
        self.forward_cls()
        self.forward_cap()
        self.optimizer.zero_grad()
        self.backward_cls(it)
        self.backward_cap(it)
        if it > 200:
            self.backward_co(it)
        self.optimizer.step()
        # big_msk_cls, msk, cls = self.net_cls(self.img_cls.cuda())
        # self.big_mask_logits_cls = big_msk_cls
        # cls_loss = F.binary_cross_entropy_with_logits(cls, self.target_classes.cuda())
        # cls_reg = self.wr*F.binary_cross_entropy(msk, torch.zeros(msk.shape).cuda(), reduce=False) * self.cls_val
        # cls_loss = cls_loss + cls_reg.mean()
        # self.optimizer.zero_grad()
        # cls_loss.backward()
        #
        # temp_gt = msk.detach().cuda()
        # temp_gt[temp_gt>0.5] = 1
        # temp_gt[temp_gt<=0.5] = 0
        # big_msk_temp, msk_temp = self.net_cap(self.img_cls.cuda())
        # aux_loss = 0.01*F.binary_cross_entropy(msk_temp, temp_gt)
        # aux_loss.backward()
        # """caption"""
        # big_msk_cap, msk, outputs = self.net_cap(self.img_cap.cuda(), self.captions.cuda(), self.lengths)
        # self.big_mask_logits_cap = big_msk_cap
        # cap_loss = F.cross_entropy(outputs, self.target_captions.cuda()) + \
        #            self.wr*F.binary_cross_entropy(msk, torch.zeros(msk.shape).cuda())
        # cap_loss.backward()
        #
        # big_msk_temp, msk_temp, _ = self.net_cls(self.img_cap.cuda())
        # temp_gt = msk.detach().cuda()
        # temp_gt[temp_gt>0.5] = 1
        # temp_gt[temp_gt<=0.5] = 0
        # aux_loss = 0.01*F.binary_cross_entropy(msk_temp, temp_gt)
        # aux_loss.backward()
        # """unlabeled"""
        # big_msk_cls, msk_cls, _ = self.net_cls(self.img_unl.cuda())
        # big_msk_cap, msk_cap = self.net_cap(self.img_unl.cuda())
        #
        # big_msk_cls = F.sigmoid(big_msk_cls)
        # big_msk_cap = F.sigmoid(big_msk_cap)
        # arr_imgs = (self.img_unl*self.v_std+self.v_mean).numpy().transpose((0, 2, 3, 1))
        # arr_cls = big_msk_cls.squeeze(1).detach().cpu().numpy()
        # arr_cap = big_msk_cap.squeeze(1).detach().cpu().numpy()
        # gt_mr = networks.mr(arr_imgs, [arr_cls, arr_cap])
        # gt_mr = torch.Tensor(gt_mr).unsqueeze(1).float()
        # aux_loss = 0.01*F.binary_cross_entropy(big_msk_cls, gt_mr.cuda())
        # aux_loss.backward()
        # aux_loss = 0.01*F.binary_cross_entropy(big_msk_cap, gt_mr.cuda())
        # aux_loss.backward()
        #
        # self.optimizer.step()

    def switch_to_train(self):
        self.net_cls.train()
        self.net_cap.train()

    def switch_to_eval(self):
        self.net_cls.eval()
        self.net_cap.eval()
