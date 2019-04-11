# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
import models
from datasets import Folder, CocoCaption, caption_collate_fn, ImageNetDetCls, ImageFiles
from evaluate_sal import fm_and_mae
from datasets.build_vocab import Vocabulary
import pickle
import json
import os
import random


from options.train_options import TrainOptions

opt = TrainOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt = opt.parse()

home = os.path.expanduser("~")

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

val_loader = torch.utils.data.DataLoader(
    Folder(opt.val_img_dir, opt.val_gt_dir,
           crop=None, flip=False, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

cap_train_loader = torch.utils.data.DataLoader(
    CocoCaption(opt.coco_img_dir, opt.coco_cap_dir, vocab,
                crop=None, flip=True, rotate=None, size=opt.imageSize,
                mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True, collate_fn=caption_collate_fn)
cls_train_loader = torch.utils.data.DataLoader(
    ImageNetDetCls(opt.imagenet_det_dir,
                   crop=None, flip=True, rotate=None, size=opt.imageSize,
                   mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
unl_train_loader = torch.utils.data.DataLoader(
    ImageFiles(opt.imagenet_cls_dir,
                   crop=None, flip=True, rotate=None, size=opt.imageSize,
                   mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


model = getattr(models, opt.model)(opt, vocab_size=len(cap_train_loader.dataset.vocab), n_class=200)
# model.load('best')


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    model.switch_to_train()
    maxfm, mae, _, _ = fm_and_mae(opt.results_dir, opt.val_gt_dir)
    model.performance = {'maxfm': maxfm, 'mae': mae}
    if hasattr(opt, 'cap_results_dir'):
        maxfm, mae, _, _ = fm_and_mae(opt.cap_results_dir, opt.val_gt_dir)
        model.performance.update({'cap_maxfm': maxfm, 'cap_mae': mae})
    if hasattr(opt, 'cls_results_dir'):
        maxfm, mae, _, _ = fm_and_mae(opt.cls_results_dir, opt.val_gt_dir)
        model.performance.update({'cls_maxfm': maxfm, 'cls_mae': mae})
    return model.performance


class CombinedIter(object):
    def __init__(self, cap_loader, cls_loader, unl_loader):
        self.cap_loader = cap_loader
        self.cls_loader = cls_loader
        self.unl_loader = unl_loader
        self.cap_iter = iter(cap_loader)
        self.i_cap = 0
        self.cls_iter = iter(cls_loader)
        self.i_cls = 0
        self.unl_iter = iter(unl_loader)
        self.i_unl = 0

    def next(self):
        if self.i_cap >= len(self.cap_loader):
            self.cap_iter = iter(self.cap_loader)
            self.i_cap = 0
        img_cap, captions, lengths = self.cap_iter.next()
        self.i_cap += 1

        if self.i_cls >= len(self.cls_loader):
            self.cls_iter = iter(self.cls_loader)
            self.i_cls = 0
        img_cls, categories = self.cls_iter.next()
        self.i_cls += 1

        if self.i_unl >= len(self.unl_loader):
            self.unl_iter = iter(self.unl_loader)
            self.i_unl = 0
        img_unl = self.unl_iter.next()
        self.i_unl += 1
        output = {'img_cap': img_cap, 'captions': captions, 'lengths': lengths,
                  'img_cls': img_cls, 'categories': categories,
                  'img_unl': img_unl}
        return output


def train(model):
    print("============================= TRAIN ============================")
    model.switch_to_train()
    # model.load('best')

    train_iter = CombinedIter(cap_train_loader, cls_train_loader, unl_train_loader)
    log = {'best': 0, 'best_it': 0}

    for i in tqdm(range(opt.start_it+1, opt.train_iters), desc='train'):
        data = train_iter.next()

        model.set_input(data)
        model.optimize_parameters(i)

        if i % opt.display_freq == 0:
            model.show_tensorboard(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            performance = test(model)
            model.show_tensorboard_eval(i)
            log[i] = performance
            if performance['maxfm'] > log['best']:
                log['best'] = performance['maxfm']
                log['best_it'] = i
                model.save('best')
            print(u'最大fm: iter%d的%.4f' % (log['best_it'], log['best']))
            for k, v in performance.items():
                print(u'这次%s: %.4f' % (k, v))
            with open(model.save_dir + '/' + 'train-log.json', 'w') as outfile:
                json.dump(log, outfile)


train(model)

print("We are done")
