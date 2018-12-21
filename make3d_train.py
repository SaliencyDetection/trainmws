# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import DepthModel
from datasets import Make3d, ImageFiles
from evaluate_depth import rel_log10_rms
import json
import os


from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


home = os.path.expanduser("~")

train_img_dir = '%s/data/datasets/depth_dataset/make3d/Train400Img'%home
train_gt_dir = '%s/data/datasets/depth_dataset/make3d/Train400Depth'%home

val_img_dir = '%s/data/datasets/depth_dataset/make3d/Test134'%home
val_gt_dir = '%s/data/datasets/depth_dataset/make3d/depth'%home


train_loader = torch.utils.data.DataLoader(
    Make3d(train_img_dir, train_gt_dir,
           crop=0.9, flip=True, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    ImageFiles(val_img_dir, crop=None, flip=False,
               mean=opt.mean, std=opt.std),
    # Make3d(val_img_dir, val_gt_dir,
    #        crop=0.9, flip=True, rotate=None, size=opt.imageSize,
    #        mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    rel, log10, rms = rel_log10_rms(opt.results_dir, val_gt_dir)
    model.switch_to_train()
    return rel, log10, rms


model = DepthModel(opt)


def train(model):
    print("============================= TRAIN ============================")
    model.switch_to_train()

    train_iter = iter(train_loader)
    it = 0
    log = {'best': 1000, 'best_it': 0}

    for i in tqdm(range(opt.train_iters), desc='train'):
        # landscape
        if it >= len(train_loader):
            train_iter = iter(train_loader)
            it = 0
        img, gt, mask = train_iter.next()
        it += 1

        model.set_input(img, gt, mask)
        model.optimize_parameters()

        if i % opt.display_freq == 0:
            model.show_tensorboard(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            rel, log10, rms = test(model)
            if rel < log['best']:
                log['best'] = rel
                log['best_it'] = i
                model.save('best')
            print(u'最大rel: %.4f, 这次rel: %.4f, log10: %.4f, rms: %.4f'%(log['best'], rel, log10, rms))
            with open(model.save_dir+'/'+'log.json', 'w') as outfile:
                json.dump(log, outfile)


train(model)

print("We are done")
