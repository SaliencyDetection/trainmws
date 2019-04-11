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


from options.test_options import TestOptions

opt = TestOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt = opt.parse()

home = os.path.expanduser("~")

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

val_loader = torch.utils.data.DataLoader(
    Folder(opt.val_img_dir, opt.val_gt_dir,
           crop=None, flip=False, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


model = getattr(models, opt.model)(opt, vocab_size=len(vocab), n_class=200)


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


model.switch_to_eval()
model.load('best')
performance = test(model)
for k, v in performance.items():
    print(u'这次%s: %.4f' % (k, v))

print("We are done")
