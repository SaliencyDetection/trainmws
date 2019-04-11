# coding=utf-8

import pdb
import torch
from tqdm import tqdm
import models
from datasets import Folder
from evaluate_sal import fm_and_mae
import json
import os
import random


from options.train_options import TrainOptions

opt = TrainOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt = opt.parse()

home = os.path.expanduser("~")

val_loader = torch.utils.data.DataLoader(
    Folder(opt.val_img_dir, opt.val_gt_dir,
           crop=None, flip=False, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
train_loader = torch.utils.data.DataLoader(
    Folder(opt.train_img_dir, './dut-train-crf_bin',
           crop=0.9, flip=True, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

model = models.SalModel(opt)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    model.switch_to_train()
    maxfm, mae, _, _ = fm_and_mae(opt.results_dir, opt.val_gt_dir)
    model.performance = {'maxfm': maxfm, 'mae': mae}
    return model.performance


class CombinedIter(object):
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)
        self.i = 0

    def next(self):
        if self.i >= len(self.loader):
            self.iter = iter(self.loader)
            self.i = 0
        img_sal, gt_sal = self.iter.next()
        self.i += 1
        output = {'img_sal': img_sal, 'gt_sal': gt_sal.unsqueeze(1)}
        return output


def train(model):
    print("============================= TRAIN ============================")
    model.switch_to_train()
    # model.load('best')

    train_iter = CombinedIter(train_loader)
    log = {'best': 0, 'best_it': 0}

    for i in tqdm(range(opt.train_iters), desc='train'):
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
