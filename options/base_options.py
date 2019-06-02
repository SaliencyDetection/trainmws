import argparse
import os
import torch
import models


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
        self.parser.add_argument('--imageSize', type=int, default=256, help='input image size')
        self.parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406], help='input image size')
        self.parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225], help='input image size')
        self.parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='CapModel', help='which model to use')
        self.parser.add_argument('--results_dir', type=str, default='',
                                 help='path to save validation results.')
        self.parser.add_argument('--base', type=str, default='densenet169',
                                 help='chooses which backbone network to use. densenet169, vgg16, etc')
        home = os.path.expanduser("~")
        self.parser.add_argument('--checkpoints_dir', type=str, default='%s/mwsFiles'%home, help='path to save params and tensorboard files')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.parser = getattr(models, self.opt.model).modify_commandline_options(self.parser, self.opt.is_train)
        self.opt, _ = self.parser.parse_known_args()

        # save to the disk
        if self.opt.name == '':
            self.opt.name = '_'.join([self.opt.model, self.opt.base])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        if self.opt.results_dir == '':
            self.opt.results_dir = '{}/results'.format(expr_dir)
        if not os.path.exists(self.opt.results_dir):
            os.makedirs(self.opt.results_dir)
        if hasattr(self.opt, 'cap_results_dir'):
            if self.opt.cap_results_dir == '':
                self.opt.cap_results_dir = '{}/cap_results'.format(expr_dir)
            if not os.path.exists(self.opt.cap_results_dir):
                os.makedirs(self.opt.cap_results_dir)
        if hasattr(self.opt, 'cls_results_dir'):
            if self.opt.cls_results_dir == '':
                self.opt.cls_results_dir = '{}/cls_results'.format(expr_dir)
            if not os.path.exists(self.opt.cls_results_dir):
                os.makedirs(self.opt.cls_results_dir)

        self.opt.batchSize = int(self.opt.batchSize/torch.cuda.device_count())*torch.cuda.device_count()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        file_name = os.path.join(expr_dir, 'opt-{}.txt'.format(self.opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
