# coding=utf-8
from .base_options import BaseOptions
import os


class TrainOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--start_it', type=int, default=0, help='recover from saved')
        self.parser.add_argument('--display_freq', type=int, default=20, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest model')
        self.parser.add_argument('--train_iters', type=int, default=100000, help='training iterations')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--is_train', type=bool, default=True, help='train, val, test, etc')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        home = os.path.expanduser("~")
        self.parser.add_argument('--coco_img_dir', type=str,
                                 default='%s/data/datasets/coco/images/train2014'%home, help='path to coco images')
        self.parser.add_argument('--coco_cap_dir', type=str,
                                 default='%s/data/datasets/coco/annotations/captions_train2014.json'%home,
                                 help='path to coco captions')
        self.parser.add_argument('--imagenet_det_dir', type=str,
                                 default='%s/data/datasets/ILSVRC2014_devkit'%home,
                                 help='path to imagenet detection dataset')
        self.parser.add_argument('--imagenet_cls_dir', type=str,
                                 default='%s/data/datasets/ILSVRC12VOC/images'%home,
                                 help='path to imagenet classification dataset')
        self.parser.add_argument('--train_img_dir', type=str,
                                 default='%s/data/datasets/saliency_Dataset/DUT-train/images'%home,
                                 help='path to saliency training images ')
        self.parser.add_argument('--val_img_dir', type=str,
                                 default='%s/data/datasets/saliency_Dataset/ECSSD/images'%home,
                                 help='path to validation images')
        self.parser.add_argument('--val_gt_dir', type=str,
                                 default='%s/data/datasets/saliency_Dataset/ECSSD/masks'%home,
                                 help='path to validation ground-truth')
        self.initialized = True
