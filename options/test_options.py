from .base_options import BaseOptions
import os


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        home = os.path.expanduser("~")
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--is_train', type=bool, default=False, help='train, val, test, etc')
        self.parser.add_argument('--val_img_dir', type=str,
                                 default='%s/data/datasets/saliency_Dataset/DUT-train/images'%home,
                                 help='path to validation images')
        self.parser.add_argument('--val_gt_dir', type=str,
                                 default='%s/data/datasets/saliency_Dataset/DUT-train/masks'%home,
                                 help='path to validation ground-truth')
        self.initialized = True
