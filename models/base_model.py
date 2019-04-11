import os
import torch
from tensorboardX import SummaryWriter
import torchvision
from datetime import datetime
import pdb


class BaseModel(object):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        if is_train:
            parser.add_argument('--no_self_train', action='store_true',
                                help='not using self training loss')
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.input = None
        self.performance = {}
        self.loss = {}

        self.v_mean = torch.Tensor(opt.mean)[None, ..., None, None]
        self.v_std = torch.Tensor(opt.std)[None, ..., None, None]

        self.input = None
        self.targets = None
        self.big_mask_logits = None
        self.mask = None


        # tensorboard
        if not os.path.exists(self.save_dir+'/runs'):
            os.mkdir(self.save_dir+'/runs')
        os.system('rm -rf %s/runs/*'%self.save_dir)
        self.writer = SummaryWriter('%s/runs/'%self.save_dir + datetime.now().strftime('%Y%m%d_%H:%M:%S'))

    def set_input(self, input):
        self.input = input

    def show_tensorboard_eval(self, num_iter):
        for k, v in self.performance.items():
            self.writer.add_scalar(k, v, num_iter)

    def show_tensorboard(self, num_iter, num_show=4):
        loss = 0
        for k, v in self.loss.items():
            self.writer.add_scalar(k, v, num_iter)
            loss += v
        self.writer.add_scalar('total loss', loss, num_iter)
        num_show = min(self.input.size(0), num_show)

        pred = torch.sigmoid(self.big_mask_logits[:num_show])
        self.writer.add_image('prediction', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)

        img = self.input.cpu()[:num_show]*self.v_std + self.v_mean
        self.writer.add_image('image', torchvision.utils.make_grid(img), num_iter)

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self, **kwargs):
        pass

    def optimize_parameters(self, **kwargs):
        pass

    def save(self, label):
        pass

    def load(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        device = next(network.parameters()).device
        torch.save(network.cpu().state_dict(), save_path)
        network.to(device)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        device = next(network.parameters()).device
        network.load_state_dict(torch.load(save_path, map_location={'cuda:%d' % device.index: 'cpu'}))

    def update_learning_rate(**kwargs):
        pass
