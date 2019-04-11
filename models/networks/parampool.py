from .tools import weight_init
import torch
import torch.nn as nn


# class ParamPool(nn.Module):
#     def __init__(self, input_c, input_s=16):
#         super(ParamPool, self).__init__()
#         self.conv_w = nn.Conv2d(input_c, 1, kernel_size=1, bias=True)
#         self.conv_x = nn.Conv2d(input_c, input_c, kernel_size=1)
#         self.conv_x.apply(weight_init)
#         # self.conv_w.apply(weight_init)
#         self.conv_w.weight.data.normal_(0.0, 0.01)
#         self.conv_w.bias.data.fill_(1.0/input_s*input_s)
#         self.input_c, self.input_s = input_c, input_s
#
#     def forward(self, x):
#         bsize, _, _, _ = x.shape
#         w = self.conv_w(x)
#         w = torch.softmax(w.view(bsize, 1, -1), 2)
#         w = w.view(bsize, 1, self.input_s, self.input_s)
#         x = self.conv_x(x)
#         x = (x*w).sum(3).sum(2)
#         return x


class ParamPool(nn.Module):
    def __init__(self, input_c):
        super(ParamPool, self).__init__()
        self.conv = nn.Conv2d(input_c, 1, kernel_size=1, bias=False)

    def forward(self, x):
        bsize, c, ssize, _ = x.shape
        w = self.conv(x)
        w = torch.softmax(w.view(bsize, 1, -1), 2)
        w = w.view(bsize, 1, ssize, ssize)
        x = (x*w).sum(3).sum(2)
        return x


