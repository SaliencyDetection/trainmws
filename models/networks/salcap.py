import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pdb
from .densenet import *
from .resnet import *
from .vgg import *
from .tools import fraze_bn, weight_init, dim_dict
from .base_network import BaseNetwork
from .parampool import ParamPool
import numpy as np
import sys
thismodule = sys.modules[__name__]


# class ParamPool(nn.Module):
#     def __init__(self, input_c, input_s):
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


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition2[-2].register_forward_hook(hook)
    model.features.transition1[-2].register_forward_hook(hook)
    # dilation
    # def remove_sequential(all_layers, network):
    #     for layer in network.children():
    #         if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
    #             remove_sequential(all_layers, layer)
    #         if list(layer.children()) == []:  # if leaf node, add it to list
    #             all_layers.append(layer)
    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    # all_layers = []
    # remove_sequential(all_layers, model.features.denseblock4)
    # for m in all_layers:
    #     if isinstance(m, nn.Conv2d) and m.kernel_size==(3, 3):
    #         m.dilation = (2, 2)
    #         m.padding = (2, 2)
    model.classifier = None
    return model


procs = {
    'densenet169': proc_densenet,
         }


class Nothing(nn.Module):
    def forward(self, input):
        return input


class EncoderCNN(nn.Module):
    def __init__(self, patt_size=512, base='densenet169'):
        super(EncoderCNN, self).__init__()
        dims = dim_dict[base][::-1]
        dims[0] += 2
        self.preds = nn.ModuleList([nn.Conv2d(d, 1, kernel_size=1) for d in dims[:3]])
        # self.linear = nn.Linear(patt_size, embed_size)
        self.reduce = nn.Conv2d(dims[0], patt_size, kernel_size=3, padding=1)
        # self.upscales = nn.ModuleList([
        #     # nn.Conv2dTranspose2d(1, 1, 4, 2, 1),
        #     Nothing(),
        #     nn.ConvTranspose2d(1, 1, 4, 2, 1),
        #     nn.ConvTranspose2d(1, 1, 16, 8, 4),
        # ])
        self.msk_size = 16
        self.param_pool = ParamPool(patt_size)
        self.apply(weight_init)

        self.feature = getattr(thismodule, base)(pretrained=True)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        xx, yy = np.meshgrid(np.arange(self.msk_size), np.arange(self.msk_size))
        self.register_buffer('xx', torch.from_numpy(xx[None, None, ...]).float())
        self.register_buffer('yy', torch.from_numpy(yy[None, None, ...]).float())

        self.apply(fraze_bn)

    def forward(self, x):
        """Extract feature vectors from input images."""
        self.feature.feats[x.device.index] = []
        bsize = x.size(0)
        x = self.feature(x)
        self.feature.feats[x.device.index][0] = F.upsample(self.feature.feats[x.device.index][0], scale_factor=4, mode='bilinear')
        self.feature.feats[x.device.index][1] = F.upsample(self.feature.feats[x.device.index][1], scale_factor=8, mode='bilinear')
        # feats = self.feature.feats[x.device.index]
        # feats += [x]
        # feats = feats[::-1]
        feats0 = torch.cat((x, self.xx.expand(bsize, 1, self.msk_size, self.msk_size)), 1)
        feats0 = torch.cat((feats0, self.yy.expand(bsize, 1, self.msk_size, self.msk_size)), 1)

        msk = self.preds[0](feats0)
        big_msk = F.upsample(msk, scale_factor=16, mode='bilinear')
        # big_msk = self.upscales[0](msk)
        # for i in range(1, len(self.preds)):
        #     big_msk = big_msk + self.preds[i](feats[i])
        #     big_msk = self.upscales[i](big_msk)

        feat = self.reduce(feats0)
        msk_feat = self.param_pool(feat*F.sigmoid(msk))
        # msk_feat = (self.param_pool(msk_feat) + F.avg_pool2d(msk_feat, kernel_size=16).squeeze(3).squeeze(2))
        # msk_feat = F.avg_pool2d(msk_feat, kernel_size=16).squeeze(3).squeeze(2)

        # msk_feat = self.linear(msk_feat)
        if self.training:
            return big_msk, msk, msk_feat
        else:
            return big_msk


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        l = len(lengths)/torch.cuda.device_count()
        s = l*torch.cuda.current_device()
        packed = pack_padded_sequence(embeddings, lengths[s:s+l], batch_first=True)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        # outputs = F.linear(hiddens[0], weight=self.embed.weight.detach())
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class SalCap(nn.Module, BaseNetwork):
    def __init__(self, vocab_size, base='densenet169',
                 embed_size=512, hidden_size=512, num_layers=1, max_seq_length=20):
        super(SalCap, self).__init__()
        self.encoder = EncoderCNN(embed_size, base=base)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)

    def forward(self, images, captions=None, lengths=None):
        if self.training:
            if captions is not None:
                big_msk, msk, msk_feat = self.encoder(images)
                outputs = self.decoder(msk_feat, captions, lengths)
                return big_msk, msk, outputs
            else:
                big_msk, msk, _  = self.encoder(images)
                return big_msk, msk
        else:
            return self.encoder(images)

