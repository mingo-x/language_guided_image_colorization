from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import init_modules


class SelfAttention(nn.Module):
    '''From DeOldify'''
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.key = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.value = nn.Conv1d(in_channel, in_channel, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)  # Matrix multiplication.
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out


class AutocolorizeVGG16Attention(nn.Module):
    def __init__(self, num_classes=313, vgg_weights=''):
        super(AutocolorizeVGG16Attention, self).__init__()
        self.num_classes = num_classes
        self.vgg16 = AutocolorizeVGG16()
        if vgg_weights != '':
            # Load pretrained weights.
            if os.path.isfile(vgg_weights):
                print("=> loading pretrained weights '{}'".format(vgg_weights))
                weights = torch.load(vgg_weights)
                self.vgg16.load_state_dict(weights['state_dict'])
            else:
                print("=> no weights found at '{}'".format(vgg_weights))
        self.attention_0 = SelfAttention(512)

        init_modules([self.attention_0])
        print('Attention modules initialized.')

    def forward(self, x):
        vgg_22 = self.vgg16.pretrained_vgg_22(x)
        vgg_32 = self.vgg16.pretrained_vgg_32(vgg_22)
        conv_0 = self.vgg16.cnn_0(vgg_32)
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        conv_1 = self.vgg16.cnn_1[: 3](concat_0)
        attended_0 = self.attention_0(conv_1)
        # print(conv_1.data[0, 0, 0, 0], attended_0.data[0,0,0,0])
        conv_1 = self.vgg16.cnn_1[3:](attended_0)
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        conv_2 = self.vgg16.cnn_2(concat_1)
        # Classifier
        output = self.vgg16.classifier(conv_2)

        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        return output, None


class AutocolorizeVGG16(nn.Module):
    def __init__(self, num_classes=313, affine={}):
        super(AutocolorizeVGG16, self).__init__()
        self.num_classes = num_classes

        vgg16_bn = models.vgg16_bn(pretrained=True)
        pretrained_vgg = list(vgg16_bn.children())[0]
        self.pretrained_vgg_22 = pretrained_vgg[: 23]
        self.pretrained_vgg_32 = pretrained_vgg[23: 33]

        # block 5
        layer_list = []
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        bn_idx = 0  # 0
        layer_list.append(nn.BatchNorm2d(512, affine=affine.get(bn_idx, True)))  # 1
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        bn_idx += 1  # 1
        layer_list.append(nn.BatchNorm2d(512, affine=affine.get(bn_idx, True)))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        bn_idx += 1  # 2
        layer_list.append(nn.BatchNorm2d(512, affine=affine.get(bn_idx, True)))  # 7
        layer_list.append(nn.ReLU(inplace=True))  # 8

        # block 6
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        bn_idx += 1  # 3
        layer_list.append(nn.BatchNorm2d(512, affine=affine.get(bn_idx, True)))  # 10
        layer_list.append(nn.ReLU(inplace=True))  # 11
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        bn_idx += 1  # 4
        layer_list.append(nn.BatchNorm2d(512, affine=affine.get(bn_idx, True)))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        self.cnn_0 = nn.Sequential(*layer_list)
        
        # block 7
        layer_list_1 = []
        layer_list_1.append(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1))
        bn_idx += 1  # 5
        layer_list_1.append(nn.BatchNorm2d(512, affine=affine.get(bn_idx, True)))
        layer_list_1.append(nn.ReLU(inplace=True))
        layer_list_1.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        bn_idx += 1  # 6
        layer_list_1.append(nn.BatchNorm2d(512, affine=affine.get(bn_idx, True)))
        layer_list_1.append(nn.ReLU(inplace=True))
        # layer_list_1.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # layer_list_1.append(nn.BatchNorm2d(512))
        # layer_list_1.append(nn.ReLU(inplace=True))
        layer_list_1.append(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1))
        self.cnn_1 = nn.Sequential(*layer_list_1)
        
        # block 8
        layer_list_2 = []
        layer_list_2.append(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        bn_idx += 1  # 7
        layer_list_2.append(nn.BatchNorm2d(256, affine=affine.get(bn_idx, True)))
        layer_list_2.append(nn.ReLU(inplace=True))
        layer_list_2.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        bn_idx += 1  # 8
        layer_list_2.append(nn.BatchNorm2d(256, affine=affine.get(bn_idx, True)))  # 4
        layer_list_2.append(nn.ReLU(inplace=True))
        self.cnn_2 = nn.Sequential(*layer_list_2)

        self.classifier = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)

        init_modules([self.cnn_0, self.cnn_1, self.cnn_2, self.classifier])
        print('Unet rest initialized.')

    def forward(self, x):      
        vgg_22 = self.pretrained_vgg_22(x)
        vgg_32 = self.pretrained_vgg_32(vgg_22)
        conv_0 = self.cnn_0(vgg_32)
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        conv_1 = self.cnn_1(concat_0)
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        conv_2 = self.cnn_2(concat_1)
        # Classifier
        output = self.classifier(conv_2)

        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        return output
