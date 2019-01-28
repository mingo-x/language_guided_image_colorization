import argparse
import cv2
from datasets import CocoStuff164k
from language_support import get_caption_encoder, FiLM, FiLMWithAttn
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage import io, color
from tensorflow import summary, Summary
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import utils
from vgg_pretrain import AutocolorizeVGG16


class FiLMx(nn.Module):
    def __init__(self, in_c, out_c, k=0, spatial_attention_ver=-1):
        super(FiLMx, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.dense = nn.Linear(self.in_c, self.out_c * 2)
        self.spatial_attention_ver = spatial_attention_ver
        if self.spatial_attention_ver == 0:
            self.sa = SpatialAttention(out_c, in_c, k)
        elif self.spatial_attention_ver == 1:
            self.sa = SpatialAttentionV1(out_c, in_c, k)
        elif self.spatial_attention_ver == 2:
            self.sa = SpatialAttentionV2(out_c, in_c, k)

        utils.init_modules([self.dense])
        b = torch.ones(self.out_c * 2)
        b[self.out_c:] = 0.
        self.dense.bias.data.copy_(b)

    def forward(self, x, caption_features):
        dense_film = self.dense(caption_features)
        gammas, betas = torch.split(dense_film, self.out_c, dim=-1)
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        output = gammas * x + betas
        if self.spatial_attention_ver != -1:
            output, attention = self.sa(output, caption_features)
            return output, attention
        # else:
        return output, None


class SegNet(nn.Module):
    def __init__(self, version=0):
        super(SegNet, self).__init__()
        print('SegNet version {}'.format(version))
        if version == 2:
            self.deconv_4 = nn.Sequential(
                nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        else:
            self.deconv_4 = None
        if version == 4:
            self.deconv_5 = None
        else:
            self.deconv_5 = nn.Sequential(
                nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        if version != 4:
            self.deconv_6 = nn.Sequential(
                nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        else:
            self.deconv_6 = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
        if version == 0 or version == 3:
            self.deconv_7 = nn.Sequential(
                nn.ConvTranspose2d(512, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        else:
            self.deconv_7 = None

        seg_conv_list = [
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1) if version != 4 else nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        if version == 2 or version == 3 or version == 4:
            seg_conv_list.extend([
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ])
        self.seg_conv = nn.Sequential(*seg_conv_list)
        self.seg_classifier = nn.Conv2d(256, 182, kernel_size=1, stride=1)
        utils.init_modules([self.deconv_4, self.deconv_5, self.deconv_6, self.deconv_7, self.seg_conv, self.seg_classifier])


class SpatialAttention(nn.Module):
    '''SCA-CNN'''
    def __init__(self, img_c, cap_c, k):
        super(SpatialAttention, self).__init__()
        self.conv_0 = nn.Conv1d(img_c, k, 1)
        self.dense = nn.Linear(cap_c, k, bias=False)
        self.tanh = nn.Tanh()
        self.conv_1 = nn.Conv1d(k, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        utils.init_modules([self.conv_0, self.dense, self.conv_1])
        print('Spatial attention v0')

    def forward(self, x, h):
        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        img_feature = self.conv_0(flatten)
        del flatten
        cap_feature = self.dense(h).unsqueeze_(-1).expand(img_feature.shape)
        alpha = self.tanh(img_feature + cap_feature)
        del img_feature, cap_feature
        alpha = self.conv_1(alpha).squeeze_(1)
        alpha = self.softmax(alpha) * (shape[2] * shape[3])  # Restore original scale
        alpha = alpha.view(shape[0], 1, shape[2], shape[3])
        out = x * alpha

        return out, alpha.view(shape[0], shape[2], shape[3])


class SpatialAttentionV1(nn.Module):
    def __init__(self, img_c, cap_c, k):
        super(SpatialAttentionV1, self).__init__()
        self.query = nn.Linear(cap_c, k)
        self.key = nn.Conv1d(img_c, k, 1)
        self.value = nn.Conv1d(img_c, img_c, 1)
        utils.init_modules([self.query, self.key, self.value])
        print('Spatail attention v1')

    def forward(self, x, h):
        shape = x.shape
        query = self.query(h).unsqueeze_(1)  # Bx1xC'
        flatten = x.view(shape[0], shape[1], -1)  # BxCxN
        key = self.key(flatten)  # BxC'xN
        value = self.value(flatten)  # BxCxN
        del flatten
        query_key = torch.bmm(query, key)  # Matrix multiplication, Bx1xN
        del query, key
        attention = F.softmax(query_key, 2)
        output = attention * value
        output = output.view(shape)
        output = output + x

        return output, attention.view(shape[0], shape[2], shape[3])


class SpatialAttentionV2(nn.Module):
    def __init__(self, img_c, cap_c, k):
        super(SpatialAttentionV2, self).__init__()
        self.query = nn.Linear(cap_c, k)
        self.key = nn.Conv1d(img_c, k, 1)
        utils.init_modules([self.query, self.key])
        print('Spatail attention v2')

    def forward(self, x, h):
        shape = x.shape
        query = self.query(h).unsqueeze_(1)  # Bx1xC'
        flatten = x.view(shape[0], shape[1], -1)  # BxCxN
        key = self.key(flatten)  # BxC'xN
        del flatten
        query_key = torch.bmm(query, key)  # Matrix multiplication, Bx1xN
        del query, key
        attention = F.softmax(query_key, 2) * (shape[2] * shape[3]) 
        attention = attention.view(shape[0], 1, shape[2], shape[3])
        output = x * attention

        return output, attention.view(shape[0], shape[2], shape[3])


class AutocolorizeVGGWithSeg(nn.Module):
    def __init__(self, num_classes=313):
        super(AutocolorizeVGGWithSeg, self).__init__()
        self.num_classes = num_classes

        layer_list = []
        # 224 b1
        layer_list.append(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))
        layer_list.append(nn.BatchNorm2d(64))
        layer_list.append(nn.ReLU(inplace=True))
        # 112 b2
        layer_list.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1))
        layer_list.append(nn.BatchNorm2d(128))
        layer_list.append(nn.ReLU(inplace=True))
        # 56 b3 
        layer_list.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1))
        layer_list.append(nn.BatchNorm2d(256))
        layer_list.append(nn.ReLU(inplace=True))
        # 28 b4
        layer_list.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.BatchNorm2d(512))
        layer_list.append(nn.ReLU(inplace=True))
        self.block_1_to_4 = nn.Sequential(*layer_list)

        block_5_list = []
        # 28 b5
        block_5_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        block_5_list.append(nn.ReLU(inplace=True))
        block_5_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        block_5_list.append(nn.ReLU(inplace=True))
        block_5_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        block_5_list.append(nn.BatchNorm2d(512))
        block_5_list.append(nn.ReLU(inplace=True))
        self.block_5 = nn.Sequential(*block_5_list)
        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        block_6_list = []
        block_6_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        block_6_list.append(nn.ReLU(inplace=True))
        block_6_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        block_6_list.append(nn.ReLU(inplace=True))
        block_6_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2))
        block_6_list.append(nn.BatchNorm2d(512))
        block_6_list.append(nn.ReLU(inplace=True))
        self.block_6 = nn.Sequential(*block_6_list)
        self.deconv_6 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        block_7_list = []
        block_7_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        block_7_list.append(nn.ReLU(inplace=True))
        block_7_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        block_7_list.append(nn.ReLU(inplace=True))
        block_7_list.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        block_7_list.append(nn.BatchNorm2d(512))
        block_7_list.append(nn.ReLU(inplace=True))
        self.block_7 = nn.Sequential(*block_7_list)
        self.deconv_7 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        block_8_list = []
        block_8_list.append(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1))
        block_8_list.append(nn.ReLU(inplace=True))
        block_8_list.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        block_8_list.append(nn.ReLU(inplace=True))
        block_8_list.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        block_8_list.append(nn.ReLU(inplace=True))
        self.block_8 = nn.Sequential(*block_8_list)

        self.classifier = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)

        self.conv_9 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.seg_classifier = nn.Conv2d(256, 182, kernel_size=1, stride=1)

        utils.init_modules([self.block_1_to_4, self.block_5, self.deconv_5, self.block_6, self.deconv_6, self.block_7, self.deconv_7, self.block_8, self.classifier, self.conv_9, self.seg_classifier])

    def forward(self, x):
        output = self.block_1_to_4(x)
        conv5 = self.block_5(output)
        deconv_5 = self.deconv_5(conv5)
        conv6 = self.block_6(conv5)
        del conv5
        deconv_6 = self.deconv_6(conv6)
        conv7 = self.block_7(conv6)
        del conv6
        deconv_7 = self.deconv_7(conv7)
        output = self.block_8(conv7)
        del conv7
        output = self.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)

        seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        del deconv_5, deconv_6, deconv_7
        seg_out = self.conv_9(seg_out)
        seg_out = self.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        return output, seg_out


class AutocolorizeUnetWithSeg(nn.Module):
    def __init__(self, num_classes=313):
        super(AutocolorizeUnetWithSeg, self).__init__()
        self.num_classes = num_classes
        self.unet = AutocolorizeVGG16()
        self.segnet = SegNet()

    def forward(self, x):
        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        block5 = self.unet.cnn_0[0: 9](vgg_32)
        deconv_5 = self.segnet.deconv_5(block5)
        conv_0 = self.unet.cnn_0[9:](block5)
        del block5
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        deconv_6 = self.segnet.deconv_6(concat_0)
        conv_1 = self.unet.cnn_1(concat_0)
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        deconv_7 = self.segnet.deconv_7(concat_1)
        output = self.unet.cnn_2(concat_1)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        del deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        return output, seg_out


class AutocolorizeUnetCap(nn.Module):
    def __init__(
        self, 
        d_hid=1024, 
        d_emb=300, 
        num_classes=313, 
        train_vocab_embeddings=None, 
        pretrained_weights='', 
        with_spatial_attention=[-1, -1, -1, -1],
        caption_encoder_version='gru',
        caption_encoder_dropout=0.2,
        k=2,
    ):
        super(AutocolorizeUnetCap, self).__init__()
        self.in_dim = [512, 512, 256, 256]
        self.num_classes = num_classes
        self.n_hidden = d_hid
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, self.n_hidden, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16(affine={2: False, 8: False})
        for i in xrange(len(with_spatial_attention)):
            if with_spatial_attention[i] != -1:
                print('Spation attention at FiLM {}'.format(i))
        self.film0 = FiLMx(self.n_hidden, self.in_dim[0], self.in_dim[0] / k, with_spatial_attention[0])
        self.film1 = FiLMx(self.n_hidden, self.in_dim[1], self.in_dim[1] / k, with_spatial_attention[1])
        self.film2 = FiLMx(self.n_hidden, self.in_dim[2], self.in_dim[2] / k, with_spatial_attention[2])
        self.film3 = FiLMx(self.n_hidden, self.in_dim[3], self.in_dim[3] / k, with_spatial_attention[3])
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                b0 = torch.cat((weights['cnn_0.7.weight'], weights['cnn_0.7.bias']), dim=-1)
                self.film0.dense.bias.data.copy_(b0)
                b3 = torch.cat((weights['cnn_2.4.weight'], weights['cnn_2.4.bias']), dim=-1)
                self.film3.dense.bias.data.copy_(b3)
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))            

    def forward(self, x, captions, caption_lens, return_attention=False):
        caption_features = self.caption_encoder(captions, caption_lens)
        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        block5 = self.unet.cnn_0[0: 8](vgg_32)
        block5, attention0 = self.film0(block5, caption_features)  # FiLM
        if not return_attention:
            del attention0
        block5 = self.unet.cnn_0[8](block5)  # ReLU
        conv_0 = self.unet.cnn_0[9:](block5)
        del block5
        conv_0, attention1 = self.film1(conv_0, caption_features)  # FiLM
        if not return_attention:
            del attention1
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        conv_1 = self.unet.cnn_1(concat_0)
        conv_1, attention2 = self.film2(conv_1, caption_features)  # FiLM
        if not return_attention:
            del attention2
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        output = self.unet.cnn_2[: 5](concat_1)
        output, attention3 = self.film3(output, caption_features)  # FiLM
        if not return_attention:
            del attention3
        del caption_features
        output = self.unet.cnn_2[5:](output)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)

        if return_attention:
            return output, None, (attention0, attention1, attention2, attention3)
        else:
            return output, None


class AutocolorizeUnetSegCap(nn.Module):
    def __init__(self, d_hid=1024, d_emb=300, num_classes=313, train_vocab_embeddings=None, pretrained_weights='', caption_encoder_version='gru'):
        super(AutocolorizeUnetSegCap, self).__init__()
        self.n_hidden = d_hid
        self.in_dim = 512
        self.num_classes = num_classes
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, d_hid, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16(affine={2: False})
        self.segnet = SegNet()
        self.dense_film = nn.Linear(self.n_hidden, self.in_dim * 2) 
        self.film = FiLM()
        utils.init_modules([self.dense_film])
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                b = torch.cat((weights['cnn_0.7.weight'], weights['cnn_0.7.bias']), dim=-1)
                self.dense_film.bias.data.copy_(b)
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))
        else:
            # Match affine in BN.
            b = torch.ones(self.in_dim * 2)
            b[self.in_dim:] = 0.
            self.dense_film.bias.data.copy_(b)

    def forward(self, x, captions, caption_lens):
        caption_features = self.caption_encoder(captions, caption_lens)
        dense_film = self.dense_film(caption_features)
        gammas, betas = torch.split(dense_film, self.in_dim, dim=-1)

        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        block5 = self.unet.cnn_0[0: 8](vgg_32)
        block5 = self.film(block5, gammas, betas)  # FiLM
        block5 = self.unet.cnn_0[8](block5)  # ReLU
        deconv_5 = self.segnet.deconv_5(block5)
        conv_0 = self.unet.cnn_0[9:](block5)
        del block5
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        deconv_6 = self.segnet.deconv_6(concat_0)
        conv_1 = self.unet.cnn_1(concat_0)
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        deconv_7 = self.segnet.deconv_7(concat_1)
        output = self.unet.cnn_2(concat_1)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        del deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        return output, seg_out       


class AutocolorizeUnetSegCapV2(nn.Module):
    def __init__(
        self, 
        d_hid=1024, 
        d_emb=300, 
        num_classes=313, 
        train_vocab_embeddings=None, 
        pretrained_weights='', 
        seg_ver=0, 
        with_spatial_attention=[-1, -1, -1, -1],
        caption_encoder_version='gru',
        caption_encoder_dropout=0.2,
        k=2,
    ):
        super(AutocolorizeUnetSegCapV2, self).__init__()
        self.in_dim = [512, 512, 256, 256]
        self.num_classes = num_classes
        self.n_hidden = d_hid
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, self.n_hidden, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16(affine={2: False, 8: False})
        self.segnet = SegNet(seg_ver)
        for i in xrange(len(with_spatial_attention)):
            if with_spatial_attention[i] != -1:
                print('Spation attention at FiLM {}'.format(i))
        self.film0 = FiLMx(self.n_hidden, self.in_dim[0], self.in_dim[0] / k, with_spatial_attention[0])
        self.film1 = FiLMx(self.n_hidden, self.in_dim[1], self.in_dim[1] / k, with_spatial_attention[1])
        self.film2 = FiLMx(self.n_hidden, self.in_dim[2], self.in_dim[2] / k, with_spatial_attention[2])
        self.film3 = FiLMx(self.n_hidden, self.in_dim[3], self.in_dim[3] / k, with_spatial_attention[3])
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                b0 = torch.cat((weights['cnn_0.7.weight'], weights['cnn_0.7.bias']), dim=-1)
                self.film0.dense.bias.data.copy_(b0)
                b3 = torch.cat((weights['cnn_2.4.weight'], weights['cnn_2.4.bias']), dim=-1)
                self.film3.dense.bias.data.copy_(b3)
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))            

    def forward(self, x, captions, caption_lens, return_attention=False):
        caption_features = self.caption_encoder(captions, caption_lens)
        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        if self.segnet.deconv_4 is not None:
            deconv_4 = self.segnet.deconv_4(vgg_32)
        else:
            deconv_4 = None
        block5 = self.unet.cnn_0[0: 8](vgg_32)
        block5, attention0 = self.film0(block5, caption_features)  # FiLM
        if not return_attention:
            del attention0
        block5 = self.unet.cnn_0[8](block5)  # ReLU
        if self.segnet.deconv_5 is not None:
            deconv_5 = self.segnet.deconv_5(block5)
        else:
            deconv_5 = None
        conv_0 = self.unet.cnn_0[9:](block5)
        del block5
        conv_0, attention1 = self.film1(conv_0, caption_features)  # FiLM
        if not return_attention:
            del attention1
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        if self.segnet.deconv_6 is not None:
            deconv_6 = self.segnet.deconv_6(concat_0)
        else:
            deconv_6 = None
        conv_1 = self.unet.cnn_1(concat_0)
        conv_1, attention2 = self.film2(conv_1, caption_features)  # FiLM
        if not return_attention:
            del attention2
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        if self.segnet.deconv_7 is not None:
            deconv_7 = self.segnet.deconv_7(concat_1)
        else:
            deconv_7 = None
        output = self.unet.cnn_2[: 5](concat_1)
        output, attention3 = self.film3(output, caption_features)  # FiLM
        if not return_attention:
            del attention3
        del caption_features
        output = self.unet.cnn_2[5:](output)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        if deconv_4 is None:
            if deconv_5 is None:
                seg_out = deconv_6
            else:
                seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        else:
            seg_out = torch.cat((deconv_4, deconv_5, deconv_6), dim=1)
        del deconv_4, deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        if return_attention:
            return output, seg_out, (attention0, attention1, attention2, attention3)
        else:
            return output, seg_out


class AutocolorizeUnetSegCapV3(nn.Module):
    def __init__(self, d_hid=512, d_emb=300, num_classes=313, train_vocab_embeddings=None, pretrained_weights='', caption_encoder_version='gru'):
        super(AutocolorizeUnetSegCapV3, self).__init__()
        self.n_hidden = d_hid
        self.num_classes = num_classes
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, d_hid, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16()
        self.segnet = SegNet()
        self.conv = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        utils.init_modules([self.conv])
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                self.conv.weight[:, : 512, :, :].data.copy_(weights['cnn_0.9.weight'])  # [out, in, k, k ]
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))

    def forward(self, x, captions, caption_lens):
        caption_features = self.caption_encoder(captions, caption_lens)
        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        block5 = self.unet.cnn_0[0: 9](vgg_32)
        deconv_5 = self.segnet.deconv_5(block5)
        caption_features = caption_features.unsqueeze_(-1).unsqueeze_(-1).expand(block5.size())
        block5 = torch.cat((block5, caption_features), dim=1)
        block5 = self.conv(block5)
        conv_0 = self.unet.cnn_0[10:](block5)
        del block5
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        deconv_6 = self.segnet.deconv_6(concat_0)
        conv_1 = self.unet.cnn_1(concat_0)
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        deconv_7 = self.segnet.deconv_7(concat_1)
        output = self.unet.cnn_2(concat_1)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        del deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        return output, seg_out


class AutocolorizeUnetSegCapV4(nn.Module):
    def __init__(self, d_hid=1024, d_emb=300, num_classes=313, train_vocab_embeddings=None, pretrained_weights='', caption_encoder_version='gru'):
        super(AutocolorizeUnetSegCapV4, self).__init__()
        self.n_hidden = d_hid
        self.num_classes = num_classes
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, d_hid, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.dense = nn.Linear(self.n_hidden, 512)
        self.unet = AutocolorizeVGG16()
        self.segnet = SegNet()
        self.conv = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        utils.init_modules([self.conv, self.dense])
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                self.conv.weight[:, : 512, :, :].data.copy_(weights['cnn_0.9.weight'])  # [out, in, k, k ]
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))

    def forward(self, x, captions, caption_lens):
        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        block5 = self.unet.cnn_0[0: 9](vgg_32)
        deconv_5 = self.segnet.deconv_5(block5)
        caption_features = self.caption_encoder(captions, caption_lens)
        caption_features = self.dense(caption_features)
        caption_features = caption_features.unsqueeze_(-1).unsqueeze_(-1).expand(block5.size())
        block5 = torch.cat((block5, caption_features), dim=1)
        del caption_features
        block5 = self.conv(block5)
        conv_0 = self.unet.cnn_0[10:](block5)
        del block5
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        deconv_6 = self.segnet.deconv_6(concat_0)
        conv_1 = self.unet.cnn_1(concat_0)
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        deconv_7 = self.segnet.deconv_7(concat_1)
        output = self.unet.cnn_2(concat_1)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        del deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        return output, seg_out


class AutocolorizeUnetSegCapV5(nn.Module):
    def __init__(self, d_hid=1024, d_emb=300, num_classes=313, train_vocab_embeddings=None, pretrained_weights='', caption_encoder_version='gru'):
        super(AutocolorizeUnetSegCapV5, self).__init__()
        self.n_hidden = d_hid
        self.in_dim = 256
        self.num_classes = num_classes
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, d_hid, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16(affine={7: False})
        self.segnet = SegNet()
        self.dense_film = nn.Linear(self.n_hidden, self.in_dim * 2) 
        self.dense_sa = nn.Linear(self.n_hidden, 256)
        self.conv_sa = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
        )
        self.film = FiLMWithAttn()
        utils.init_modules([self.dense_film, self.dense_sa, self.conv_sa])
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                b = torch.cat((weights['cnn_2.1.weight'], weights['cnn_2.1.bias']), dim=-1)
                self.dense_film.bias.data.copy_(b)
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))
        else:
            # Match affine in BN.
            b = torch.ones(self.in_dim * 2)
            b[self.in_dim:] = 0.
            self.dense_film.bias.data.copy_(b)

    def forward(self, x, captions, caption_lens):
        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        block5 = self.unet.cnn_0[0: 9](vgg_32)
        deconv_5 = self.segnet.deconv_5(block5)
        conv_0 = self.unet.cnn_0[9:](block5)
        del block5
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        deconv_6 = self.segnet.deconv_6(concat_0)
        conv_1 = self.unet.cnn_1(concat_0)
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        deconv_7 = self.segnet.deconv_7(concat_1)
        seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        del deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)

        caption_features = self.caption_encoder(captions, caption_lens)
        attn_features = self.dense_sa(caption_features)
        dense_film = self.dense_film(caption_features)
        del caption_features
        gammas, betas = torch.split(dense_film, self.in_dim, dim=-1)
        del dense_film
        attn_features = attn_features.unsqueeze_(-1).unsqueeze_(-1).expand(seg_out.size())
        spatial_attn = torch.cat((seg_out, attn_features), dim=1)
        del attn_features
        spatial_attn = self.conv_sa(spatial_attn)

        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        output = self.unet.cnn_2[: 2](concat_1)
        del concat_1
        output = self.film(output, gammas, betas, spatial_attn)
        del gammas, betas, spatial_attn
        output = self.unet.cnn_2[2:](output)
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)

        return output, seg_out 


class AutocolorizeUnetSegCapV6(nn.Module):
    def __init__(self, d_hid=1024, d_emb=300, num_classes=313, train_vocab_embeddings=None, pretrained_weights='', caption_encoder_version='gru'):
        super(AutocolorizeUnetSegCapV6, self).__init__()
        self.n_hidden = d_hid
        self.in_dim = [512, 512, 512, 256]
        self.num_classes = num_classes
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, d_hid, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16(affine={0: False, 3: False, 5: False, 7: False})
        self.segnet = SegNet(version=2)
        self.dense_film0 = nn.Linear(self.n_hidden, self.in_dim[0] * 2) 
        self.dense_film1 = nn.Linear(self.n_hidden, self.in_dim[1] * 2)
        self.dense_film2 = nn.Linear(self.n_hidden, self.in_dim[2] * 2)  
        self.dense_film3 = nn.Linear(self.n_hidden, self.in_dim[3] * 2) 
        self.film0 = FiLM()
        self.film1 = FiLM()
        self.film2 = FiLM()
        self.film3 = FiLM()
        utils.init_modules([self.dense_film0, self.dense_film1, self.dense_film2, self.dense_film3])
        # Match affine in BN.
        b0 = torch.ones(self.in_dim[0] * 2)
        b0[self.in_dim[0]:] = 0.
        self.dense_film0.bias.data.copy_(b0)
        b1 = torch.ones(self.in_dim[1] * 2)
        b1[self.in_dim[1]:] = 0.
        self.dense_film1.bias.data.copy_(b1)
        b2 = torch.ones(self.in_dim[2] * 2)
        b2[self.in_dim[2]:] = 0.
        self.dense_film2.bias.data.copy_(b2)
        b3 = torch.ones(self.in_dim[3] * 2)
        b3[self.in_dim[3]:] = 0.
        self.dense_film3.bias.data.copy_(b3)
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                b0 = torch.cat((weights['cnn_0.1.weight'], weights['cnn_0.1.bias']), dim=-1)
                self.dense_film0.bias.data.copy_(b0)
                b1 = torch.cat((weights['cnn_0.10.weight'], weights['cnn_0.10.bias']), dim=-1)
                self.dense_film0.bias.data.copy_(b1)
                b2 = torch.cat((weights['cnn_1.1.weight'], weights['cnn_1.1.bias']), dim=-1)
                self.dense_film0.bias.data.copy_(b2)
                b3 = torch.cat((weights['cnn_2.4.weight'], weights['cnn_2.4.bias']), dim=-1)
                self.dense_film3.bias.data.copy_(b3)
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))            

    def forward(self, x, captions, caption_lens):
        caption_features = self.caption_encoder(captions, caption_lens)
        dense_film0 = self.dense_film0(caption_features)
        dense_film1 = self.dense_film1(caption_features)
        dense_film2 = self.dense_film2(caption_features)
        dense_film3 = self.dense_film3(caption_features)
        gammas0, betas0 = torch.split(dense_film0, self.in_dim[0], dim=-1)
        gammas1, betas1 = torch.split(dense_film1, self.in_dim[1], dim=-1)
        gammas2, betas2 = torch.split(dense_film2, self.in_dim[2], dim=-1)
        gammas3, betas3 = torch.split(dense_film3, self.in_dim[3], dim=-1)
        del dense_film0, dense_film1, dense_film2, dense_film3

        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        deconv_4 = self.segnet.deconv_4(vgg_32)
        block5 = self.unet.cnn_0[0: 2](vgg_32)
        block5 = self.film0(block5, gammas0, betas0)  # FiLM
        del gammas0, betas0
        block5 = self.unet.cnn_0[2: 9](block5)  # ReLU
        deconv_5 = self.segnet.deconv_5(block5)
        block6 = self.unet.cnn_0[9: 11](block5)
        del block5
        block6 = self.film1(block6, gammas1, betas1)
        del gammas1, betas1
        block6 = self.unet.cnn_0[11:](block6)
        concat_0 = torch.cat((vgg_32, block6), dim=1)
        del vgg_32, block6
        deconv_6 = self.segnet.deconv_6(concat_0)
        block7 = self.unet.cnn_1[: 2](concat_0)
        del concat_0
        block7 = self.film2(block7, gammas2, betas2)  # FiLM
        del gammas2, betas2
        block7 = self.unet.cnn_1[2:](block7)
        concat_1 = torch.cat((vgg_22, block7), dim=1)
        del vgg_22, block7
        output = self.unet.cnn_2[: 2](concat_1)
        output = self.film3(output, gammas3, betas3)  # FiLM
        del gammas3, betas3
        output = self.unet.cnn_2[2:](output)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        seg_out = torch.cat((deconv_4, deconv_5, deconv_6), dim=1)
        del deconv_4, deconv_5, deconv_6
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        return output, seg_out


class AutocolorizeUnetSegCapV7(nn.Module):
    def __init__(self, d_hid=1024, d_emb=300, num_classes=313, train_vocab_embeddings=None, pretrained_weights='', caption_encoder_version='gru'):
        super(AutocolorizeUnetSegCapV7, self).__init__()
        self.n_hidden = d_hid
        self.in_dim = [512, 512, 512, 256]
        self.num_classes = num_classes
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, d_hid, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16(affine={0: False, 3: False, 5: False, 7: False})
        self.segnet = SegNet()
        self.dense_film0 = nn.Linear(self.n_hidden, self.in_dim[0] * 2) 
        self.dense_film1 = nn.Linear(self.n_hidden, self.in_dim[1] * 2)
        self.dense_film2 = nn.Linear(self.n_hidden, self.in_dim[2] * 2)  
        self.dense_film3 = nn.Linear(self.n_hidden, self.in_dim[3] * 2) 
        self.film0 = FiLM()
        self.film1 = FiLM()
        self.film2 = FiLM()
        self.film3 = FiLM()
        utils.init_modules([self.dense_film0, self.dense_film1, self.dense_film2, self.dense_film3])
        # Match affine in BN.
        b0 = torch.ones(self.in_dim[0] * 2)
        b0[self.in_dim[0]:] = 0.
        self.dense_film0.bias.data.copy_(b0)
        b1 = torch.ones(self.in_dim[1] * 2)
        b1[self.in_dim[1]:] = 0.
        self.dense_film1.bias.data.copy_(b1)
        b2 = torch.ones(self.in_dim[2] * 2)
        b2[self.in_dim[2]:] = 0.
        self.dense_film2.bias.data.copy_(b2)
        b3 = torch.ones(self.in_dim[3] * 2)
        b3[self.in_dim[3]:] = 0.
        self.dense_film3.bias.data.copy_(b3)
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                b0 = torch.cat((weights['cnn_0.1.weight'], weights['cnn_0.1.bias']), dim=-1)
                self.dense_film0.bias.data.copy_(b0)
                b1 = torch.cat((weights['cnn_0.10.weight'], weights['cnn_0.10.bias']), dim=-1)
                self.dense_film0.bias.data.copy_(b1)
                b2 = torch.cat((weights['cnn_1.1.weight'], weights['cnn_1.1.bias']), dim=-1)
                self.dense_film0.bias.data.copy_(b2)
                b3 = torch.cat((weights['cnn_2.4.weight'], weights['cnn_2.4.bias']), dim=-1)
                self.dense_film3.bias.data.copy_(b3)
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))            

    def forward(self, x, captions, caption_lens):
        caption_features = self.caption_encoder(captions, caption_lens)
        dense_film0 = self.dense_film0(caption_features)
        dense_film1 = self.dense_film1(caption_features)
        dense_film2 = self.dense_film2(caption_features)
        dense_film3 = self.dense_film3(caption_features)
        gammas0, betas0 = torch.split(dense_film0, self.in_dim[0], dim=-1)
        gammas1, betas1 = torch.split(dense_film1, self.in_dim[1], dim=-1)
        gammas2, betas2 = torch.split(dense_film2, self.in_dim[2], dim=-1)
        gammas3, betas3 = torch.split(dense_film3, self.in_dim[3], dim=-1)
        del dense_film0, dense_film1, dense_film2, dense_film3

        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        block5 = self.unet.cnn_0[: 2](vgg_32)
        block5 = self.film0(block5, gammas0, betas0)  # FiLM
        del gammas0, betas0
        block5 = self.unet.cnn_0[2: 9](block5)  # ReLU
        deconv_5 = self.segnet.deconv_5(block5)
        conv_0 = self.unet.cnn_0[9: 11](block5)
        del block5
        conv_0 = self.film1(conv_0, gammas1, betas1)  # FiLM
        conv_0 = self.unet.cnn_0[11:](conv_0)
        del gammas1, betas1
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        deconv_6 = self.segnet.deconv_6(concat_0)
        conv_1 = self.unet.cnn_1[: 2](concat_0)
        conv_1 = self.film2(conv_1, gammas2, betas2)  # FiLM
        del gammas2, betas2, concat_0
        conv_1 = self.unet.cnn_1[2:](conv_1)
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        deconv_7 = self.segnet.deconv_7(concat_1)
        output = self.unet.cnn_2[: 2](concat_1)
        output = self.film3(output, gammas3, betas3)  # FiLM
        del gammas3, betas3
        output = self.unet.cnn_2[2:](output)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        del deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        return output, seg_out   


class AutocolorizeUnetSegCapV8(nn.Module):
    def __init__(
        self, 
        d_hid=1024, 
        d_emb=300, 
        num_classes=313, 
        train_vocab_embeddings=None, 
        pretrained_weights='', 
        seg_ver=0, 
        with_spatial_attention=[-1, -1, -1, -1],
        caption_encoder_version='gru',
        caption_encoder_dropout=0.2,
        k=2,
    ):
        super(AutocolorizeUnetSegCapV8, self).__init__()
        self.in_dim = [512, 512, 256, 256]
        self.num_classes = num_classes
        self.n_hidden = d_hid
        self.caption_encoder = get_caption_encoder(caption_encoder_version, d_emb, self.n_hidden, len(train_vocab_embeddings), train_vocab_embeddings, caption_encoder_dropout)
        self.unet = AutocolorizeVGG16(affine={2: False, 8: False})
        self.segnet = SegNet(seg_ver)
        for i in xrange(len(with_spatial_attention)):
            if with_spatial_attention[i] != -1:
                print('Spation attention at FiLM {}'.format(i))
        self.film0 = FiLMx(self.n_hidden, self.in_dim[0], self.in_dim[0] / k, with_spatial_attention[0])
        self.film1 = FiLMx(self.n_hidden, self.in_dim[1], self.in_dim[1] / k, with_spatial_attention[1])
        self.film2 = FiLMx(self.n_hidden, self.in_dim[2], self.in_dim[2] / k, with_spatial_attention[2])
        self.film3 = FiLMx(self.n_hidden, self.in_dim[3], self.in_dim[3] / k, with_spatial_attention[3])
        if pretrained_weights != '':
            if os.path.isfile(pretrained_weights):
                print("=> loading pretrained weights '{}'".format(pretrained_weights))
                weights = torch.load(pretrained_weights)['state_dict']
                self.unet.load_state_dict(weights, strict=False)
                b0 = torch.cat((weights['cnn_0.7.weight'], weights['cnn_0.7.bias']), dim=-1)
                self.film0.dense.bias.data.copy_(b0)
                b3 = torch.cat((weights['cnn_2.4.weight'], weights['cnn_2.4.bias']), dim=-1)
                self.film3.dense.bias.data.copy_(b3)
            else:
                print("=> no weights found at '{}'".format(pretrained_weights))            

    def forward(self, x, captions, caption_lens, return_attention=False):
        caption_features = self.caption_encoder(captions, caption_lens)
        vgg_22 = self.unet.pretrained_vgg_22(x)
        vgg_32 = self.unet.pretrained_vgg_32(vgg_22)
        if self.segnet.deconv_4 is not None:
            deconv_4 = self.segnet.deconv_4(vgg_32)
        else:
            deconv_4 = None
        block5 = self.unet.cnn_0[0: 8](vgg_32)
        deconv_5 = self.segnet.deconv_5(block5)
        block5, attention0 = self.film0(block5, caption_features)  # FiLM
        if not return_attention:
            del attention0
        block5 = self.unet.cnn_0[8](block5)  # ReLU
        conv_0 = self.unet.cnn_0[9:](block5)
        del block5
        conv_0, attention1 = self.film1(conv_0, caption_features)  # FiLM
        if not return_attention:
            del attention1
        concat_0 = torch.cat((vgg_32, conv_0), dim=1)
        del vgg_32, conv_0
        deconv_6 = self.segnet.deconv_6(concat_0)
        conv_1 = self.unet.cnn_1(concat_0)
        conv_1, attention2 = self.film2(conv_1, caption_features)  # FiLM
        if not return_attention:
            del attention2
        del concat_0
        concat_1 = torch.cat((vgg_22, conv_1), dim=1)
        del vgg_22, conv_1
        if self.segnet.deconv_7 is not None:
            deconv_7 = self.segnet.deconv_7(concat_1)
        else:
            deconv_7 = None
        output = self.unet.cnn_2[: 5](concat_1)
        output, attention3 = self.film3(output, caption_features)  # FiLM
        if not return_attention:
            del attention3
        del caption_features
        output = self.unet.cnn_2[5:](output)
        del concat_1
        # Classifier
        output = self.unet.classifier(output)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        
        if deconv_4 is None:
            seg_out = torch.cat((deconv_5, deconv_6, deconv_7), dim=1)
        else:
            seg_out = torch.cat((deconv_4, deconv_5, deconv_6), dim=1)
        del deconv_4, deconv_5, deconv_6, deconv_7
        seg_out = self.segnet.seg_conv(seg_out)
        seg_out = self.segnet.seg_classifier(seg_out)
        seg_out = seg_out.permute(0, 2, 3, 1).contiguous().view(-1, 182)

        if return_attention:
            return output, seg_out, (attention0, attention1, attention2, attention3)
        else:
            return output, seg_out


def train(net, optimizer, epoch, img_save_folder, step=0, lambda_c=2., lambda_s=1.):
    print('Start training... epoch {}'.format(epoch))
    stime = time.time()    
    i = step
    val_it = iter(val_generator)

    for data_dict in train_generator:
        input_ims = Variable(data_dict['luma'].float().cuda(async=True))
        optimizer.zero_grad()            
        if with_cap:
            input_captions = Variable(data_dict['cap'].long().cuda(async=True))
            input_caption_lens = Variable(data_dict['cap_len'].long().cuda(async=True))
            color_outputs, seg_outputs = net(input_ims, input_captions, input_caption_lens)
            del input_captions, input_caption_lens
        else:
            color_outputs, seg_outputs = net(input_ims)
        del input_ims
        targets = Variable(data_dict['target'].float().cuda(async=True))
        priors = Variable(data_dict['color_prior'].float().cuda(async=True))
        if 'bbox' in data_dict:
            bboxes = Variable(data_dict['bbox'].float().cuda(async=True)).view(-1)
        else:
            bboxes = None
        loss_rb, loss_rb_nm, _ = _cross_entropy_soft(color_outputs, targets.view(-1, 313), prior=priors.view(-1), bbox=bboxes)
        if with_seg:
            labels = Variable(data_dict['label'].long().cuda(async=True)).view(-1)
            seg_loss, seg_loss_nm = _cross_entropy(seg_outputs, labels, seg_label_prior, train_dataset.ignore_label)
            total_loss = lambda_c * loss_rb + lambda_s * seg_loss
            del labels
        else:
            total_loss = loss_rb
        total_loss.backward()
        optimizer.step()
        del color_outputs, seg_outputs, targets, priors

        if i % 10 == 0:
            if with_seg:
                print('Eepoch {}, batch {}, loss rb {}, loss rb nm {}, seg {}, seg nm {}, total {}, time: {} s'.format(
                    epoch, i, loss_rb.item(), loss_rb_nm.item(), seg_loss.item(), seg_loss_nm.item(), total_loss.item(), time.time() - stime))
            else:
                print('Eepoch {}, batch {}, loss rb {}, loss rb nm {} time: {} s'.format(
                    epoch, i, loss_rb.item(), loss_rb_nm.item(), time.time() - stime))
            stime = time.time()

            if i % 200 == 0:  # args.logs:
                net.eval()
                with torch.no_grad():
                    val_losses = []
                    val_losses_rb = []
                    if with_seg:
                        val_seg_losses = []
                    for j in xrange(20):
                        try:
                            val_data_dict = next(val_it)
                        except StopIteration:
                            val_it = iter(val_generator)
                            val_data_dict = next(val_it)
                        val_input_ims = Variable(val_data_dict['luma'].float().cuda(async=True))
                        if with_cap:
                            val_input_captions = Variable(val_data_dict['cap'].long().cuda(async=True))
                            val_input_caption_lens = Variable(val_data_dict['cap_len'].long().cuda(async=True))
                            val_color_outputs, val_seg_outputs, val_attentions = net(val_input_ims, val_input_captions, val_input_caption_lens, True)
                        else:
                            val_color_outputs, val_seg_outputs = net(val_input_ims)
                        if with_seg:
                            val_labels = Variable(val_data_dict['label'].long().cuda(async=True)).view(-1)
                            _, val_seg_loss_nm = _cross_entropy(val_seg_outputs, val_labels, seg_label_prior, train_dataset.ignore_label)
                            val_seg_losses.append(val_seg_loss_nm.item())
                            
                        val_color_priors = Variable(val_data_dict['color_prior'].float().cuda(async=True))
                        val_targets = Variable(val_data_dict['target'].float().cuda(async=True))
                        _, val_loss_rb_nm, val_loss = _cross_entropy_soft(val_color_outputs, val_targets.view(-1, 313), prior=val_color_priors.view(-1))
                        val_losses.append(val_loss.item())
                        val_losses_rb.append(val_loss_rb_nm.item())
                        
                    val_loss_mean = np.mean(val_losses)
                    val_loss_rb = np.mean(val_losses_rb)
                    if with_seg:
                        val_seg_loss = np.mean(val_seg_losses)
                        val_total_loss = lambda_c * val_loss_rb + lambda_s * val_seg_loss
                        print('Eval loss at epoch {}, batch {}, loss {}, loss rb {}, seg loss {}, total loss {}'.format(epoch, i, val_loss_mean, val_loss_rb, val_seg_loss, val_total_loss))
                    else:
                        print('Eval loss at epoch {}, batch {}, loss {}, loss rb {}'.format(epoch, i, val_loss_mean, val_loss_rb))

                    val_color_outputs *= 2.63  # Annealed.
                    dec_inp = nn.Softmax(dim=1)(val_color_outputs)  # N*H*WxC
                    AB_vals = dec_inp.mm(cuda_cc)  # N*H*WxC
                    # Reshape and select last image of batch.
                    AB_val = AB_vals.view(args.batch_size, 56, 56, 2)[-1].data.cpu().numpy()[:, :, :]
                    if 'bbox' in val_data_dict:
                        bbox = val_data_dict['bbox'][-1]
                        bbox -= 1.
                        bbox /= 2.
                        AB_val *= bbox
                    AB_val = cv2.resize(AB_val, (224, 224), interpolation=cv2.INTER_CUBIC)
                    val_img_l = val_data_dict['l' if input_as_rgb else 'luma'][-1].numpy()      
                    val_img_l = (val_img_l + 1.) * 50.
                    val_img_l = np.transpose(val_img_l, (1, 2, 0))
                    img_dec = np.dstack((val_img_l, AB_val))
                    if with_cap:
                        word_list = val_data_dict['cap'][-1, : val_data_dict['cap_len'][-1]].numpy()
                        words = '_'.join(vrev.get(w, 'unk') for w in word_list)
                        io.imsave('{}/{}_{}_{}.jpg'.format(img_save_folder, epoch, i, words), color.lab2rgb(img_dec))
                        for k in xrange(len(val_attentions)):
                            att = val_attentions[k]
                            if att is not None:
                                a = att[-1].data.cpu().numpy()
                                a = utils.normalize(a)
                                plt.imsave('{}/{}_{}_{}_a{}.png'.format(img_save_folder, epoch, i, words, k), a)
                        del val_attentions
                    else:
                        io.imsave('{}/{}_{}.jpg'.format(img_save_folder, epoch, i), color.lab2rgb(img_dec))
                del val_data_dict
                net.train()

                step = epoch * epoch_size + i
                train_loss_nm_sum = _scalar_summary('train/loss_rb_nm', loss_rb_nm.item())
                train_loss_rb_sum = _scalar_summary('train/loss_rb', loss_rb.item())
                val_loss_sum = _scalar_summary('val/loss', np.mean(val_losses))
                val_loss_rb_sum = _scalar_summary('val/loss_rb', val_loss_rb)
                summary_writer.add_summary(train_loss_nm_sum, step)
                summary_writer.add_summary(train_loss_rb_sum, step)
                summary_writer.add_summary(val_loss_sum, step)
                summary_writer.add_summary(val_loss_rb_sum, step)
                if with_seg:
                    train_seg_loss_sum = _scalar_summary('train/seg_loss', seg_loss.item())
                    train_total_loss_sum = _scalar_summary('train/total_loss', total_loss.item())
                    val_seg_loss_sum = _scalar_summary('val/seg_loss', val_seg_loss)
                    val_total_loss_sum = _scalar_summary('val/total_loss', val_total_loss)
                    summary_writer.add_summary(train_seg_loss_sum, step)
                    summary_writer.add_summary(train_total_loss_sum, step)
                    summary_writer.add_summary(val_seg_loss_sum, step)
                    summary_writer.add_summary(val_total_loss_sum, step)

            if i % 2000 == 0: 
                torch.save({
                    'epoch': epoch,
                    'iter': i,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_rb.item(),
                }, args.model_save_folder + '/model_' + str(epoch) + '_' + str(i) + '.pth.tar')
        
        i += 1
    return net


def _cross_entropy_soft(input, target, prior=None, bbox=None):
    """ Cross entropy that accepts soft targets
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    ce = torch.sum(-target * logsoftmax(input), dim=1)  # N*H*W
    if prior is not None:
        new_ce = ce * prior
        if bbox is not None:
            nnew_ce = new_ce * bbox
        else:
            nnew_ce = new_ce
        return torch.mean(nnew_ce), torch.sum(new_ce) / torch.sum(prior), torch.mean(ce)
    return None, None, torch.mean(ce)


def _cross_entropy(input, target, weights=None, ignore_index=-100, bbox=None):
    idx = target != ignore_index
    t = target[idx]
    logsoftmax = nn.LogSoftmax(dim=1)
    logit = -logsoftmax(input[idx])[torch.arange(len(t)).long(), t]  # _x1
    if weights is not None:
        w = weights[t]
        logit *= w
        if bbox is not None:
            nlogit = logit * bbox[idx]
        else:
            nlogit = logit
        return torch.mean(nlogit), torch.sum(logit) / torch.sum(w)
    return torch.mean(logit), None


def _lr_scheduler(optimizer, epoch, lr_decay=0.316, lr_decay_epoch=7, schedule=None):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if schedule is None:
        if epoch == 0 or epoch % lr_decay_epoch:
            return optimizer
    else:
        if len(schedule) and epoch != schedule[0]:
            return optimizer
        else:
            schedule.pop(0)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    print(optimizer)

    return optimizer


def _scalar_summary(tag, value):
    return Summary(value=[Summary.Value(tag=tag, simple_value=value)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='resnet coco colorization')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--start-epoch', '-se', type=int, default=0, help='starting epoch')
    parser.add_argument('--end-epoch', '-ee', type=int, default=30, help='ending epoch')
    parser.add_argument('--gpuid', '-g', default='0', type=str, help='which gpu to use')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='batch size')
    parser.add_argument('--model', '-m', default=0, type=int, help='Model type.')
    parser.add_argument('--model_save_folder', '-d', default='/srv/glusterfs/xieya/tmp/', help='prefix of the model save file')
    parser.add_argument('--nclasses', '-nc', type=int, default=313, help='Number of classes.')
    parser.add_argument('--grid_file', default='resources/ab_grid.npy', type=str, help='Grid file.')
    parser.add_argument('--weights', default='', type=str, help='Pretrained weights.')
    parser.add_argument('--pre_weights', '-p', default='', type=str, help='Pretrained weights.')
    parser.add_argument('--weight_decay', '-wd', default=1e-5, type=float, help='Weight decay parameter.')
    parser.add_argument('--decay', default=0, type=int, help='Decay at current epoch.')
    parser.add_argument('--seg_ver', default=0, type=int, help='SegNet version.')
    parser.add_argument('--empty_cap', default=0, type=int, help='With empty cap.')
    parser.add_argument('--gru', default=1, type=int, help='Use GRU or BiLSTM.')
    parser.add_argument('--vg', default=0, type=int, help='Use regions from visual genome.')
    parser.add_argument('--dropout', default=0.2, type=float, help='Embedding dropout.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    torch.backends.cudnn.benchmark = True

    # initialize quantized LAB encoder
    lookup_enc = utils.LookupEncode(args.grid_file)
    cuda_cc = Variable(torch.from_numpy(lookup_enc.cc).float().cuda())

    summary_writer = summary.FileWriter(args.model_save_folder)

    print('Model {}'.format(args.model))
    print('Learning rate {}'.format(args.lr))
    print('Weight decay {}'.format(args.weight_decay))
    schedule = None
    with_cap = args.model in [2, 3, 4, 5, 6, 7, 8, 9, 10]
    with_vg = args.vg == 1
    if with_cap:
        caption_encoder_version = 'gru' if args.gru == 1 else 'lstm'
        caption_encoder_dropout = args.dropout
        train_vocab = pickle.load(open('/srv/glusterfs/xieya/data/language/vocabulary.p', 'r'))
        train_vocab_embeddings = pickle.load(open('/srv/glusterfs/xieya/data/language/embedding.p', 'rb'))
        # if with_vg:
        #     train_vocab = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/vocabulary.p', 'r'))
        #     train_vocab_embeddings = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/embedding.p', 'rb'))
        # else:
        #     train_vocab = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/vocabulary.p', 'r'))
        #     train_vocab_embeddings = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/embedding.p', 'rb'))
        vrev = dict((v, k) for (k, v) in train_vocab.iteritems())
    if args.model == 0:
        print('VGG with segmentation')
        net = AutocolorizeVGGWithSeg()
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 1:
        print('Unet with segmentation')
        net = AutocolorizeUnetWithSeg()
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.segnet.deconv_5.parameters()) + list(net.segnet.deconv_6.parameters()) + list(net.segnet.deconv_7.parameters()) + list(net.segnet.seg_conv.parameters()) + list(net.segnet.seg_classifier.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 2:
        print('Unet with caption')
        net = AutocolorizeUnetCap(
            train_vocab_embeddings=train_vocab_embeddings, 
            pretrained_weights=args.pre_weights, 
            caption_encoder_version=caption_encoder_version,
            with_spatial_attention=[-1, -1, -1, -1],
            k=2,
        )
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.caption_encoder.parameters()) + list(net.film0.parameters()) + list(net.film1.parameters()) + list(net.film2.parameters()) + list(net.film3.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 3:
        print('Unet seg with caption')
        net = AutocolorizeUnetSegCap(train_vocab_embeddings=train_vocab_embeddings, pretrained_weights=args.pre_weights, caption_encoder_version=caption_encoder_version)
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.segnet.deconv_5.parameters()) + list(net.segnet.deconv_6.parameters()) + list(net.segnet.deconv_7.parameters()) + list(net.segnet.seg_conv.parameters()) + list(net.segnet.seg_classifier.parameters())},
                {'params': list(net.caption_encoder.parameters()) + list(net.dense_film.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 4:
        print('Unet seg v{} with 4 FiLM blocks'.format(args.seg_ver))
        net = AutocolorizeUnetSegCapV2(
            train_vocab_embeddings=train_vocab_embeddings, 
            pretrained_weights=args.pre_weights, 
            seg_ver=args.seg_ver, 
            caption_encoder_version=caption_encoder_version,
            caption_encoder_dropout=caption_encoder_dropout,
        )
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': net.segnet.parameters()},
                {'params': list(net.caption_encoder.parameters()) + list(net.film0.parameters()) + list(net.film1.parameters()) + list(net.film2.parameters()) + list(net.film3.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 5:
        print('Unet seg with 1 concat')
        net = AutocolorizeUnetSegCapV3(train_vocab_embeddings=train_vocab_embeddings, pretrained_weights=args.pre_weights, caption_encoder_version=caption_encoder_version)
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.segnet.deconv_5.parameters()) + list(net.segnet.deconv_6.parameters()) + list(net.segnet.deconv_7.parameters()) + list(net.segnet.seg_conv.parameters()) + list(net.segnet.seg_classifier.parameters())},
                {'params': list(net.caption_encoder.parameters()) + list(net.conv.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 6:
        print('Unet seg with 1 FiLM as block 8 and spatial attention')
        net = AutocolorizeUnetSegCapV5(train_vocab_embeddings=train_vocab_embeddings, pretrained_weights=args.pre_weights, caption_encoder_version=caption_encoder_version)
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.segnet.deconv_5.parameters()) + list(net.segnet.deconv_6.parameters()) + list(net.segnet.deconv_7.parameters()) + list(net.segnet.seg_conv.parameters()) + list(net.segnet.seg_classifier.parameters())},
                {'params': list(net.caption_encoder.parameters()) + list(net.dense_film.parameters()) + list(net.dense_sa.parameters()) + list(net.conv_sa.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 7:
        print('Unet seg with 1 concat + dense')
        net = AutocolorizeUnetSegCapV4(train_vocab_embeddings=train_vocab_embeddings, pretrained_weights=args.pre_weights, caption_encoder_version=caption_encoder_version)
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.segnet.deconv_5.parameters()) + list(net.segnet.deconv_6.parameters()) + list(net.segnet.deconv_7.parameters()) + list(net.segnet.seg_conv.parameters()) + list(net.segnet.seg_classifier.parameters())},
                {'params': list(net.caption_encoder.parameters()) + list(net.conv.parameters()) + list(net.dense.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 8:
        print('Unet seg v2 with 4 FiLM blocks at the front of each block')
        net = AutocolorizeUnetSegCapV6(train_vocab_embeddings=train_vocab_embeddings, pretrained_weights=args.pre_weights, caption_encoder_version=caption_encoder_version)
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.segnet.deconv_4.parameters()) + list(net.segnet.deconv_5.parameters()) + list(net.segnet.deconv_6.parameters()) + list(net.segnet.seg_conv.parameters()) + list(net.segnet.seg_classifier.parameters())},
                {'params': list(net.caption_encoder.parameters()) + list(net.dense_film0.parameters()) + list(net.dense_film1.parameters()) + list(net.dense_film2.parameters()) + list(net.dense_film3.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 9:
        print('Unet seg with 4 FiLM at front')
        net = AutocolorizeUnetSegCapV7(train_vocab_embeddings=train_vocab_embeddings, pretrained_weights=args.pre_weights, caption_encoder_version=caption_encoder_version)
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': list(net.segnet.deconv_5.parameters()) + list(net.segnet.deconv_6.parameters()) + list(net.segnet.deconv_7.parameters()) + list(net.segnet.seg_conv.parameters()) + list(net.segnet.seg_classifier.parameters())},
                {'params': list(net.caption_encoder.parameters()) + list(net.dense_film0.parameters()) + list(net.dense_film1.parameters()) + list(net.dense_film2.parameters()) + list(net.dense_film3.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 10:
        print('Unet seg v{} with 4 FiLM blocks with spatial attention'.format(args.seg_ver))
        net = AutocolorizeUnetSegCapV2(
            train_vocab_embeddings=train_vocab_embeddings, 
            pretrained_weights=args.pre_weights, 
            seg_ver=args.seg_ver,
            with_spatial_attention=[0, 0, -1, 0],
            caption_encoder_version=caption_encoder_version,
            k=2,
        )
        net.cuda()
        optimizer = torch.optim.Adam(
            [
                {'params': net.unet.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.unet.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.unet.cnn_0.parameters()) + list(net.unet.cnn_1.parameters()) + list(net.unet.cnn_2.parameters()) + list(net.unet.classifier.parameters())},
                {'params': net.segnet.parameters()},
                {'params': list(net.caption_encoder.parameters()) + list(net.film0.parameters()) + list(net.film1.parameters()) + list(net.film2.parameters()) + list(net.film3.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
        # schedule = [5, 8, 11]

    start_step = 0
    if args.weights != '':
        # Load pretrained weights.
        if os.path.isfile(args.weights):
            print("=> loading pretrained weights '{}'".format(args.weights))
            weights = torch.load(args.weights)
            net.load_state_dict(weights['state_dict'])
            optimizer.load_state_dict(weights['optimizer'])
            args.start_epoch = weights['epoch']
            start_step = weights['iter'] + 1
        else:
            print("=> no weights found at '{}'".format(args.weights))

    print(net)
    print(optimizer)

    input_as_rgb = args.model in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    with_seg = args.model in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    with_empty_cap = args.empty_cap == 1
    with_cap_filter = args.model in [0]
    if with_cap:
        print('With caption')
    if with_vg:
        print('Visual genome')
    train_dataset = CocoStuff164k(
        root='/scratch/xieya', 
        split='train2017', 
        as_rgb=input_as_rgb, 
        with_cap=with_cap, 
        with_seg=with_seg, 
        with_empty_cap=with_empty_cap, 
        with_vg=with_vg,
        with_cap_filter=with_cap_filter,
    )
    val_dataset = CocoStuff164k(
        root='/scratch/xieya', 
        split='val2017', 
        as_rgb=input_as_rgb, 
        with_cap=with_cap, 
        with_seg=with_seg, 
        random_crop=False, 
        with_vg=with_vg,
        with_cap_filter=with_cap_filter,
    )
    train_generator = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
    val_generator = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_size = len(train_dataset)
    epoch_size = train_size / args.batch_size

    alpha = 1.
    gamma = 0.5
    seg_label_prior = Variable(torch.from_numpy(
        utils.prior_boosting('/srv/glusterfs/xieya/prior/label_probs.npy', alpha, gamma)).float().cuda())
                                                                                      
    img_save_folder = args.model_save_folder
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)                                    

    print('Start training from epoch {}'.format(args.start_epoch))
    if args.decay == 1:
        decay_epoch = args.start_epoch
    else:
        decay_epoch = 100
    for epoch in range(args.start_epoch, args.end_epoch):
        optimizer = _lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=decay_epoch, schedule=schedule)
        decay_epoch = 100
        net = train(net, optimizer, epoch, img_save_folder, start_step)
        start_step = 0
