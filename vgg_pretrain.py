from __future__ import print_function

import argparse
import cv2
from datasets import imagenetDataset
from language_support import CaptionEncoderLSTM, FiLM
import numpy as np
import os
from skimage import color, io
from tensorflow import summary, Summary
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms
from utils import prior_boosting, init_modules, LookupEncode


class AutocolorizeVGG(nn.Module):
    def __init__(self, vocab_size, d_hid=256, d_emb=300, num_classes=625, train_vocab_embeddings=None):
        super(AutocolorizeVGG, self).__init__()
        self.n_lstm_hidden = d_hid
        self.in_dims = [64, 128, 256, 512, 512, 512, 512, 256]
        self.num_classes = num_classes
        self.caption_encoder = CaptionEncoderLSTM(d_emb, d_hid, vocab_size, train_vocab_embeddings)

        self.dense_film1 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[0] * 2)
        self.dense_film2 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[1] * 2)
        self.dense_film3 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[2] * 2)
        self.dense_film4 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[3] * 2)
        self.dense_film5 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[4] * 2)
        self.dense_film6 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[5] * 2)
        self.dense_film7 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[6] * 2)
        self.dense_film8 = nn.Linear(self.n_lstm_hidden * 2, self.in_dims[7] * 2)        

        self.film1 = FiLM()
        self.film2 = FiLM()
        self.film3 = FiLM()
        self.film4 = FiLM()
        self.film5 = FiLM()
        self.film6 = FiLM()
        self.film7 = FiLM()
        self.film8 = FiLM()

        # 224
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 112
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # 56
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # 28
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        # 28
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        n_dim = 512 if self.num_classes == 625 else 256
        self.conv8_1 = nn.ConvTranspose2d(512, n_dim, kernel_size=4, stride=2, padding=1)
        self.conv8_2 = nn.Conv2d(n_dim, n_dim, kernel_size=3, stride=1, padding=1)
        self.conv8_3 = nn.Conv2d(n_dim, n_dim, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(n_dim, self.num_classes, kernel_size=1, stride=1)

        # self.trainable_params = {}
        # key = 0
        # for m in self.modules():
        #     if isinstance(m, (nn.Linear)):
        #         for p in m.parameters():
        #             p.requires_grad = False
        #     elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         self.trainable_params[key] = m.parameters()
        #         key += 1

    def forward(self, x):
        # x /= 50.
        # x -= 1.
        # x = x.permute(0, 3, 1, 2)
        # Block 1
        block_idx = 0
        output = F.relu(self.conv1_1(x))
        output = F.relu(self.conv1_2(output))
        output = self.bn1(output)

        # Block 2
        block_idx += 1
        output = F.relu(self.conv2_1(output))
        output = F.relu(self.conv2_2(output))
        output = self.bn2(output)

        # Block 3
        block_idx += 1
        output = F.relu(self.conv3_1(output))
        output = F.relu(self.conv3_2(output))
        output = F.relu(self.conv3_3(output))
        output = self.bn3(output)

        # Block 4
        block_idx += 1
        output = F.relu(self.conv4_1(output))
        output = F.relu(self.conv4_2(output))
        output = F.relu(self.conv4_3(output))
        output = self.bn4(output)

        # Block 5
        block_idx += 1
        output = F.relu(self.conv5_1(output))
        output = F.relu(self.conv5_2(output))
        output = F.relu(self.conv5_3(output))
        output = self.bn5(output)

        # Block6
        block_idx += 1
        output = F.relu(self.conv6_1(output))
        output = F.relu(self.conv6_2(output))
        output = F.relu(self.conv6_3(output))
        output = self.bn6(output)

        # Block7
        block_idx += 1
        output = F.relu(self.conv7_1(output))
        output = F.relu(self.conv7_2(output))
        output = F.relu(self.conv7_3(output))
        output = self.bn7(output)

        # Block8
        block_idx += 1
        output = F.relu(self.conv8_1(output))
        output = F.relu(self.conv8_2(output))
        output = F.relu(self.conv8_3(output))

        # Classifier
        output = self.classifier(output)

        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, self.num_classes)
        return output, None


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


def train(net, optimizer, epoch, prior_probs, img_save_folder, step=0):
    stime = time.time()    
    i = step
    val_it = iter(val_generator)

    for data_dict, targets in train_generator:
        input_ims = Variable(data_dict['l_rgb'].cuda(async=True))
        optimizer.zero_grad()
        output = net(input_ims)
        del input_ims
        if use_soft:
            target = Variable(targets.float().cuda(async=True))
            prior = Variable(data_dict['p'].float().cuda(async=True))
            loss_rb, loss_rb_nm = loss_fn(output, target.view(-1, 313), prior=prior.view(-1))
        else:
            target = Variable(targets.long().cuda(async=True))
            loss_rb = loss_rb_fn(output, target.view(-1))
        loss_rb.backward()
        optimizer.step()

        if i % 10 == 0:
            if use_soft:
                print('Eepoch {}, batch {}, loss rb {}, loss rb nm {}, time: {} s'.format(epoch, i, loss_rb.item(), loss_rb_nm, time.time() - stime))
            else:
                print('Eepoch {}, batch {}, loss rb {}, time: {} s'.format(epoch, i, loss_rb.item(), time.time() - stime))
            stime = time.time()

            if i % 200 == 0:  # args.logs:
                net.eval()
                with torch.no_grad():
                    val_losses = []
                    val_losses_rb = []
                    for j in xrange(20):
                        try:
                            val_data_dict, val_targets = next(val_it)
                        except StopIteration:
                            val_it = iter(val_generator)
                            val_data_dict, val_targets = next(val_it)
                        val_input_ims = Variable(val_data_dict['l_rgb'].cuda(async=True))
                        val_output = net(val_input_ims)
                        if use_soft:
                            val_prior = Variable(val_data_dict['p'].float().cuda(async=True))
                            val_target = Variable(val_targets.float().cuda(async=True))
                            val_loss, _ = loss_fn(val_output, val_target.view(-1, 313))
                            _, val_loss_rb = loss_fn(val_output, val_target.view(-1, 313), prior=val_prior.view(-1))
                        else:
                            val_target = Variable(val_targets.long().cuda(async=True))
                            val_loss = loss_fn(val_output, val_target.view(-1)).item()
                            val_loss_rb = loss_rb_fn(val_output, val_target.view(-1)).item()
                        val_losses.append(val_loss)
                        val_losses_rb.append(val_loss_rb)
                    if use_attention:
                        gamma_val = net.attention_0.gamma.item()
                        print('Eval loss at epoch {}, batch {}, loss {}, loss rb {}, gamma {}'.format(epoch, i, np.mean(val_losses), np.mean(val_losses_rb), gamma_val))
                    else:
                        print('Eval loss at epoch {}, batch {}, loss {}, loss rb {}'.format(epoch, i, np.mean(val_losses), np.mean(val_losses_rb)))   
                    val_output *= 2.63  # Annealed.
                    dec_inp = nn.Softmax(dim=1)(val_output)  # N*H*WxC
                    AB_vals = dec_inp.mm(cuda_cc)  # N*H*WxC
                    # Reshape and select last image of batch.
                    AB_val = AB_vals.view(args.batch_size, 56, 56, 2)[-1].data.cpu().numpy()[:, :, :]
                    AB_val = cv2.resize(AB_val, (224, 224),
                                        interpolation=cv2.INTER_CUBIC)
                    val_img_l = val_data_dict['l'][-1].numpy()          
                    img_dec = np.dstack((np.expand_dims(val_img_l, axis=2), AB_val))
                    io.imsave('{}/{}_{}.jpg'.format(img_save_folder, epoch, i), color.lab2rgb(img_dec))
                net.train()

                step = epoch * epoch_size + i
                if use_soft:
                    train_loss_nm_sum = scalar_summary('train/loss_rb_nm', loss_rb_nm)
                    summary_writer.add_summary(train_loss_nm_sum, step)
                train_loss_rb_sum = scalar_summary('train/loss_rb', loss_rb.item())
                val_loss_sum = scalar_summary('val/loss', np.mean(val_losses))
                val_loss_rb_sum = scalar_summary('val/loss_rb', np.mean(val_losses_rb))
                summary_writer.add_summary(train_loss_rb_sum, step)
                summary_writer.add_summary(val_loss_sum, step)
                summary_writer.add_summary(val_loss_rb_sum, step)
                if use_attention:
                    gamma_sum = scalar_summary('tarin/gamma', gamma_val)
                    summary_writer.add_summary(gamma_sum, step)

            if i % 2000 == 0: 
                torch.save({
                    'epoch': epoch,
                    'iter': i,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_rb.item(),
                }, args.model_save_folder + '/model_' + str(epoch) + '_' + str(i) + '.pth.tar')
        
        del loss_rb

        i += 1
        if i > epoch_size:
            print("Epoch finished.")
            break
    return net


def lr_scheduler(optimizer, epoch, lr_decay=0.316, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch == 0 or epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    print(optimizer)

    return optimizer


def _set_lr(optimizer, lr, wd=None):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if wd is not None:
            param_group['weight_decay'] = wd
    return optimizer


def scalar_summary(tag, value):
    return Summary(value=[Summary.Value(tag=tag, simple_value=value)])


def cross_entropy_soft(input, target, prior=None, size_average=True, weight=None):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        ce = torch.sum(-target * logsoftmax(input), dim=1)  # N*H*W
        if weight is not None:
            maxv, argmax = torch.max(target, 1)
            w = weight[argmax]
            ce *= w
            return torch.mean(ce), (torch.sum(ce) / torch.sum(w)).item()
        if prior is not None:
            ce *= prior
            return torch.mean(ce), (torch.sum(ce) / torch.sum(prior)).item()
        return torch.mean(ce), None
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1)), None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='resnet coco colorization')
    parser.add_argument('--lr', default=0.00316, type=float, help='learning rate')
    parser.add_argument('--start-epoch', '-se', type=int, default=0, help='starting epoch')
    parser.add_argument('--end-epoch', '-ee', type=int, default=30, help='ending epoch')
    parser.add_argument('--gpuid', '-g', default='0', type=str, help='which gpu to use')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='batch size')
    parser.add_argument('--model_save_folder', '-d', default='/srv/glusterfs/xieya/tmp/', help='prefix of the model save file')
    parser.add_argument('--nclasses', '-nc', type=int, default=313, help='Number of classes.')
    parser.add_argument('--grid_file', default='/home/xieya/colorization-tf/resources/pts_in_hull.npy', type=str, help='Grid file.')
    parser.add_argument('--prior_file', default='/home/xieya/colorization-tf/resources/prior_probs_smoothed.npy', type=str, help='Priors file.')
    parser.add_argument('--weights', default='', type=str, help='Pretrained weights.')
    parser.add_argument('--pre_weights', default='', type=str, help='Pretrained weights.')
    parser.add_argument('--attention', '-a', default='', type=str, help='Use self attention.')
    parser.add_argument('--soft', '-s', default='1', type=str, help='Use soft target.')
    parser.add_argument('--weight_decay', '-wd', default=1e-5, type=float, help='Weight decay parameter.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    torch.backends.cudnn.benchmark = True
    use_attention = args.attention == '1'
    use_soft = args.soft == '1'
    print('Learning rate {}'.format(args.lr))
    print('Weight decay {}'.format(args.weight_decay))     

    # initialize quantized LAB encoder
    lookup_enc = LookupEncode(args.grid_file)
    cuda_cc = Variable(torch.from_numpy(lookup_enc.cc).float().cuda())

    # color rebalancing
    alpha = 1.
    gamma = 0.5
    gradient_prior_factor = Variable(torch.from_numpy(
        prior_boosting(args.prior_file, alpha, gamma)).float().cuda())

    if use_soft:
        loss_fn = cross_entropy_soft
    else:
        loss_rb_fn = nn.CrossEntropyLoss(weight=gradient_prior_factor)
        loss_fn = nn.CrossEntropyLoss()

    if use_attention:
        print('Pretrained Unet with self attention.')
        net = AutocolorizeVGG16Attention(vgg_weights=args.pre_weights)
        optimizer = torch.optim.Adam(
            [
                {'params': net.vgg16.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.vgg16.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.vgg16.cnn_0.parameters()) + list(net.vgg16.cnn_1.parameters()) + list(net.vgg16.cnn_2.parameters()) + list(net.vgg16.classifier.parameters())},
                {'params': net.attention_0.parameters()}
            ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('Pretrained Unet.')
        net = AutocolorizeVGG16()
        optimizer = torch.optim.Adam(
            [
                {'params': net.pretrained_vgg_22.parameters(), 'lr': args.lr / 100.},
                {'params': net.pretrained_vgg_32.parameters(), 'lr': args.lr / 10.},
                {'params': list(net.cnn_0.parameters()) + list(net.cnn_1.parameters()) + list(net.cnn_2.parameters()) + list(net.classifier.parameters())},
            ], lr=args.lr, weight_decay=args.weight_decay)
    net.cuda()

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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trs1 = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(224, interpolation=3),
        transforms.RandomCrop(224),  # RandomResizedCrop?
        transforms.RandomHorizontalFlip()])
    trs2 = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    val_trs1 = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224)])
    val_trs2 = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_dataset = imagenetDataset('/srv/glusterfs/xieya/data/imagenet1k_uncompressed/train224.txt', args.grid_file, trs1=trs1, trs2=trs2, soft=use_soft)
    val_dataset = imagenetDataset('/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val224.txt', args.grid_file, training=False, trs1=val_trs1, trs2=val_trs2, soft=use_soft)
    train_generator = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
    val_generator = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_size = len(train_dataset)
    epoch_size = train_size / args.batch_size

    summary_writer = summary.FileWriter(args.model_save_folder)
                                                                                      
    img_save_folder = args.model_save_folder
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)                                    

    print('Start training from epoch {}'.format(args.start_epoch))
    for epoch in range(args.start_epoch, args.end_epoch):
        optimizer = lr_scheduler(optimizer, epoch)
        net = train(net, optimizer, epoch, gradient_prior_factor, img_save_folder, start_step)
        start_step = 0
