import argparse
import cv2
import math
import numpy as np
import os
import pickle
import random
from skimage import color, img_as_ubyte, io
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

from datasets import CocoStuff164k, imagenetDataset
from vgg_pretrain import AutocolorizeVGG16Attention
from vgg_with_segmentation import AutocolorizeVGGWithSeg, AutocolorizeUnetWithSeg, AutocolorizeUnetSegCap, AutocolorizeUnetSegCapV2, \
    AutocolorizeUnetSegCapV3, AutocolorizeUnetSegCapV5, AutocolorizeUnetSegCapV4, AutocolorizeUnetSegCapV6, \
    AutocolorizeUnetSegCapV7 , AutocolorizeUnetCap
import utils

_OUT_DIR = '/srv/glusterfs/xieya/image/color/segcap_1'
_VIDEO_INPUT_FOLDER = '/srv/glusterfs/xieya/data/DAVIS/JPEGImages/Full-Resolution/bear'
_VIDEO_OUTPUT_FOLDER = '/srv/glusterfs/xieya/video/bear'


def _cross_entropy(input, target, weights=None, ignore_index=-100):
    idx = target != ignore_index
    t = target[idx]
    logsoftmax = nn.LogSoftmax(dim=1)
    logit = -logsoftmax(input[idx])[torch.arange(len(t)).long(), t]  # _x1
    if weights is not None:
        w = weights[t]
        logit *= w
        return torch.mean(logit), torch.sum(logit) / torch.sum(w)
    return torch.mean(logit), None


def _cross_entropy_soft(input, target, prior=None):
    """ Cross entropy that accepts soft targets
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    ce = torch.sum(-target * logsoftmax(input), dim=1)  # N*H*W
    if prior is not None:
        new_ce = ce * prior
        return torch.mean(new_ce), torch.sum(new_ce) / torch.sum(prior), torch.mean(ce)
    return None, None, torch.mean(ce)


def _psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def colorize_imgnet(net, total=10000, batch_size=1, eval_metrics=False, file_list='/srv/glusterfs/xieya/image/grayscale/test.txt', with_cap=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_trs1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224),
    ])
    val_trs2 = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    val_dataset = imagenetDataset(file_list, training=False, trs1=val_trs1, trs2=val_trs2, soft=True, return_gt=eval_metrics)
    val_generator = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    if not os.path.exists(_OUT_DIR): 
        os.makedirs(_OUT_DIR)
    if eval_metrics:
        losses = []
        losses_rb = []
        prior_means = []
        psnrs = []
    img_count = 0

    if with_cap:
        train_vocab = pickle.load(open(args.vocabulary_path, 'r'))

    fin = open(file_list, 'r')
    for data_dict, targets in val_generator:
        input_ims = Variable(data_dict['l_rgb'].cuda(async=True))
        if with_cap:
            p = next(fin)
            f = os.path.split(p)[1]
            caption = raw_input('Caption for {}?'.format(f))
            caption_words = caption.strip().split(' ')
            enc_caption_ = np.zeros((1, 20), dtype='uint32')
            input_length_ = np.zeros((1,), dtype='uint8')
            for j in range(len(caption_words)):
                enc_caption_[0, j] = train_vocab.get(caption_words[j], 0)
            input_length_[0] = len(caption_words)
            input_caption = Variable(torch.from_numpy(enc_caption_.astype('int32')).long().cuda())
            input_caption_len = torch.from_numpy(input_length_.astype('int32')).long().cuda()
            outputs, _ = net(input_ims, input_caption, input_caption_len)
        else:
            outputs = net(input_ims)
        if eval_metrics:
            target = Variable(targets.float().cuda(async=True))
            prior = Variable(data_dict['p'].float().cuda(async=True))
            loss_rb, _, loss = _cross_entropy_soft(outputs, target.view(-1, 313), prior=prior.view(-1))
            losses.append(loss.item())
            losses_rb.append(loss_rb.item())
            prior_means.append(np.mean(data_dict['p'].numpy()))
            gt_rgbs = data_dict['rgb'].numpy()
        outputs *= 2.63
        dec_inp = nn.Softmax(dim=1)(outputs)
        AB_vals = dec_inp.mm(cuda_cc)
        AB_vals = AB_vals.view(-1, 56, 56, 2).data.cpu().numpy()
        ls = data_dict['l'].numpy()
        for i in xrange(AB_vals.shape[0]):
            img_count += 1
            AB_val = AB_vals[i]
            AB_val = cv2.resize(AB_val, (ls[i].shape[1], ls[i].shape[0]), interpolation=cv2.INTER_CUBIC)
            print(AB_val.shape, ls[i].shape)
            img_dec = img_as_ubyte(color.lab2rgb(np.dstack((np.expand_dims(ls[i], axis=2), AB_val))))
            io.imsave(os.path.join(_OUT_DIR, '{}.jpg'.format(img_count)), img_dec)
            if eval_metrics:
                psnrs.append(_psnr(gt_rgbs[i], img_dec))
            print(img_count)
        if img_count >= total:
            break

    if eval_metrics:
        print('Cross entropy {}'.format(np.mean(losses)))
        print('Cross entropy rb {}'.format(np.mean(losses_rb)))
        print('Cross entropy rb nm {}'.format(np.mean(losses_rb) / np.mean(prior_means)))
        print('PSNR {}'.format(np.mean(psnrs)))

    fin.close()


def colorize_segcoco(net, as_rgb, total=10000, batch_size=4, eval_metrics=False, save=True):
    test_dataset = CocoStuff164k(root=args.data_root, split='val2017', as_rgb=as_rgb, random_crop=False, with_seg=False, return_gt=eval_metrics, with_cap=True)
    test_generator = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    img_count = 0
    if not os.path.exists(_OUT_DIR): 
        os.makedirs(_OUT_DIR)
    if eval_metrics:
        losses = []
        losses_rb = []
        prior_means = []
        psnrs = []
    with torch.no_grad():
        for data_dict in test_generator:
            input_ims = Variable(data_dict['luma'].float().cuda(async=True))
            outputs, _ = net(input_ims)
            if eval_metrics:
                targets = Variable(data_dict['target'].float().cuda(async=True))
                priors = Variable(data_dict['color_prior'].float().cuda(async=True))
                loss_rb, _, loss = _cross_entropy_soft(outputs, targets.view(-1, 313), prior=priors.view(-1))
                losses.append(loss.item())
                losses_rb.append(loss_rb.item())
                prior_means.append(np.mean(data_dict['color_prior'].numpy()))
                gt_rgbs = data_dict['bgr'].numpy()[:, :, :, :: -1]
            outputs *= 2.63
            dec_inp = nn.Softmax(dim=1)(outputs)
            AB_vals = dec_inp.mm(cuda_cc)
            AB_vals = AB_vals.view(-1, 56, 56, 2).data.cpu().numpy()
            ls = data_dict['l' if as_rgb else 'luma'].numpy()
            ls = (ls + 1.) * 50.
            ls = np.transpose(ls, (0, 2, 3, 1))
            for i in xrange(AB_vals.shape[0]):
                img_count += 1
                AB_val = AB_vals[i]
                AB_val = cv2.resize(AB_val, (224, 224), interpolation=cv2.INTER_CUBIC)
                img_dec = img_as_ubyte(color.lab2rgb(np.dstack((ls[i], AB_val))))
                if save:
                    io.imsave(os.path.join(_OUT_DIR, '{}.jpg'.format(img_count)), img_dec)
                if eval_metrics:
                    psnrs.append(_psnr(gt_rgbs[i], img_dec))
                    print(img_count, psnrs[-1])

            if img_count == total:
                break
    if eval_metrics:
        print('Cross entropy {}'.format(np.mean(losses)))
        print('Cross entropy rb {}'.format(np.mean(losses_rb)))
        print('Cross entropy rb nm {}'.format(np.mean(losses_rb) / np.mean(prior_means)))
        print('PSNR {}'.format(np.mean(psnrs)))


def _replace_color(caps, lens, color_set, color_list):
    for i in xrange(len(caps)):
        word_list = caps[i, : lens[i]].numpy()
        for j in xrange(lens[i]):
            if word_list[j] in color_set:
                # Replace with a random color
                r = random.choice(color_list)
                while r == word_list[j]:
                    r = random.choice(color_list)
                # print(vrev.get(word_list[j], 'unk'), vrev.get(r))
                word_list[j] = r
    return caps


def colorize_segcoco_with_cap(
    net, 
    as_rgb, 
    total=10000, 
    batch_size=4, 
    eval_metrics=False, 
    with_new_cap=False, 
    save=True,
    random_cap=False,
    with_seg=True,
    without_gray=False,
):
    print('Colorizing with captions')
    test_dataset = CocoStuff164k(
        root=args.data_root, 
        split='val2017', 
        as_rgb=as_rgb, 
        with_cap=True, 
        random_cap=False, 
        with_seg=with_seg, 
        random_crop=False, 
        return_gt=True, 
        with_vg=with_vg,
        old_coco=(not with_combined_dict),
        without_gray=without_gray
    )
    test_generator = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    if with_combined_dict:
        train_vocab = pickle.load(open(args.vocabulary_path, 'r'))
    elif with_vg:
        train_vocab = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/vocabulary.p', 'r'))
    else:
        train_vocab = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/vocabulary.p', 'r'))
    if random_cap:
        color_sub_list = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white']
        color_sub_list = [train_vocab[c] for c in color_sub_list]
        color_voc = pickle.load(open(args.color_vocabulary_path, 'rb'))
        color_set = set()
        for c in color_voc:
            if c in train_vocab:
                color_set.add(train_vocab[c])
    vrev = dict((v, k) for (k, v) in train_vocab.iteritems())  
    if with_seg:
        seg_label_prior = Variable(torch.from_numpy(
            utils.prior_boosting('resources/label_probs.npy', 1., .5)).float().cuda())
    img_count = 0
    if not os.path.exists(_OUT_DIR): 
        os.makedirs(_OUT_DIR)
    if eval_metrics:
        losses = []
        losses_rb = []
        if with_seg:
            seg_losses = []
        prior_means = []
        psnrs = []
    with torch.no_grad():
        for data_dict in test_generator:
            input_ims = Variable(data_dict['luma'].float().cuda(async=True))
            data_cap = data_dict['cap']
            if random_cap:
                data_cap = _replace_color(data_cap, data_dict['cap_len'], color_set, color_sub_list)
                input_captions = Variable(data_cap.long().cuda(async=True))
            else:
                input_captions = Variable(data_cap.long().cuda(async=True))
            input_caption_lens = Variable(data_dict['cap_len'].long().cuda(async=True))
            outputs, seg_outputs, attentions = net(input_ims, input_captions, input_caption_lens, True)
            if eval_metrics:
                targets = Variable(data_dict['target'].float().cuda(async=True))
                priors = Variable(data_dict['color_prior'].float().cuda(async=True))
                loss_rb, _, loss = _cross_entropy_soft(outputs, targets.view(-1, 313), prior=priors.view(-1))
                if with_seg:
                    labels = Variable(data_dict['label'].long().cuda(async=True))
                    _, seg_loss = _cross_entropy(seg_outputs, labels.view(-1), seg_label_prior, test_dataset.ignore_label)

                losses.append(loss.item())
                losses_rb.append(loss_rb.item())
                if with_seg:
                    seg_losses.append(seg_loss.item())
                prior_means.append(np.mean(data_dict['color_prior'].numpy()))
                gt_rgbs = data_dict['bgr'].numpy()[:, :, :, :: -1]
            outputs *= 2.63
            dec_inp = nn.Softmax(dim=1)(outputs)
            AB_vals = dec_inp.mm(cuda_cc)
            AB_vals = AB_vals.view(-1, 56, 56, 2).data.cpu().numpy()
            ls = data_dict['l' if as_rgb else 'luma'].numpy()
            ls = (ls + 1.) * 50.
            ls = np.transpose(ls, (0, 2, 3, 1))
            # labels = Variable(data_dict['label'].long().cuda(async=True))
            gt_rgbs = data_dict['bgr'].numpy()[:, :, :, :: -1]
            for i in xrange(AB_vals.shape[0]):
                img_count += 1
                if img_count <= skip_num:
                    continue
                AB_val = AB_vals[i]
                AB_val = cv2.resize(AB_val, (224, 224), interpolation=cv2.INTER_CUBIC)
                img_dec = img_as_ubyte(color.lab2rgb(np.dstack((ls[i], AB_val))))
                # img_ab = img_as_ubyte(color.lab2rgb(np.dstack((np.full(ls[i].shape, 50.), AB_val))))
                word_list = data_cap[i, : data_dict['cap_len'][i]].numpy()
                words = '_'.join(vrev.get(w, 'unk') for w in word_list)
                if save:
                    io.imsave(os.path.join(_OUT_DIR, '{}.jpg'.format(img_count, words)), img_dec)
                if eval_metrics:
                    psnrs.append(_psnr(gt_rgbs[i], img_dec))
                print(img_count)

                if with_new_cap:
                    print(words)
                    new_caption = raw_input('New caption?')
                    new_caption = new_caption.strip().split(' ')
                    new_enc_caption_ = torch.zeros(1, 20)
                    new_input_length_ = torch.zeros(1)
                    for j in range(len(new_caption)):
                        new_enc_caption_[0, j] = train_vocab.get(new_caption[j], 0)
                    new_input_length_[0] = len(new_caption)
                    new_input_caption = Variable(new_enc_caption_.long().cuda(async=True))
                    new_input_caption_len = Variable(new_input_length_.long().cuda(async=True))
                    new_outputs, _, new_attentions = net(input_ims[i: i + 1], new_input_caption, new_input_caption_len, True)
                    new_outputs *= 2.63
                    new_dec_inp = nn.Softmax(dim=1)(new_outputs)
                    new_AB_val = new_dec_inp.mm(cuda_cc)
                    new_AB_val = new_AB_val.view(56, 56, 2).data.cpu().numpy()
                    new_AB_val = cv2.resize(new_AB_val, (224, 224), interpolation=cv2.INTER_CUBIC)
                    new_img_dec = img_as_ubyte(color.lab2rgb(np.dstack((ls[i], new_AB_val))))
                    new_words = '_'.join(new_caption)
                    io.imsave(os.path.join(_OUT_DIR, '{}_n_{}.jpg'.format(img_count, new_words)), new_img_dec)

            if img_count == total:
                break
            
    if eval_metrics:
        print('Cross entropy {}'.format(np.mean(losses)))
        print('Cross entropy rb {}'.format(np.mean(losses_rb)))
        print('Cross entropy rb nm {}'.format(np.mean(losses_rb) / np.mean(prior_means)))
        if with_seg:
            print('Seg cross entropy {}'.format(np.mean(seg_losses)))
        print('PSNR {}'.format(np.mean(psnrs)))


def colorize_video(net):
    train_vocab = pickle.load(open(args.vocabulary_path, 'r'))
    caption = raw_input('Caption?')
    caption_words = caption.strip().split(' ')
    folder_name = '_'.join(caption_words)
    folder_path = os.path.join(_VIDEO_OUTPUT_FOLDER, folder_name)
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_trs1 = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224)])
    val_trs2 = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    val_dataset = imagenetDataset(_VIDEO_INPUT_FOLDER + '.txt', training=False, trs1=val_trs1, trs2=val_trs2, soft=True, return_gt=False)
    val_generator = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    enc_caption_ = np.zeros((1, 20), dtype='uint32')
    input_length_ = np.zeros((1,), dtype='uint8')
    for j in range(len(caption_words)):
        enc_caption_[0, j] = train_vocab.get(caption_words[j], 0)
    input_length_[0] = len(caption_words)
    input_caption = Variable(torch.from_numpy(enc_caption_.astype('int32')).long().cuda())
    input_caption_len = torch.from_numpy(input_length_.astype('int32')).long().cuda()

    img_cnt = 0
    for data_dict, targets in val_generator:
        input_im = Variable(data_dict['l_rgb'].cuda(async=True))    
        output, _ = net(input_im, input_caption, input_caption_len)
        output *= 2.63
        dec_inp = nn.Softmax(dim=1)(output)
        AB_vals = dec_inp.mm(cuda_cc)
        AB_vals = AB_vals.view(-1, 56, 56, 2).data.cpu().numpy()
        ls = data_dict['l'].numpy()
        for i in xrange(AB_vals.shape[0]):
            img_cnt += 1
            AB_val = AB_vals[i]
            AB_val = cv2.resize(AB_val, (224, 224), interpolation=cv2.INTER_CUBIC)
            img_dec = img_as_ubyte(color.lab2rgb(np.dstack((np.expand_dims(ls[i], axis=2), AB_val))))
            io.imsave(os.path.join(folder_path, '{}.jpg'.format(img_cnt)), img_dec)
            # if eval_metrics:
                # psnrs.append(_psnr(gt_rgbs[i], img_dec))
            print(img_cnt)


def net_and_decode(net, img_lab, input_im, input_caption, input_caption_len):
    output, _ = net(input_im, input_caption, input_caption_len)
    # softmax output and multiply by grid
    output *= 2.63
    dec_inp = nn.Softmax()(output)  # 12544x625
    AB_val = dec_inp.mm(cuda_cc)  # 12544x2
    # reshape and select last image of batch]
    AB_val = AB_val.view(1, 56, 56, 2).data.cpu().numpy()[0]

    AB_val = cv2.resize(AB_val, (224, 224), interpolation=cv2.INTER_CUBIC)
    img_dec = utils.labim2rgb(np.dstack((np.expand_dims(img_lab[0, :, :, 0], axis=2), AB_val)))
                
    return img_dec.astype('uint8')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='resnet coco colorization')
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='Inference batch size.')
    parser.add_argument('--color_vocabulary_path', default='/srv/glusterfs/xieya/data/color/vocabulary.p', help='Path to color vocabulary file.')
    parser.add_argument('--d_emb', default=300, type=int, help='Word-embedding dimension')
    parser.add_argument('--data_root', default='/srv/glusterfs/xieya/data/coco_seg/', help='Root directory of training data.')
    parser.add_argument('--deoldify', default=0, type=int, help='Colorize deoldify pictures.')
    parser.add_argument('--embedding_path', default='/srv/glusterfs/xieya/data/language/embedding.p', help='Path to word embedding file.')
    parser.add_argument('--eval', '-e', default=0, type=int, help='Evaluate metrics.')
    parser.add_argument('--gpuid', '-g', default='0', type=str, help='Which gpu to use.')
    parser.add_argument('--gru', default=1, type=int, help='Use GRU as language encoder.')
    parser.add_argument('--image_save_folder', '-d', default='', help='Folder where colorized results are stored')
    parser.add_argument('--language_ver', '-l', default=0, type=int, help='0: new coco, 1: new vg, 2: old coco, 3: old vg')
    parser.add_argument('--model', '-m', default=0, type=int, help='Model ID: 10 -> Our full method.')
    parser.add_argument('--save', '-s', default=1, type=int, help='Save or not.')
    parser.add_argument('--seg_ver', default=0, type=int, help='SegNet version.')
    parser.add_argument('--skip', default=0, type=int, help='Skip first N images.')
    parser.add_argument('--random_cap', default=0, type=int, help='Colorize with randomly generated caption.')
    parser.add_argument('--weights', '-w', default='', type=str, help='Pretrained weights.')
    parser.add_argument('--with_new_cap', '-c', default=0, type=int, help='Colorize with user input captions.')
    parser.add_argument('--without_gray', default=0, type=int, help='Without gray')
    parser.add_argument('--video', default=0, type=int, help='Colorize video.')
    parser.add_argument('--vocabulary_path', default='/srv/glusterfs/xieya/data/language/vocabulary.p', help='Path to vocabulary file.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    torch.backends.cudnn.benchmark = True

    if args.image_save_folder != '':
        _OUT_DIR = args.image_save_folder      
    print("Output directory:", _OUT_DIR)  

    # initialize quantized LAB encoder
    lookup_enc = utils.LookupEncode('resources/ab_grid.npy')
    num_classes = lookup_enc.cc.shape[0]
    cuda_cc = Variable(torch.from_numpy(lookup_enc.cc).float().cuda())

    caption_encoder_version = 'gru' if args.gru == 1 else 'lstm'
    with_seg = args.model in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    with_cap = args.model in [2, 3, 4, 5, 6, 7, 8, 9, 10]
    with_combined_dict = args.language_ver in [0, 1]
    with_vg = args.language_ver in [1, 3]
    if with_cap:
        if with_combined_dict:
            train_vocab_embeddings = pickle.load(open(args.embedding_path, 'rb'))
        elif with_vg:
            train_vocab_embeddings = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/embedding.p', 'rb'))
        else:
            train_vocab_embeddings = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/embedding.p', 'rb'))
    if args.model == 0:
        print('VGG16 + attention')
        net = AutocolorizeVGG16Attention()
    elif args.model == 1:
        print('Coco seg')
        net = AutocolorizeVGGWithSeg()
    elif args.model == 2:
        print('Unet with cap')
        net = AutocolorizeUnetCap(
            train_vocab_embeddings=train_vocab_embeddings,
            caption_encoder_version=caption_encoder_version,
            with_spatial_attention=[0, 0, -1, 0],
            k=2,
        )
    elif args.model == 3:
        print('Unet seg + 1 FiLM')
        net = AutocolorizeUnetSegCap(train_vocab_embeddings=train_vocab_embeddings, caption_encoder_version=caption_encoder_version)
    elif args.model == 4:
        print('Unet seg + 4 FiLM')
        net = AutocolorizeUnetSegCapV2(train_vocab_embeddings=train_vocab_embeddings, seg_ver=args.seg_ver, caption_encoder_version=caption_encoder_version)
    elif args.model == 5:
        print('Unet seg + 1 Concat')
        net = AutocolorizeUnetSegCapV3(train_vocab_embeddings=train_vocab_embeddings, caption_encoder_version=caption_encoder_version)
    elif args.model == 6:
        print('Unet seg + 1 FiLM with spatial attention')
        net = AutocolorizeUnetSegCapV5(train_vocab_embeddings=train_vocab_embeddings, caption_encoder_version=caption_encoder_version)
    elif args.model == 7:
        print('Unet seg + 1 Concat with dense layer')
        net = AutocolorizeUnetSegCapV4(train_vocab_embeddings=train_vocab_embeddings, caption_encoder_version=caption_encoder_version)
    elif args.model == 8:
        print('Unet seg v2 with 4 FiLM blocks at the front of each block')
        net = AutocolorizeUnetSegCapV6(train_vocab_embeddings=train_vocab_embeddings, caption_encoder_version=caption_encoder_version)
    elif args.model == 9:
        print('Unet seg with 4 FiLM at front')
        net = AutocolorizeUnetSegCapV7(train_vocab_embeddings=train_vocab_embeddings, caption_encoder_version=caption_encoder_version)
    elif args.model == 10:
        print('Unet seg v{} with 4 FiLM blocks with spatial attention'.format(args.seg_ver))
        net = AutocolorizeUnetSegCapV2(
            train_vocab_embeddings=train_vocab_embeddings, 
            seg_ver=args.seg_ver, 
            with_spatial_attention=[0, 0, -1, 0],
            caption_encoder_version=caption_encoder_version,
            k=2,
        )
    elif args.model == 11:
        print('Unet seg')
        net = AutocolorizeUnetWithSeg()

    if args.weights != '':
        # Load pretrained weights.
        if os.path.isfile(args.weights):
            print("=> loading pretrained weights '{}'".format(args.weights))
            weights = torch.load(args.weights)
            net.load_state_dict(weights['state_dict'])
            print('Loaded.')
        else:
            print("=> no weights found at '{}'".format(args.weights))
            exit()
    net.cuda()
    net.eval()

    if args.video == 1:
        colorize_video(net)
        exit()

    if args.deoldify == 1:
        colorize_imgnet(net, batch_size=1, file_list='/srv/glusterfs/xieya/data/deoldify/test.txt', with_cap=True)
        exit()

    eval_metrics = args.eval == 1
    with_new_cap = args.with_new_cap == 1
    save = args.save == 1
    skip_num = args.skip
    random_cap = args.random_cap == 1
    without_gray = args.without_gray == 1
    if args.model == 0:
        colorize_segcoco(net, as_rgb=True, batch_size=args.batch_size, eval_metrics=eval_metrics)
    elif args.model in [1]:
        colorize_segcoco(net, as_rgb=False, total=10000, batch_size=args.batch_size, eval_metrics=eval_metrics)
    elif args.model in [11]:
        colorize_segcoco(net, as_rgb=True, batch_size=args.batch_size, eval_metrics=eval_metrics)
    elif args.model in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        colorize_segcoco_with_cap(
            net, 
            as_rgb=True, 
            batch_size=args.batch_size, 
            eval_metrics=eval_metrics, 
            with_new_cap=with_new_cap, 
            save=save,
            random_cap=random_cap,
            with_seg=with_seg,
            without_gray=without_gray,
        )
