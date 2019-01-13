import cv2
from glob import glob
import json
import numpy as np
import os
import pickle
import random
from skimage import color, img_as_ubyte, io, transform
import torchvision.transforms as transforms
import torch.utils.data as data
import utils

'''
https://github.com/kazuto1011/deeplab-pytorch
'''


class _CocoStuff(data.Dataset):
    """COCO-Stuff base class"""

    def __init__(
        self,
        root,
        split='train',
        crop_size=224,
        flip=True,
        as_rgb=False,
        with_cap=False,
        random_cap=True,
        with_seg=True,
        random_crop=True,
        return_gt=False,
        with_empty_cap=False,
        with_vg=False,
        old_coco=False,
        with_cap_filter=False,
        without_gray=False,
    ):
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.flip = flip
        self.as_rgb = as_rgb
        self.train = 'train' in self.split
        self.with_cap = with_cap
        self.with_seg = with_seg
        self.random_crop = random_crop
        self.return_gt = return_gt
        self.with_empty_cap = with_empty_cap
        self.with_vg = with_vg
        self.with_cap_filter = with_cap_filter
        self.without_gray = without_gray

        if self.without_gray:
            self.gray_list = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/val_filtered_gray.p', 'rb'))
        self.files = []
        self.ignore_label = None
        if self.as_rgb:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.trns = transforms.Compose([transforms.ToTensor(), normalize])
        if self.with_cap or self.with_cap_filter:
            if self.with_vg:
                self.im2cap = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/im2regs.p', 'rb'))
                self.caps = json.load(open('/srv/glusterfs/xieya/data/visual_genome/224_sz20_colored_region_descriptions.json', 'rb'))
            else:
                if old_coco:
                    self.im2cap = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/im2cap.p', 'rb'))
                    self.caps = np.load('/srv/glusterfs/xieya/data/coco_seg/annotations/captions.npy')
                    self.lens = np.load('/srv/glusterfs/xieya/data/coco_seg/annotations/caption_lengths.npy')
                else:
                    self.im2cap = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/im2cap_comb.p', 'rb'))
                    self.caps = np.load('/srv/glusterfs/xieya/data/coco_seg/annotations/captions_comb.npy')
                    self.lens = np.load('/srv/glusterfs/xieya/data/coco_seg/annotations/caption_lengths_comb.npy')
            self.random_cap = random_cap

        grid_file = '/home/xieya/colorization-tf/resources/pts_in_hull.npy'
        self.lookup_enc = utils.LookupEncode(grid_file)
        self.nn_enc = utils.NNEncode(10, 5., km_filepath=grid_file) 
        self.color_prior = utils.prior_boosting('/srv/glusterfs/xieya/prior/coco2017_313_soft.npy', 1., .5)

        self._set_files()
        
        print('Dataset {}'.format(self.split))
        if self.train:
            print('Random lighting')
            if self.flip:
                print('Random flipping')
            else:
                print('No flipping')
        else:
            print('No lighting augmentation')
            print('No flipping')
        if self.random_crop:
            print('Random cropping')
        else:
            print('Center cropping')
        if self.with_empty_cap:
            print('Augment with empty captions')
        if self.with_cap_filter:
            print('With caption filtering.')

    def _set_files(self):
        raise NotImplementedError()

    def _transform(self, in_dict):
        data_dict = {}
        image = in_dict['image']
        # Random cropping
        base_h, base_w, _ = image.shape
        thr_x = base_w - self.crop_size
        thr_y = base_h - self.crop_size
        if 'bbox' in in_dict:
            regx, regy, regw, regh = in_dict['bbox']
            regw = min(regw, base_w - regx)
            regh = min(regh, base_h - regy)
            thr_x = min(thr_x, regx)
            thr_y = min(thr_y, regy)
        if self.random_crop:
            start_h = random.randint(0, thr_y)
            start_w = random.randint(0, thr_x)
        else:
            start_h = (thr_y + 1) / 2
            start_w = (thr_x + 1) / 2
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h: end_h, start_w: end_w]
        if 'label' in in_dict:
            label = in_dict['label']
            label = label[start_h:end_h, start_w:end_w]
        if 'bbox' in in_dict:
            regx -= start_w
            regy -= start_h

        if self.train and self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image)  # HWC
                if 'label' in in_dict:
                    label = np.fliplr(label)  # HW
                if 'bbox' in in_dict:
                    regx = self.crop_size - regx - regw

        if self.return_gt:
            data_dict['bgr'] = img_as_ubyte(image)

        # Downsampling
        if 'label' in in_dict:
            label_ss = cv2.resize(label, (self.crop_size / 4, self.crop_size / 4), interpolation=cv2.INTER_NEAREST).astype('int64')
            data_dict['label'] = label_ss
        rgb = np.clip(image[:, :, :: -1], 0., 1.)
        lab = color.rgb2lab(rgb)
        # Random lighting
        if self.train:
            lab[:, :, 0] = _random_lighting(lab[:, :, 0])

        if self.as_rgb:
            if not self.train:
                img_l = lab[:, :, 0: 1] / 50. - 1.
                img_l = img_l.transpose(2, 0, 1)
                data_dict['l'] = img_l
            gray = lab.copy()
            gray[:, :, 1:] = 0.
            luma = img_as_ubyte(color.lab2rgb(gray))
            luma = self.trns(luma)  # Normalize and to_tensor.
        else:
            luma = lab[:, :, 0: 1] / 50. - 1.
            # HWC -> CHW
            luma = luma.transpose(2, 0, 1)
        data_dict['luma'] = luma
            
        ab_ss = transform.downscale_local_mean(lab[:, :, 1:], (4, 4, 1))
        target = self.nn_enc.encode_points_mtx_nd(ab_ss, axis=2)
        data_dict['target'] = target
        onehot = self.lookup_enc.encode_points(ab_ss)
        color_prior = self.color_prior[onehot]
        data_dict['color_prior'] = color_prior

        if self.with_cap:
            data_dict['cap'] = in_dict['cap']
            data_dict['cap_len'] = in_dict['cap_len']
            if 'bbox' in in_dict:
                bbox = np.full((self.crop_size, self.crop_size, 1), 1)
                bbox[regy: regy + regh, regx: regx + regw] = 3
                data_dict['bbox'] = bbox[::4, ::4]

        return data_dict

    def _load_data(self, image_id):
        raise NotImplementedError()

    def __getitem__(self, index):
        image_id = self.files[index]
        data_dict = self._load_data(image_id)
        data = self._transform(data_dict)
        return data

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root Location: {}\n".format(self.root)
        return fmt_str


class CocoStuff164k(_CocoStuff):
    """COCO-Stuff 164k dataset"""

    def __init__(self, **kwargs):
        super(CocoStuff164k, self).__init__(**kwargs)
        self.ignore_label = 255

    def _set_files(self):
        # Create data list by parsing the "images" folder
        if self.split in ["train2017", "val2017", "test2017"]:
            file_list = sorted(glob(os.path.join(self.root, "images_224", self.split, "*.jpg")))
            file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
            if self.with_cap or self.with_cap_filter:
                file_list = list(filter(lambda x: x in self.im2cap, file_list))
            if self.without_gray:
                new_file_list = [file_list[i] for i in xrange(len(file_list)) if i + 1 not in self.gray_list]
                file_list = new_file_list
            self.files = file_list
            print('Size {}'.format(len(self.files)))
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, image_id):
        out_dict = {}
        # Set paths
        image_path = os.path.join(self.root, "images_224", self.split, image_id + ".jpg")
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if self.split in ["train2017", "val2017"] and self.with_seg:
            label_path = os.path.join(self.root, "annotations_224", self.split, image_id + ".png")
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
            out_dict['label'] = label

        if self.with_cap:
            cap_list = self.im2cap[image_id]
            if self.random_cap:
                # Randomly pick one cap for the image.`
                if self.with_empty_cap:
                    cap_list *= 3
                    cap_list += [-1]
                r = random.choice(cap_list)
            else:
                r = cap_list[0]

            if self.with_vg:
                if r == -1:
                    cap = np.zeros((20,), dtype='int32')
                    cap_len = 1
                    bbox = (0, 0, 0, 0)
                else:
                    reg = self.caps[r]
                    cap = np.asarray(reg['phrase']).astype('int32')
                    cap_len = reg['phrase_len']
                    bbox = (reg['x'], reg['y'], reg['width'], reg['height'])
                if self.train:
                    out_dict['bbox'] = bbox
            else:
                if r == -1:
                    cap = np.zeros((20,), dtype='int32')
                    cap_len = 1
                else:
                    cap = self.caps[r]
                    cap_len = self.lens[r]
            out_dict['cap'] = cap
            out_dict['cap_len'] = cap_len

        image = image.astype(np.float32) / 255.
        out_dict['image'] = image

        return out_dict


class imagenetDataset(data.Dataset):
    def __init__(
        self, 
        file_path, 
        grid_file='/home/xieya/colorization-tf/resources/pts_in_hull.npy', 
        training=True, 
        trs1=None, 
        trs2=None, 
        soft=False,
        return_gt=False,
    ):
        self.file_path = file_path
        self.file_list = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                self.file_list.append(line.strip())
        self.prior = utils.prior_boosting('/home/xieya/colorization-tf/resources/prior_probs_smoothed.npy', 1., .5)
        self.soft = soft
        self.training = training
        self.trs1 = trs1
        self.trs2 = trs2
        self.return_gt = return_gt
        if self.soft:
            self.nn_enc = utils.NNEncode(10, 5., km_filepath='/home/xieya/colorization-tf/resources/pts_in_hull.npy')    
        self.lookup_enc = utils.LookupEncode(grid_file)
        if self.training:
            print('Training dataset.')
        else:
            print('Validation dataset.')
        if self.soft:
            print('Soft target.')
        else:
            print('One-hot target.')

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img_bgr = cv2.imread(img_path)
        while img_bgr is None or not (len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3):
            print(img_path)
            img_path = self.file_list[random.randint(0, len(self.file_list) - 1)]
            img_bgr = cv2.imread(img_path)
        img_rgb = img_bgr[:, :, ::-1]
        img_l_o = color.rgb2lab(img_rgb)[:, :, 0]

        if self.trs1 is not None:
            img_rgb = self.trs1(img_rgb)  # Cropping and flipping.

        img_lab = color.rgb2lab(img_rgb)
        img_ab = transform.downscale_local_mean(img_lab[:, :, 1:], (4, 4, 1))
        if self.soft:
            target = self.nn_enc.encode_points_mtx_nd(img_ab, axis=2)
            onehot = self.lookup_enc.encode_points(img_ab)
            priors = self.prior[onehot]
        else:
            target = self.lookup_enc.encode_points(img_ab)

        if self.training:
            img_lab[:, :, 0] = _random_lighting(img_lab[:, :, 0])  # [0, 100]
        img_lab[:, :, 1:] = 0.
        img_rgb_gray = img_as_ubyte(color.lab2rgb(img_lab))

        if self.trs2 is not None:
            img_rgb_gray = self.trs2(img_rgb_gray)  # Normalize and to_tensor.

        data_dict = {'l_rgb': img_rgb_gray}
        if self.soft:
            data_dict['p'] = priors
        if not self.training:
            data_dict['l'] = img_l_o
        if self.return_gt:
            data_dict['rgb'] = np.asarray(img_rgb)

        return data_dict, target

    def __len__(self):
        return len(self.file_list)


def _rand0(x):
    # [-x, x)
    return (random.random() * 2 - 1) * x


def _random_lighting(img_l, val_range=(0., 100.)):
    # input: [0, 100]
    # Random lighting by FastAI
    b = _rand0(0.1 * (val_range[1] - val_range[0]))
    c = _rand0(0.1)
    c = -1 / (c - 1) if c < 0 else c + 1
    mu = np.mean(img_l)

    return np.clip((img_l - mu) * c + mu + b, val_range[0], val_range[1])
