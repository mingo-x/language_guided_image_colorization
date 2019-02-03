import cv2
import math
import numpy as np
import sklearn.neighbors as sknn
import torch.nn as nn
from torch.nn.init import kaiming_normal_, kaiming_uniform_, orthogonal_, xavier_uniform_


def cvrgb2lab(img_rgb):
    cv_im_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype('float32')
    cv_im_lab[:, :, 0] *= (100. / 255)
    cv_im_lab[:, :, 1:] -= 128.
    return cv_im_lab


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal_
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform_
    else:
        return
    print('Weights initialization:')
    for md in modules:
        if md is None:
            continue
        for m in md.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                print(m)
                init_params(m.weight)
            elif isinstance(m, (nn.GRU, nn.LSTM)):
                print(m)
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        orthogonal_(param.data)


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def prior_boosting(prior_file, alpha, gamma):
    prior_probs = np.load(prior_file)

    # define uniform probability
    uni_probs = np.zeros_like(prior_probs)
    uni_probs[prior_probs != 0] = 1.
    uni_probs = uni_probs / np.sum(uni_probs)

    # convex combination of empirical prior and uniform distribution       
    prior_mix = (1 - gamma) * prior_probs + gamma * uni_probs

    # set prior factor
    prior_factor = prior_mix ** -alpha
    prior_factor[prior_factor == np.inf] = 0.  # mask out unused classes
    prior_factor = prior_factor / np.sum(prior_probs * prior_factor)  # re-normalize

    return prior_factor


def flatten_nd_array(pts_nd,axis=1):
    ''' 
    https://github.com/superhans/colorfromlanguage
    '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt


def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    '''
    https://github.com/superhans/colorfromlanguage
    '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)

        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out


def labim2rgb(lab):
    return (255. * cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)).astype('uint8')


class LookupEncode():
    '''Encode points using lookups'''
    def __init__(self, km_filepath=''):

        self.cc = np.load(km_filepath)
        self.grid_width = 10
        self.offset = np.abs(np.amin(self.cc)) + 17 # add to get rid of negative numbers
        self.x_mult = 300 # differentiate x from y
        self.labels = {}
        for idx, (x,y) in enumerate(self.cc):
            x += self.offset
            x *= self.x_mult
            y += self.offset
            if x+y in self.labels:
                print('Id collision!!!')
            self.labels[x+y] = idx


    # returns bsz x 224 x 224 of bin labels (625 possible labels)
    def encode_points(self, pts_nd):
        # One-hot encoding
        pts_flt = pts_nd.reshape((-1, 2))

        # round AB coordinates to nearest grid tick
        pgrid = np.round(pts_flt / self.grid_width) * self.grid_width

        # get single number by applying offsets
        pvals = pgrid + self.offset
        pvals = pvals[:, 0] * self.x_mult + pvals[:, 1]

        labels = np.zeros(pvals.shape,dtype='int32')
        labels.fill(-1)

        # lookup in label index and assign values
        for k in self.labels:
            labels[pvals == k] = self.labels[k]
        if len(labels[labels == -1]) > 0:
            print("Point outside of grid!!!")
            labels[labels == -1] = 0

        return labels.reshape(pts_nd.shape[:-1])


    # return lab grid marks from probability distribution over bins
    def decode_points(self, pts_enc):
        print pts_enc
        return pts_enc.dot(self.cc)


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self, NN, sigma, km_filepath=''):
        self.cc = np.load(km_filepath)

        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = sknn.NearestNeighbors(n_neighbors=self.NN, algorithm='auto').fit(self.cc)
        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        P = pts_flt.shape[0]

        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P, self.K))
            self.p_inds = np.arange(0, P, dtype='int')[:, np.newaxis]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2 / (2 * self.sigma**2))
        wts = wts / np.sum(wts, axis=1)[:, np.newaxis]

        self.pts_enc_flt[self.p_inds, inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)

        return np.copy(pts_enc_nd)


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)
