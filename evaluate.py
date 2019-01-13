import math
import numpy as np
import os
import pickle
from skimage import color, io, transform
from sklearn.metrics import auc
import utils

_AUC_THRESHOLD = 150
_GT_DIR = '/srv/glusterfs/xieya/image/color/gt'
_PRIOR_PATH = '/srv/glusterfs/xieya/prior/coco2017_313_soft.npy'
_LOOKUP = utils.LookupEncode('/home/xieya/colorization-tf/resources/pts_in_hull.npy')
_LOG_FREQ = 100


def _l2_acc(gt_ab, pred_ab, prior_factor_0, prior_factor_5):
    '''
    L2 accuracy given different threshold.
    '''
    ab_idx = _LOOKUP.encode_points(gt_ab)
    prior_0 = prior_factor_0[ab_idx]
    prior_5 = prior_factor_5[ab_idx]

    l2_dist = np.sqrt(np.sum(np.square(gt_ab - pred_ab), axis=2))
    ones = np.ones_like(l2_dist)
    zeros = np.zeros_like(l2_dist)
    scores = []
    scores_rb_0 = []
    scores_rb_5 = []
    total = np.sum(ones)
    prior_0_sum = np.sum(prior_0)
    prior_5_sum = np.sum(prior_5)
    for thr in range(0, _AUC_THRESHOLD + 1):
        score = np.sum(
            np.where(np.less_equal(l2_dist, thr), ones, zeros)) / total
        score_rb_0 = np.sum(
            np.where(np.less_equal(l2_dist, thr), prior_0, zeros)) / prior_0_sum
        score_rb_5 = np.sum(
            np.where(np.less_equal(l2_dist, thr), prior_5, zeros)) / prior_5_sum
        scores.append(score)
        scores_rb_0.append(score_rb_0)
        scores_rb_5.append(score_rb_5)
    return scores, scores_rb_0, prior_0_sum, scores_rb_5, prior_5_sum


def _mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def _psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def evaluate_from_rgb(in_dir, without_gray=False):
    '''
        AUC / image
        AUC / pixel
        RMSE of AB / pixel
    '''
    prior_factor_0 = utils.prior_boosting(_PRIOR_PATH, 1., 0.)
    prior_factor_5 = utils.prior_boosting(_PRIOR_PATH, 1., .5)

    l2_accs = []
    l2_rb_0_accs = []
    l2_rb_5_accs = []
    prior_0_weights = []
    prior_5_weights = []
    auc_scores = []
    auc_rb_0_scores = []
    auc_rb_5_scores = []
    mse_ab_scores = []
    psnr_rgb_scores = []
    x = [i for i in range(0, _AUC_THRESHOLD + 1)]
    img_count = 0

    if without_gray:
        gray_list = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/val_filtered_gray.p', 'rb'))

    for img_id in xrange(1, 2487):
        if without_gray and img_id in gray_list:
            continue
        img_name = "{}.jpg".format(img_id)
        img_path = os.path.join(in_dir, img_name)
        img_rgb = io.imread(img_path)
        gt_path = os.path.join(_GT_DIR, img_name)
        gt_rgb = io.imread(gt_path)
        img_ab = color.rgb2lab(img_rgb)[:, :, 1:]
        gt_ab = color.rgb2lab(gt_rgb)[:, :, 1:]
        ab_ss = transform.downscale_local_mean(img_ab, (4, 4, 1))
        gt_ab_ss = transform.downscale_local_mean(gt_ab, (4, 4, 1))
        l2_acc, l2_rb_0_acc, prior_0_weight, l2_rb_5_acc, prior_5_weight = _l2_acc(gt_ab_ss, ab_ss, prior_factor_0, prior_factor_5)
        l2_accs.append(l2_acc)
        l2_rb_0_accs.append(l2_rb_0_acc)
        l2_rb_5_accs.append(l2_rb_5_acc)
        prior_0_weights.append(prior_0_weight)
        prior_5_weights.append(prior_5_weight)
        auc_score = auc(x, l2_acc) / _AUC_THRESHOLD
        auc_rb_0_score = auc(x, l2_rb_0_acc) / _AUC_THRESHOLD
        auc_rb_5_score = auc(x, l2_rb_5_acc) / _AUC_THRESHOLD
        auc_scores.append(auc_score)
        auc_rb_0_scores.append(auc_rb_0_score)
        auc_rb_5_scores.append(auc_rb_5_score)
        mse_ab_score = _mse(gt_ab_ss, ab_ss)
        mse_ab_scores.append(mse_ab_score)
        psnr_rgb_score = _psnr(gt_rgb, img_rgb)
        psnr_rgb_scores.append(psnr_rgb_score)

        # summary = '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(img_id, auc_score, auc_rb_0_score, auc_rb_5_score, np.sqrt(mse_ab_score))
        # print(summary)
        img_count += 1
        if img_count % _LOG_FREQ == 0:
            print(img_count)
            # fout.flush()

    l2_accs = np.asarray(l2_accs)
    prior_0_weights = np.asarray(prior_0_weights)
    prior_5_weights = np.asarray(prior_5_weights)
    l2_rb_0_accs = np.asarray(l2_rb_0_accs)
    l2_rb_5_accs = np.asarray(l2_rb_5_accs)
    
    # AUC / pix
    l2_acc_per_pix = np.mean(l2_accs, axis=0)
    l2_rb_0_acc_per_pix = np.average(l2_rb_0_accs, weights=prior_0_weights, axis=0)
    l2_rb_5_acc_per_pix = np.average(l2_rb_5_accs, weights=prior_5_weights, axis=0)
    auc_per_pix = auc(x, l2_acc_per_pix) / _AUC_THRESHOLD
    auc_rb_0_per_pix = auc(x, l2_rb_0_acc_per_pix) / _AUC_THRESHOLD
    auc_rb_5_per_pix = auc(x, l2_rb_5_acc_per_pix) / _AUC_THRESHOLD
    print("AUC per pix\t{0}".format(auc_per_pix))
    print("AUC rebalanced gamma 0. per pix\t{0}".format(auc_rb_0_per_pix))
    print("AUC rebalanced gamma 0.5 per pix\t{0}".format(auc_rb_5_per_pix))

    # AUC / img
    print("AUC per image\t{0}".format(np.mean(auc_scores)))
    print("AUC rebalanced gamma 0. per image\t{0}".format(np.mean(auc_rb_0_scores)))
    print("AUC rebalanced gamma 0.5 per image\t{0}".format(np.mean(auc_rb_5_scores)))

    # PSNR RGB / img
    print("PSNR RGB per image\t{0}".format(np.mean(psnr_rgb_scores)))

    # RMSE AB / pix
    print("RMSE AB per pix\t{0}".format(np.sqrt(np.mean(mse_ab_scores))))
    print(in_dir)


if __name__ == '__main__':
    evaluate_from_rgb('/srv/glusterfs/xieya/image/color/e4', without_gray=True)
