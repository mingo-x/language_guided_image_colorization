#!/srv/glusterfs/xieya/anaconda2/bin/python

# ----- Parameters passed to the cluster -------
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue

#$ -t 1:10

#$ -S /srv/glusterfs/xieya/anaconda2/bin/python

#$ -l h_rt=5:59:59

#$ -l h_vmem=8G

#$ -o /srv/glusterfs/xieya/log

#$ -e /srv/glusterfs/xieya/log

#$ -j y

import cv2
import json
import numpy as np
import os
import pickle
import re

_LANGUAGE_DIR = '/srv/glusterfs/xieya/data/language'
_LOG_FREQ = 100
_ROOT = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed'
_SIZE = 224
_SPLIT = 'val'

_TASK_ID = os.environ.get('SGE_TASK_ID')
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1
else:
    _TASK_ID = 0
_TASK_NUM = 10


def _to_224(img_name):
    if '/' in img_name:
        img_path = img_name
        img_class, img_name = os.path.split(img_path)
        if _SPLIT == 'train':
            img_class = os.path.split(img_class)[1]
            out_dir = os.path.join(_ROOT, _SPLIT + '_224', img_class)
        else:
            out_dir = os.path.join(_ROOT, _SPLIT + '_224')
    else:
        # Set paths
        img_path = os.path.join(_ROOT, "images", _SPLIT, img_name)
        out_dir = os.path.join(_ROOT, "images_224", _SPLIT)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_id = os.path.splitext(img_name)[0]

    # Load an image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(img_path)
        return
    img = img.astype(np.float32)
    # Resize
    raw_h, raw_w, _ = img.shape
    if raw_h < raw_w:
        base_size = (int(_SIZE * raw_w / raw_h), _SIZE)
    else:
        base_size = (_SIZE, int(_SIZE * raw_h / raw_w))
    if min(raw_w, raw_h) < _SIZE:  # Zooming
        img = cv2.resize(img, base_size, interpolation=cv2.INTER_CUBIC)
    else:  # Shrinking
        img = cv2.resize(img, base_size, interpolation=cv2.INTER_AREA)
    # Save
    cv2.imwrite(os.path.join(out_dir, img_name), img.astype('uint8'))
    if _SPLIT in ['train2017', 'val2017']:
        label_path = os.path.join(_ROOT, "annotations", _SPLIT, img_id + ".png")
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        label = cv2.resize(label, base_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(_ROOT, "annotations_224", _SPLIT, img_id + ".png"), label.astype('uint8'))


def _tokenize(c):
    c = c.encode('utf-8').strip().lower()
    # remove all non-alphanum chars (keep spaces between word):
    c = re.sub("[^a-z0-9 ]+", "", c)
    # remove all double spaces in the caption:
    c = re.sub("  ", " ", c)
    # convert the caption into a vector of words:
    c = c.split(" ")
    # remove any empty chars still left in the caption:
    while "" in c:
        index = c.index("")
        del c[index]
    return c


def _word_count(voc_dict, file):
    data = json.load(open(file, 'r'))
    print('Caption json loaded.')
    captions = data['annotations']
    c_count = 0
    for cap in captions:
        c = _tokenize(cap['caption'])
        for w in c:
            if w not in voc_dict:
                voc_dict[w] = 1
            else:
                voc_dict[w] += 1
        c_count += 1
        if c_count % _LOG_FREQ == 0:
            print(c_count, c)


def word_count():
    voc_dict = {}
    _word_count(voc_dict, '/srv/glusterfs/xieya/data/coco_seg/annotations/captions_train2014.json')
    print('Voc size: {}'.format(len(voc_dict)))
    freq_count = 0
    for w in voc_dict:
        if voc_dict[w] >= 5:
            freq_count += 1
    print('Voc >= 5 count: {}'.format(freq_count))
    _word_count(voc_dict, '/srv/glusterfs/xieya/data/coco_seg/annotations/captions_val2014.json')
    print('Voc size: {}'.format(len(voc_dict)))
    freq_count = 0
    for w in voc_dict:
        if voc_dict[w] >= 5:
            freq_count += 1
    print('Voc >= 5 count: {}'.format(freq_count))
    pickle.dump(voc_dict, open('/srv/glusterfs/xieya/data/coco_seg/word_count.p', 'wb'))


def word_to_emb(emb_file, dataset='coco_seg'):
    emb_dict = pickle.load(open(os.path.join(_LANGUAGE_DIR, emb_file), 'rb'))
    wc = pickle.load(open('/srv/glusterfs/xieya/data/{}/word_count.p'.format(dataset), 'rb'))
    print('Loaded.')
    voc_dict = {'unk': 0}
    embeddings = [emb_dict['unk']]
    unk_cnt = 0
    for w in wc:
        if wc[w] < 5:
            continue
        if w in emb_dict:
            idx = len(voc_dict)
            embeddings.append(emb_dict[w])
            voc_dict[w] = idx
            print(w, idx)
        else:
            unk_cnt += 1

    embeddings = np.asarray(embeddings)
    print('Embedded size: {}'.format(len(voc_dict)))
    print('Unknown size: {}'.format(unk_cnt))
    pickle.dump(voc_dict, open('/srv/glusterfs/xieya/data/{}/vocabulary.p'.format(dataset), 'wb'))
    pickle.dump(embeddings, open('/srv/glusterfs/xieya/data/{}/embedding.p'.format(dataset), 'wb'))


def combine_voc(emb_file):
    vg_vocab = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/vocabulary.p', 'r'))
    coco_vocab = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/vocabulary.p', 'r'))
    emb_dict = pickle.load(open(os.path.join(_LANGUAGE_DIR, emb_file), 'rb'))
    embeddings = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/embedding.p', 'rb'))
    for w in coco_vocab:
        if w not in vg_vocab:
            if w in emb_dict:
                emb_idx = len(vg_vocab)
                embeddings = np.append(embeddings, [emb_dict[w]], axis=0)
                vg_vocab[w] = emb_idx
            else:
                print w

    print('Voc size {}'.format(len(vg_vocab)))
    print('Emb size {}'.format(len(embeddings)))
    pickle.dump(vg_vocab, open(os.path.join(_LANGUAGE_DIR, 'vocabulary.p'), 'wb'))
    pickle.dump(embeddings, open(os.path.join(_LANGUAGE_DIR, 'embedding.p'), 'wb'))


def word_count_vg():
    voc_dict = {}
    regions = json.load(open('/srv/glusterfs/xieya/data/visual_genome/region_descriptions.json', 'r'))
    print('Region json loaded.')
    reg_cnt = 0
    for img in regions:
        for reg in img['regions']:
            c = _tokenize(reg['phrase'])
            for w in c:
                if w not in voc_dict:
                    voc_dict[w] = 1
                else:
                    voc_dict[w] += 1
            reg_cnt += 1
            if reg_cnt % _LOG_FREQ == 0:
                print(reg_cnt, c)
    print('Voc size: {}'.format(len(voc_dict)))
    freq_count = 0
    for w in voc_dict:
        if voc_dict[w] >= 5:
            freq_count += 1
    print('Voc >= 5 count: {}'.format(freq_count))
    pickle.dump(voc_dict, open('/srv/glusterfs/xieya/data/visual_genome/word_count.p', 'wb'))


def _im_id_2_file(images):
    id_2_file = {}
    for img in images:
        img_id = img['id']
        img_file = img['file_name'].split('_')[-1]
        id_2_file[img_id] = img_file
    pickle.dump(id_2_file, open('/srv/glusterfs/xieya/data/coco_seg/imid2file.p', 'wb'))
    return id_2_file


def _struct_data(file, voc_dict, color_set, im_file_2_cap_id, caps, lens):
    data = json.load(open(file, 'r'))
    print('Caption json loaded.')
    captions = data['annotations']
    images = data['images']
    im_id_2_file = _im_id_2_file(images)
    nocolor_cnt = 0     
    for cap in captions:
        c = _tokenize(cap['caption'])
        has_color = False
        tokens = np.zeros(20, dtype=np.int32)
        for i in xrange(min(len(c), 20)):
            tokens[i] = voc_dict.get(c[i], 0)
            if tokens[i] in color_set:
                has_color = True

        if has_color:
            img_id = cap['image_id']
            img_file = im_id_2_file[img_id]
            cap_idx = len(caps)
            caps.append(tokens)
            lens.append(min(len(c), 20))
            img_file_id = os.path.splitext(img_file)[0]
            if img_file_id not in im_file_2_cap_id:
                im_file_2_cap_id[img_file_id] = []
            im_file_2_cap_id[img_file_id].append(cap_idx)
        else:
            nocolor_cnt += 1
    return nocolor_cnt


def structure_data():
    voc_dict = pickle.load(open('/srv/glusterfs/xieya/data/language/vocabulary.p', 'rb'))
    im_file_2_cap_id = {}
    caps = []
    lens = []
    nocolor_cnt = 0
    color_voc = pickle.load(open('/srv/glusterfs/xieya/data/color/vocabulary.p', 'rb'))
    color_set = set()
    for c in color_voc:
        if c in voc_dict:
            color_set.add(voc_dict[c])
        else:
            print('{} not in vocabulary'.format(c))

    new_nocolor_cnt = _struct_data(
        '/srv/glusterfs/xieya/data/coco_seg/annotations/captions_train2014.json', 
        voc_dict, 
        color_set, 
        im_file_2_cap_id, 
        caps, 
        lens)
    # caps.extend(new_caps)
    # lens.extend(new_lens)
    nocolor_cnt += new_nocolor_cnt
    new_nocolor_cnt = _struct_data(
        '/srv/glusterfs/xieya/data/coco_seg/annotations/captions_val2014.json', 
        voc_dict, 
        color_set, 
        im_file_2_cap_id, 
        caps, 
        lens)
    # caps.extend(new_caps)
    # lens.extend(new_lens)
    nocolor_cnt += new_nocolor_cnt

    caps = np.asarray(caps)
    lens = np.asarray(lens)
    print('Total captions {}'.format(len(caps)))
    caps.dump('/srv/glusterfs/xieya/data/coco_seg/annotations/captions_comb.npy')
    lens.dump('/srv/glusterfs/xieya/data/coco_seg/annotations/caption_lengths_comb.npy')
    pickle.dump(im_file_2_cap_id, open('/srv/glusterfs/xieya/data/coco_seg/im2cap_comb.p', 'wb'))
    print('No color captions: {}'.format(nocolor_cnt))
    print(len(im_file_2_cap_id))


def filter_colored_regions(region_json_file):
    voc_dict = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/vocabulary.p', 'rb'))
    regions = json.load(open(os.path.join('/srv/glusterfs/xieya/data/visual_genome/', region_json_file), 'r'))
    print('Region json loaded.')
    color_voc = pickle.load(open('/srv/glusterfs/xieya/data/color/vocabulary.p', 'rb'))
    color_set = set()
    for c in color_voc:
        if c in voc_dict:
            color_set.add(voc_dict[c])
        else:
            print('{} not in vocabulary'.format(c))
    
    new_data = []
    region_count = 0
    has_color_count = 0

    for img in regions:
        img_id = img['id']
        new_regs = []
        for reg in img['regions']:
            region_count += 1
            c = _tokenize(reg['phrase'])
            has_color = False
            tokens = [0] * 20
            cap_len = min(len(c), 20)
            for i in xrange(cap_len):
                tokens[i] = voc_dict.get(c[i], 0)
                if tokens[i] in color_set:
                    has_color = True

            if has_color:
                has_color_count += 1
                new_reg = {'region_id': reg['region_id'], 'x': reg['x'], 'y': reg['y'], 'width': reg['width'], 'height': reg['height'], 'phrase': tokens, 'phrase_len': cap_len}
                new_regs.append(new_reg)

        if len(new_regs) > 0:
            new_data.append({'id': img_id, 'regions': new_regs})
        print(has_color_count)

    print('Total regions: {}'.format(region_count))
    print('Total images: {}'.format(len(new_data)))
    print('Regions with color: {}'.format(has_color_count))
    json.dump(new_data, open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', 'colored_' + region_json_file), 'w'))


def _scale_to_int(x, scale):
    return int(np.round(x / scale))


def scale_regions(region_file_name, keep_coco=True, size_thr=16, scale_size=224):
    regions = json.load(open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', region_file_name), 'r'))
    print('Region json loaded.')
    img_metas = json.load(open('/srv/glusterfs/xieya/data/visual_genome/image_data.json', 'r'))
    img_map = {}
    for i in xrange(len(img_metas)):
        img_map[img_metas[i]['image_id']] = i
    print('Image json loaded.')
    small_reg_cnt = 0
    new_regs = []
    im2regs = {}
    for img in regions:
        img_id = img['id']
        img_meta = img_metas[img_map[img_id]]
        coco_id = img_meta['coco_id']
        if keep_coco and coco_id is None:  #
            continue
        coco_id = '{:012d}'.format(int(coco_id))
        print(coco_id)
        img_w = img_meta['width']
        img_h = img_meta['height']
        scale = 1. * min(img_w, img_h) / scale_size
        for reg in img['regions']:
            x = reg['x']
            y = reg['y']
            w = reg['width']
            h = reg['height']

            nx = _scale_to_int(x, scale)
            ny = _scale_to_int(y, scale)
            nw = _scale_to_int(w, scale)
            nh = _scale_to_int(h, scale)
            if nw < size_thr or nh < size_thr:
                small_reg_cnt += 1
                continue
            reg['x'] = max(nx, 0)
            reg['y'] = max(ny, 0)
            reg['width'] = nw
            reg['height'] = nh
            reg_idx = len(new_regs)
            reg.pop('region_id')
            new_regs.append(reg)

            if coco_id not in im2regs:
                im2regs[coco_id] = [reg_idx]
            else:
                im2regs[coco_id].append(reg_idx)
        # print(len(new_regs))

    json.dump(new_regs, open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', '{}_sz{}_'.format(scale_size, size_thr) + region_file_name), 'w'))
    pickle.dump(im2regs, open('/srv/glusterfs/xieya/data/visual_genome/im2regs.p', 'wb'))
    print('Total images: {}'.format(len(im2regs)))
    print('Total regions: {}'.format(len(new_regs)))
    print('Small reg count: {}'.format(small_reg_cnt))


def main(img_list_file):
    with open(img_list_file, 'r') as fin:
        img_idx = 0
        img_cnt = 0
        for img_name in fin:
            if img_idx % _TASK_NUM == _TASK_ID:
                img_name = img_name.strip()
                _to_224(img_name)
                img_cnt += 1
                if img_cnt % _LOG_FREQ == 0:
                    print(img_cnt)
            img_idx += 1


if __name__ == "__main__":
    # main('/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val.txt')
    structure_data()
    # word_count()
    # word_count_vg()
    # word_to_emb('glove.6B.300d.p', 'visual_genome')
    # filter_colored_regions('region_descriptions.json')
    # scale_regions('colored_region_descriptions.json', size_thr=20, scale_size=224)
    # combine_voc('glove.6B.300d.p')
