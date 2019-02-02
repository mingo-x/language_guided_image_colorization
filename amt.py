import csv
import os
import pickle
import random


def prepare_data(total=50, group_size=11, mode=0, with_test=False):
    '''
    Randomly select images for evaluation.
    '''
    gray_list = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/val_filtered_gray.p', 'rb'))
    gray_pattern = 'https://s3.eu-central-1.amazonaws.com/imagecolorization/gray/{}.jpg'
    gt_pattern = 'https://s3.eu-central-1.amazonaws.com/imagecolorization/segcap_21/{}.jpg'  # 0
    if mode == 0:  # CIC v.s. gt
        base_pattern = 'https://s3.eu-central-1.amazonaws.com/imagecolorization/baseline/{}.jpg'  # 1
    elif mode == 1:  # gt v.s. new
        base_pattern = 'https://s3.eu-central-1.amazonaws.com/imagecolorization/segcap_21_new/{}.jpg'  # 1
    _total = 2487

    with open('/srv/glusterfs/xieya/data/coco_seg/amt_data_3.csv', 'wb') as fout:
        csv_writer = csv.writer(fout, delimiter=',')
        header = []
        for i in xrange(group_size):
            header.extend(
                ['gray_url_{}'.format(i + 1), 
                 'image_A_url_{}'.format(i + 1), 
                 'image_B_url_{}'.format(i + 1), 
                 'gt_{}'.format(i + 1)])
        csv_writer.writerow(header)
        idx = 0
        for i in xrange(total):
            csv_row = []
            has_test = not with_test
            for j in xrange(group_size):
                if not has_test:
                    if (random.random() < 1. / group_size) or (j == group_size - 1):
                        # Insert a test case.
                        r = random.choice(gray_list)
                        if random.random() < 0.5:
                            # A: gt, B: base
                            csv_row.extend([gray_pattern.format(r), gt_pattern.format(r), base_pattern.format(r), 'A1'])
                        else:
                            # A: base, B: gt
                            csv_row.extend([gray_pattern.format(r), base_pattern.format(r), gt_pattern.format(r), 'B1'])
                        has_test = True
                        print('group {} id {}'.format(i, j))
                        continue
                idx += 1
                while idx + 1 < _total and (random.random() >= 0.25 or idx in gray_list):
                    idx += 1
                if random.random() < 0.5:
                    # A: gt, B: base
                    csv_row.extend([gray_pattern.format(idx), gt_pattern.format(idx), base_pattern.format(idx), 'A'])
                else:
                    # A: base, B: gt
                    csv_row.extend([gray_pattern.format(idx), base_pattern.format(idx), gt_pattern.format(idx), 'B'])
            csv_writer.writerow(csv_row)


def calculate_score(group_size=11):
    '''
    Calculate the preference rate from the annotation results.
    '''
    with open('data/amt_data_5_res.csv', 'r') as flabel, open('data/amt_data_5.csv', 'r') as fdata:
        label_reader = csv.reader(flabel, delimiter=',')
        data_reader = csv.reader(fdata, delimiter=',')
        next(label_reader)
        next(data_reader)
        gt_count = 0
        total = 0

        for data in data_reader:
            labels = next(label_reader)
            for i in range(group_size):
                target = data[i * 4 + 3]
                if target == 'A1' or target == 'B1':
                    continue
                total += 1
                if 'A' in target and labels[i] == 'A':
                    gt_count += 1
                if 'B' in target and labels[i] == 'B':
                    gt_count += 1

        print(gt_count)
        print(total)


def replace_model(old, new):
    '''
    Replace the model id in the previously generated evaluation image list, 
    so as to assure that we use the same set of images when comparing different models.
    '''
    cnt = 0
    with open('data/amt_data_3.csv', 'r') as fin, open('data/amt_data_5.csv', 'w', newline='') as fout:
        csv_reader = csv.reader(fin, delimiter=',')
        csv_writer = csv.writer(fout, delimiter=',')
        header = next(csv_reader)
        csv_writer.writerow(header)

        for data in csv_reader:
            new_row = []
            for cell in data:
                if old in cell:
                    cell = cell.replace(old, new)
                    cnt += 1
                new_row.append(cell)
            csv_writer.writerow(new_row)
        print(cnt)


def get_ids():
    ids = []
    with open('data/amt_data_2.csv', 'r') as fin:
        csv_reader = csv.reader(fin, delimiter=',')
        next(csv_reader)
        for data in csv_reader:
            for i in range(11):
                p = data[4 * i]
                f = os.path.split(p)[1]
                x = int(os.path.splitext(f)[0])
                if x in ids:
                    print(x)
                ids.append(x)
    pickle.dump(ids, open('data/quality.p', 'wb'), 2)
    print(len(ids))


if __name__ == "__main__":
    # prepare_data(mode=1, group_size=10)
    calculate_score(10)
    # replace_model('segcap_21', 'language_2')
    # get_ids()
