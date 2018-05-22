import argparse
import os
import sys
import random
import numpy as np
import time
from tqdm import tqdm
from collections import Counter
from collections import deque
import operator
import math

def load_centroids(data_file):
    centroids = []
    count = 0
    with open(data_file, 'r') as fin:
        for line in fin:
            line = line[:-1].split()
            centroids.append(list(map(float, line)))
            count += 1
    return count, np.array(centroids, dtype=np.float32)

def batch_centroids(centroids, indices):
    batch_c = []
    for idx in indices:
        batch_c.append(centroids[idx])
    return np.array(batch_c, dtype=np.float32)

def load_data(gram_num, label_dir, utter_dir, dic_file, min_count):
    label_files = os.listdir(label_dir)
    utter_files = os.listdir(utter_dir)

    word_freq = Counter()
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f_l:
            for line in f_l:
                word = line[:-1]
                word_freq[word] += 1
    sorted_words = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)
    with open(os.path.join(label_dir, '../word_count'), 'w') as fout:
        for word in sorted_words:
            fout.write(word[0] + ', ' + str(word[1]) + '\n')

    feats = []
    feat_idx = []
    skip_feat_idx = []
    utters = []
    dic = {'-1': 0}
    label_count = 1
    count = 0
    for (label_file, utter_file) in zip(label_files, utter_files):
        with open(os.path.join(label_dir, label_file), 'r') as f_label:
            with open(os.path.join(utter_dir, utter_file), 'r') as f_utter:
                for label, utter in tqdm(zip(f_label, f_utter)):
                    feats.append(label[:-1])
                    utters.append(utter[:-1])
    for i, feat in enumerate(feats):
        if i < gram_num or len(feats)-i <= gram_num:
            continue
        continue_bool = False
        # for ff in feats[i-gram_num:i+gram_num+1]:
            # if word_freq[ff] < min_count:
                # continue_bool = True
                # break
        # if continue_bool:
            # continue
        if word_freq[feats[i]] < min_count:
            continue

        for ff in feats[i-gram_num:i+gram_num+1]:
            if not ff in dic:
                if word_freq[ff] >= min_count:
                    dic[ff] = label_count
                    label_count += 1

        feat_idx.append(dic[feat])
        skip_feat = []
        ut = utters[i]
        for j, (ff, uu) in enumerate(zip(feats[i-gram_num:i+gram_num+1], utters[i-gram_num:i+gram_num+1])):
            if j != gram_num:
                if word_freq[ff] >= min_count and ut == uu:
                    skip_feat.append(dic[ff])
                else:
                    skip_feat.append(0)
        skip_feat_idx.append(skip_feat)
        count += 1
    print ('# of feats: ' + str(count))
    idx_freq = {}
    for word in word_freq:
        if word in dic:
            idx_freq[dic[word]] = word_freq[word] / count
    with open(dic_file, 'w') as f_dic:
        f_dic.write('label idx\n')
        for label, idx in dic.items():
            f_dic.write(label+' '+str(idx)+'\n')
    return count, np.array(feat_idx), np.array(skip_feat_idx), dic, idx_freq

def batch_pair_data(feats, skip_feats, feat_indices, neg_num):
    batch_pos_feat = []
    batch_neg_feats = []
    batch_skip_feats = []
    for idx in feat_indices:
        # choices = []
        # for ii in skip_feats[idx]:
            # if ii != 0:
                # choices.append(ii)
        # if len(choices) == 0:
            # continue
        feat = feats[idx]
        # ch = random.choice(choices)
        batch_pos_feat.append(feat)
        batch_skip_feats.append(skip_feats[idx])
        # for skip_feat in skip_feats[idx]:
        for i in range(neg_num):
            neg_feat = None
            while_bool = True
            while while_bool:
                while_bool = False
                neg_feat = random.choice(feats)
                if neg_feat == feat:
                    while_bool = True
                    continue
                for skip in skip_feats[idx]:
                    if neg_feat == skip:
                        while_bool = True
                        break
            batch_neg_feats.append(neg_feat)
    return np.array(batch_pos_feat, dtype=np.int64), \
           np.array(batch_neg_feats, dtype=np.int64), \
           np.array(batch_skip_feats, dtype=np.int64)
