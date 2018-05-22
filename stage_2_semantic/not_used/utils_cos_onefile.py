import argparse
import os
import sys
import random
import numpy as np
import time
from tqdm import tqdm
from collections import Counter
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

def subsampling_bool(freq, sampling=0.0001):
    prob = (math.sqrt(sampling/freq)+sampling/freq)
    return (math.sqrt(sampling/freq)+sampling/freq) - np.random.rand()

def subsample_data(feats, labels, utters, spks, word_freq, total_count, 
                         min_count, gram_num, sampling_factor, subsample=True):
    feat_idx = []
    skip_feat_idx = []
    masks = []
    spk2idx = {}
    idx2spk = {}
    count = 0
    for i, l in enumerate(labels):
        if i < gram_num or len(labels)-i <= gram_num:
            continue
        continue_bool = False
        # for ll in labels[i-gram_num:i+gram_num+1]:
            # if word_freq[ll] < min_count:
                # continue_bool = True
        if word_freq[l] < min_count:
            continue_bool = True
        if subsample == True and subsampling_bool(word_freq[l]/total_count, sampling_factor) < 0.:
            continue_bool = True
        if continue_bool:
            continue
        feat_idx.append(i)
        skip_feat_idx.append([j for ii, j in enumerate(range(i-gram_num, i+gram_num+1)) if (ii != gram_num)])
        tmp = []
        utter = utters[i]
        for ii, (ll, skip_utter) in enumerate(zip(labels[i-gram_num:i+gram_num+1], utters[i-gram_num:i+gram_num+1])):
            if ii != gram_num:
                if word_freq[ll] < min_count or skip_utter != utter:
                    tmp.append(0)
                else:
                    tmp.append(1)
        masks.append(tmp)
        spk = spks[i]
        if not spk in spk2idx:
            spk2idx[spk] = [i]
        else:
            spk2idx[spk].append(i)
        idx2spk[i] = spk
        count += 1

    print ('# of feats after: ' + str(count))
    return count, feat_idx, skip_feat_idx, spk2idx, idx2spk, masks

def load_data(example_file, label_file, utter_file):
    feats = []
    labels = []
    utters = []
    spks = []
    count = 0
    with open(example_file, 'r') as f_example:
        with open(label_file, 'r') as f_label:
            with open(utter_file, 'r') as f_utter:
                for example, label, utter in tqdm(zip(f_example, f_label, f_utter)):
                    example = example[:-1].split()
                    feats.append(list(map(float, example)))
                    labels.append(label[:-1])
                    utter = utter[:-1]
                    utters.append(utter)
                    spk = utter.split('-')[0]
                    spks.append(spk)
                    count += 1
    print ('# of original feats: ' + str(count))

    return feats, labels, utters, spks

def load_test_data(example_file, label_file, utter_file, gram_num):
    feats = []
    feat_idx = []
    skip_feat_idx = []
    labels = []
    spks = []
    count = 0
    with open(example_file, 'r') as f_example:
        with open(label_file, 'r') as f_label:
            with open(utter_file, 'r') as f_utter:
                for example, label, utter in tqdm(zip(f_example, f_label, f_utter)):
                    example = example[:-1].split()
                    feats.append(list(map(float, example)))
                    labels.append(label[:-1])
                    utter = utter[:-1]
                    spk = utter.split('-')[0]
                    spks.append(spk)
                    count += 1

    spk2idx = {}
    idx2spk = {}
    count = 0
    for i, feat in enumerate(feats):
        if i < gram_num or len(feats)-i <= gram_num:
            continue
        feat_idx.append(i)
        skip_feat_idx.append([j for ii, j in enumerate(range(i-gram_num, i+gram_num+1)) if (ii != gram_num)])
        spk = spks[i]
        if not spk in spk2idx:
            spk2idx[spk] = [i]
        else:
            spk2idx[spk].append(i)
        idx2spk[i] = spk
        count += 1
    print ('# of feats: ' + str(count))
    return count, feats, feat_idx, skip_feat_idx, labels, spk2idx, idx2spk

def batch_pair_data(feats, feat_idx, skip_feat_idx, labels, spk2idx, idx2spk, masks, feat_indices, neg_num):
    batch_pos_feat = []
    batch_neg_feats = []
    batch_skip_feats = []
    batch_labels = []
    batch_skip_labels = []
    batch_neg_labels = []
    # batch_masks = []

    for i in feat_indices:
        # for ii in skip_feat_idx[i]:
            # batch_skip_feats.append(feats[ii])
            # batch_neg_feats.append([])
            # for j in range(neg_num):
                # neg_idx = -1
                # neg_label = None
                # while_bool = True
                # while while_bool:
                    # while_bool = False
                    # # neg_idx = random.choice(spk2idx[spk])
                    # neg_idx = random.choice(feat_idx)
                    # neg_label = labels[neg_idx]
                    # # if neg_label == labels[idx]:
                        # # while_bool = True
                        # # continue
                    # if neg_label == labels[ii]:
                        # while_bool = True
                        # break
                # batch_neg_feats[-1].append(feats[neg_idx])
        # batch_masks.append(masks[i])
        # spk = idx2spk[idx]
        
        choices = []
        for ii, m in zip(skip_feat_idx[i], masks[i]):
            if m != 0:
                choices.append(ii)
        if len(choices) == 0:
            # print ('empty!')
            continue
        idx = feat_idx[i]
        batch_pos_feat.append(feats[idx])
        batch_labels.append(labels[idx])
        ch = random.choice(choices)
        batch_skip_feats.append(feats[ch])
        batch_skip_labels.append(labels[ch])
        for j in range(neg_num):
            neg_idx = -1
            neg_label = None
            while_bool = True
            while while_bool:
                while_bool = False
                # neg_idx = random.choice(spk2idx[spk])
                neg_idx = random.choice(feat_idx)
                neg_label = labels[neg_idx]
                # if neg_label == labels[idx]:
                    # while_bool = True
                    # continue
                if neg_label == labels[ch]:
                    while_bool = True
                    continue
            batch_neg_feats.append(feats[neg_idx])
            batch_neg_labels.append(labels[neg_idx])
        # batch_masks.append(masks[i])
        spk = idx2spk[idx]
    return np.array(batch_pos_feat, dtype=np.float32), \
           np.array(batch_neg_feats, dtype=np.float32), \
           np.array(batch_skip_feats, dtype=np.float32), \
           np.array(batch_labels), \
           np.array(batch_skip_labels), \
           np.array(batch_neg_labels)

def batch_pair_test_data(feats, feat_idx, skip_feat_idx, feat_indices):
    batch_pos_feat = []
    batch_skip_feats = []
    for i in feat_indices:
        batch_pos_feat.append(feats[idx])
        for ii in skip_feat_idx[i]:
            batch_skip_feats.append(feats[ii])
    return np.array(batch_pos_feat, dtype=np.float32), \
           np.array(batch_skip_feats, dtype=np.float32)
