import argparse
import os
import sys
import random
import numpy as np
import time
from tqdm import tqdm
import pickle

def split_data(feat_dir, n_files, proportion, seq_len):
    feats_dir = os.path.join(feat_dir, 'pkls', str(seq_len))
    train_scp = os.path.join(feat_dir, 'train_AE.scp')
    test_scp = os.path.join(feat_dir, 'test_AE.scp')
    file_list = os.listdir(feats_dir)[:n_files]
    # random.shuffle(file_list)
    n_train_files = int(n_files*proportion)
    print ("Training number of speakers: " + str(n_train_files))
    print ("Testing number of speakers: " + str(n_files - n_train_files))
    with open(train_scp, 'w') as fout:
        for file in file_list[:n_train_files]:
            fout.write(file + '\n')
    with open(test_scp, 'w') as fout:
        for file in file_list[n_train_files:]:
            fout.write(file + '\n')

def load_data(feats_dir, scp_file):
    feats = []
    spk2feat = {}
    feat2label = {}
    spk_list = []
    count = 0
    with open(scp_file, 'r') as fin1:
        for line in tqdm(fin1):
            line = line[:-1]
            spk_list.append(line)
            spk_indices = []
            with open(os.path.join(feats_dir, line), 'rb') as fin2:
                data = pickle.load(fin2)
                for feat in data:
                    feats.append(feat[0])
                    feat2label[count] = (str(int(float(feat[1]))), feat[2], line)
                    spk_indices.append(count)
                    count += 1
                spk2feat[line] = spk_indices
    print ('# of words: ' + str(count))
    return count, feats, spk2feat, feat2label, spk_list

def batch_pair_data(feats, spk2feat, feat2label, feat_indices, spk_list):
    batch_data = []
    batch_data_pos = []
    batch_data_neg = []
    for idx in feat_indices:
        # feat
        feat = feats[idx]
        word = feat2label[idx][0]
        utter = feat2label[idx][1]
        spk = feat2label[idx][2]

        # feat_pos
        idx_pos = random.choice(spk2feat[spk])
        feat_pos = feats[idx_pos]

        # feat_neg
        spk_list.remove(spk)
        rand_spk = random.choice(spk_list)
        spk_list.append(spk)
        idx_neg = random.choice(spk2feat[rand_spk])
        feat_neg = feats[idx_neg]
        batch_data.append(feat)
        batch_data_pos.append(feat_pos)
        batch_data_neg.append(feat_neg)
        # print (spk + ': (' + str(idx) + ', ' + str(idx_pos) + ') <---> ' + rand_spk + ': ' + str(idx_neg))
    return np.array(batch_data, dtype=np.float32), np.array(batch_data_pos, dtype=np.float32), \
        np.array(batch_data_neg, dtype=np.float32)
