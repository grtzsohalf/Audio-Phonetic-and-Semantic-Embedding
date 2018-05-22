import sys
import re
import os 
import argparse
from collections import deque
from tqdm import tqdm

def skip_gram_utter(que, gram_num):
    utters = ''
    for i, utter in enumerate(que):
        if i != gram_num:
            utters += (str(utter) + ' ')
    return utters[:-1]

def skip_gram_label(que, gram_num):
    labels = ''
    for i, label in enumerate(que):
        if i != gram_num:
            labels += (str(label) + ' ')
    return labels[:-1]

def skip_gram_feat(feats_que, gram_num):
    feats = ''
    for i, feat in enumerate(feats_que):
        if i != gram_num:
            feats += (feat + ' ')
    return feats[:-1]

def get_clean(filename):
    clean_spks = []
    with open(filename, 'r') as fin:
        fin.readline()
        for line in fin:
            line = line[:-1].split()
            ID = line[0]
            subset = line[4].split('-')[1]
            if subset == 'clean':
                clean_spks.append(ID)
        return clean_spks

def main():
    # clean_spks = get_clean(os.path.join(FLAG.all_examples, '../SPEAKERS.TXT'))
    # print (clean_spks)
    count = 0
    spk_count = -1
    last_spk = None
    with open(FLAG.feat_file, 'r') as f_feats:
        for line in tqdm(f_feats):
            line = line[:-1].split(',')
            utter = line[-1]
            spk = utter.split('-')[0]
            # if not spk in clean_spks:
                # continue
            feats = ''
            for feat in line[:-2]:
                feats += (feat + ' ')
            feats = feats[:-1]
            label = line[-2]
            if spk != last_spk:
                spk_count += 1
                last_spk = spk
            example_file = os.path.join(FLAG.all_examples, 'example_'+str(spk_count // 200))
            label_file = os.path.join(FLAG.all_labels, 'label_'+str(spk_count // 200))
            utter_file = os.path.join(FLAG.all_utters, 'utter_'+str(spk_count // 200))
            # window_size = (2*FLAG.gram_num+1)
            # feats_que = deque()
            # labels_que = deque()
            # utters_que = deque()
            with open(example_file, 'a') as f_examples:
                with open(label_file, 'a') as f_labels:
                    with open(utter_file, 'a') as f_utters:
                            f_examples.write(feats + '\n')
                            f_labels.write(label + '\n')
                            f_utters.write(utter + '\n')
                            count += 1
        print ('spk count: '+str(spk_count+1))
        print ('total # of feats: '+str(count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='pack the examples for VQAudio2Vec')
    parser.add_argument('feat_file',
        help='the file to store the feat vectors')
    parser.add_argument('all_examples',
        help='the dir to store all examples')
    parser.add_argument('all_labels',
        help='the dir to store all labels')
    parser.add_argument('all_utters',
        help='the dir to store all utters')

    parser.add_argument('--feat_dim', type=int,
        default=256,
        help='the feat dimension, default=256')
    # parser.add_argument('--gram_num', type=int,
        # default=2,
        # help='the feat dimension, default=2')
    
    FLAG = parser.parse_args()
    # ph_file_list = sorted(os.listdir(FLAG.ph_vec_dir))
    # print (ph_file_list)

    main()
    
