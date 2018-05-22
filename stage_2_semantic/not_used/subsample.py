#!/usr/bin/env python3

import argparse
import os
import sys
import random
import math
from tqdm import tqdm
from collections import Counter
import operator

import numpy as np

FLAGS = None

next_random = 1

def subsampling_bool(freq, sampling):
    global next_random
    next_random = (next_random * 25214903917 + 11) & 0xFFFF
    prob = (math.sqrt(sampling/freq)+sampling/freq)
    return next_random/65536.0 - (math.sqrt(sampling/freq)+sampling/freq)
    # return 1. - (math.sqrt(sampling/freq)+sampling/freq)

def subsample(label_dir, label_list, sampling, min_count):
    word_dic = Counter()
    count = 0
    for l_file in label_list:
        with open(os.path.join(label_dir, l_file), 'r') as f_l:
            for line in f_l:
                word = line.split(',')[0]
                word_dic[word] += 1
                count += 1
    sorted_words = sorted(word_dic.items(), key=operator.itemgetter(1), reverse=True)
    with open(os.path.join(label_dir, '../word_count'), 'w') as fout:
        for word in sorted_words:
            fout.write(word[0] + ', ' + str(word[1]) + '\n')
    prob_dic = {}
    for word in sorted_words:
        if word[1] < min_count:
            prob_dic[word[0]] = 1.
        else:
            prob_dic[word[0]] = subsampling_bool(word[1]/count, sampling)
            if prob_dic[word[0]] < 0.:
                prob_dic[word[0]] = 0.
    return prob_dic

def main():
    example_list = os.listdir(FLAGS.example_dir)
    label_list = os.listdir(FLAGS.label_dir)
    utter_list = os.listdir(FLAGS.utter_dir)
    num_file = len(example_list)
    subsampling_dic = subsample(FLAGS.label_dir, label_list, FLAGS.sampling, FLAGS.min_count)
    subsampled_words = Counter()
    for u_file, e_file, l_file in tqdm(zip(utter_list, example_list, label_list)):
        count = 0
        with open(os.path.join(FLAGS.example_dir, e_file), 'r') as f_e:
            with open(os.path.join(FLAGS.label_dir, l_file), 'r') as f_l:
                with open(os.path.join(FLAGS.utter_dir, u_file), 'r') as f_u:
                    with open(os.path.join(FLAGS.subsampled_example_dir, e_file), 'w') as f_out_e:
                        with open(os.path.join(FLAGS.subsampled_label_dir, l_file), 'w') as f_out_l:
                            with open(os.path.join(FLAGS.subsampled_utter_dir, u_file), 'w') as f_out_u:
                                for u, e, l in zip(f_u, f_e, f_l):
                                    count += 1
                                    label = l.split(',')[0]
                                    utter = u.split(',')[0]
                                    spk = utter.split('-')[0]
                                    context_labels = l[:-1].split(',')[1].split()
                                    write_bool = True
                                    for c_l in context_labels:
                                        if subsampling_dic[c_l] == 1.:
                                            write_bool = False
                                            break
                                    if write_bool == False:
                                        continue
                                    prob = subsampling_dic[label]
                                    if np.random.choice([True, False], p=[1-prob, prob]):
                                        try:
                                            f_out_e.write(e[:-1]+'\n')
                                            f_out_l.write(l[:-1]+'\n')
                                            # f_out_u.write(spk+'\n')
                                            f_out_u.write(u[:-1]+'\n')
                                            subsampled_words[label] += 1
                                        except:
                                            print (l)
        print (count)
    sorted_words = sorted(subsampled_words.items(), key=operator.itemgetter(1), reverse=True)
    with open(os.path.join(FLAGS.label_dir, '../subsampled_word_count'), 'w') as fout:
        for word in sorted_words:
            fout.write(word[0] + ', ' + str(word[1]) + '\n')

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description = 
         'transform text format features into tfrecords')

   parser.add_argument(
        'example_dir',
        metavar='<example dir>',
        type=str,
        help='example dir'
        )
   parser.add_argument(
        'label_dir',
        metavar='<label dir>',
        type=str,
        help='label dir'
        )
   parser.add_argument(
        'utter_dir',
        metavar='<utter dir>',
        type=str,
        help='utter dir'
        )
   parser.add_argument(
        'subsampled_example_dir',
        metavar='<subsampled example dir>',
        type=str,
        help='subsampled_example_dir'
        )
   parser.add_argument(
        'subsampled_label_dir',
        metavar='<subsampled label dir>',
        type=str,
        help='subsampled_label_dir'
        )
   parser.add_argument(
        'subsampled_utter_dir',
        metavar='<subsampled utter dir>',
        type=str,
        help='subsampled_utter_dir'
        )
   parser.add_argument(
        'sampling',
        metavar='<subsampling factor>',
        type=float,
        help='subsampling factor'
        )
   parser.add_argument(
        'min_count',
        metavar='<min count>',
        type=int,
        help='min count'
        )
   parser.add_argument(
        '--feats_dim',
        metavar='<feats-dim>',
        type=int,
        default=256,
        help='feature dimension'
        )
   parser.add_argument(
        '--norm_var',
        metavar='<True|False>',
        type=bool,
        default=False,
        help='Normalize Variance of each sentence'
        )
   parser.add_argument(
        '--norm_mean',
        metavar='<True|False>',
        type=bool,
        default=False,
        help='Normalize mean of each sentence'
        )
   FLAGS = parser.parse_args()
   main()
