#!/bin/bash

[ -f path.sh ] && . ./path.sh || exit
#path=/home/grtzsohalf/Audio-Word2Vec
#feat_dir=/nfs/Caishen/grtzsohalf/yeeee/English
phonetic_dir=$feat_dir/phonetic_all

feat_dim=128
#gram_num=4

[ -d $phonetic_dir ] || exit 1
[ -f $feat_dir/all_examples ] && rm -rf $feat_dir/all_examples
[ -f $feat_dir/all_labels ] && rm -rf $feat_dir/all_labels
[ -f $feat_dir/all_utters ] && rm -rf $feat_dir/all_utters

mkdir -p $feat_dir/all_examples
mkdir -p $feat_dir/all_labels
mkdir -p $feat_dir/all_utters

python3 $path/utils/pack_examples.py --feat_dim=$feat_dim \
$phonetic_dir $feat_dir/all_examples $feat_dir/all_labels $feat_dir/all_utters
