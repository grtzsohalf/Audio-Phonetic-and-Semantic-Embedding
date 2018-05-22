#!/bin/bash

[ -f path.sh ] && . ./path.sh || exit
#path=/home/grtzsohalf/Audio-Word2Vec
#feat_dir=/nfs/Caishen/grtzsohalf/yeeee/English
feat_file=/nfs/Guanyin/hoa/Yichen/all_data

feat_dim=400
#gram_num=4

[ -f $feat_file ] || exit 1
[ -f $feat_dir/original_examples ] && rm -rf $feat_dir/original_examples
[ -f $feat_dir/original_labels ] && rm -rf $feat_dir/original_labels
[ -f $feat_dir/original_utters ] && rm -rf $feat_dir/original_utters

mkdir -p $feat_dir/original_examples
mkdir -p $feat_dir/original_labels
mkdir -p $feat_dir/original_utters

python3 $path/utils/pack_original.py --feat_dim=$feat_dim \
$feat_file $feat_dir/original_examples $feat_dir/original_labels $feat_dir/original_utters
