#!/bin/bash

[ -f path.sh ] && . ./path.sh

path=/home/grtzsohalf/Audio-Word2Vec
feat_dir=/nfs/YueLao/grtzsohalf/yeeee/English

sampling=0.0001
min_count=5

subsampled_examples=$feat_dir/subsampled_examples
subsampled_labels=$feat_dir/subsampled_labels
subsampled_utters=$feat_dir/subsampled_utters
mkdir -p $subsampled_examples
mkdir -p $subsampled_labels
mkdir -p $subsampled_utters

if [ ! -f $feat_dir/sub_extracted ];then
  python3 $path/src/subsample.py $feat_dir/all_examples $feat_dir/all_labels $feat_dir/all_utters \
    $subsampled_examples $subsampled_labels $subsampled_utters $sampling $min_count
  echo 1 > $feat_dir/sub_extracted
fi
