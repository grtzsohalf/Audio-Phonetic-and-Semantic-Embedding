#!/bin/bash

[ -f path.sh ] && . ./path.sh || exit

if [ $# != 4 ] ; then 
  echo "usage: test.sh <lr> <hidden_dim> <CUDA_DEVICE>"
  echo "e.g. test.sh 0.0001 128 128 0"
  exit 1
fi

batch_size=64
gram_num=5
neg_sample_num=5
min_count=5
sampling_factor=0.001
#path=/home/grtzsohalf/Audio-Word2Vec
#feat_dir=/nfs/Caishen/grtzsohalf/yeeee/English
init_lr=$1
feat_dim=$2
dim=$3
device_id=$4

exp_dir=$exp_dir/2_layer_sigmoid
mkdir -p $exp_dir
model_dir=$exp_dir/model_lr${init_lr}_$dim
log_dir=$exp_dir/log_lr${init_lr}_$dim

tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

embedding_dir=$feat_dir/embedding_all/2_layer_sigmoid/${init_lr}_$dim

mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir
mkdir -p $embedding_dir

### testing ###
export CUDA_VISIBLE_DEVICES=$device_id

python3 $path/src/test.py --init_lr=$init_lr --batch_size=$batch_size --feat_dim=$feat_dim \
  --hidden_dim=$dim --gram_num=$gram_num --neg_sample_num=$neg_sample_num --min_count=$min_count $tf_log_dir $tf_model_dir \
  --sampling_factor=$sampling_factor $feat_dir/all_examples $feat_dir/all_words $feat_dir/all_utters $embedding_dir/train_embeddings
