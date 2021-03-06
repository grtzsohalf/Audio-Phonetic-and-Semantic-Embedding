#!/bin/bash

[ -f path.sh ] && . ./path.sh || exit

if [ $# != 3 ] ; then 
  echo "usage: test.sh <lr> <hidden_dim> <CUDA_DEVICE>"
  echo "e.g. test.sh 0.001 128 0"
  exit 1
fi

batch_size=64
feat_dim=71
gram_num=5
neg_sample_num=5
min_count=5
sampling_factor=0.001
#path=/home/grtzsohalf/Audio-Word2Vec
#feat_dir=/nfs/Caishen/grtzsohalf/yeeee/English
init_lr=$1
dim=$2
device_id=$3

exp_dir=$exp_dir/semantic_exps/bag_2_layer_sigmoid_exp
mkdir -p $exp_dir
model_dir=$exp_dir/model_lr${init_lr}_$dim
log_dir=$exp_dir/log_lr${init_lr}_$dim

tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

embedding_dir=$feat_dir/embedding/bag_2_layer_sigmoid/${init_lr}_$dim

mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir
mkdir -p $embedding_dir

### testing ###
export CUDA_VISIBLE_DEVICES=$device_id

python3 $path/src/test.py --init_lr=$init_lr --batch_size=$batch_size --feat_dim=$feat_dim \
  --hidden_dim=$dim --gram_num=$gram_num --neg_sample_num=$neg_sample_num --min_count=$min_count $tf_log_dir $tf_model_dir \
  --sampling_factor=$sampling_factor $feat_dir/all_bags $feat_dir/all_words $feat_dir/all_utters $embedding_dir/train_embeddings
