#!/bin/bash

[ -f path.sh ] && . ./path.sh || exit

if [ $# != 4 ] ; then 
  echo "usage: train.sh <lr> <hidden_dim> <CUDA_DEVICE> <n_epochs for training>"
  echo "e.g. train.sh 0.001 128 0 20"
  exit 1
fi

#train_num=1246455
batch_size=64
feat_dim=256
gram_num=5
neg_sample_num=5
min_count=5
sampling_factor=0.001
#path=/home/grtzsohalf/Audio-Word2Vec
#feat_dir=/nfs/Caishen/grtzsohalf/yeeee/English
init_lr=$1
dim=$2
device_id=$3
n_epochs=$4

exp_dir=$exp_dir/semantic_exps/uni_2_layer_sigmoid_exp
mkdir -p $exp_dir
model_dir=$exp_dir/model_lr${init_lr}_$dim
log_dir=$exp_dir/log_lr${init_lr}_$dim

tf_model_dir=$model_dir/tf_model
tf_log_dir=$log_dir/tf_log

embedding_dir=$feat_dir/embedding/uni_2_layer_sigmoid/${init_lr}_$dim

mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $tf_model_dir
mkdir -p $tf_log_dir

mkdir -p $embedding_dir

### training ###
export CUDA_VISIBLE_DEVICES=$device_id
python3 $path/src/train.py --init_lr=$init_lr --batch_size=$batch_size --feat_dim=$feat_dim \
  --hidden_dim=$dim --n_epochs=$n_epochs --gram_num=$gram_num --neg_sample_num=$neg_sample_num --min_count=$min_count\
  --sampling_factor=$sampling_factor $tf_log_dir $tf_model_dir \
  $feat_dir/all_unis $feat_dir/all_words $feat_dir/all_utters #$train_num
