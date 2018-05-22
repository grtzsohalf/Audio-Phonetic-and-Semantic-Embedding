#!/bin/bash -ex

if [ $# != 2 ]; then
    echo "usage: $0 audio_emb text_emb"
    exit 1
fi


mb=200
init_lr=0.01
decay_factor=0.9
max_step=1000
max_instep=2000

export CUDA_VISIBLE_DEVICES=0
python3 ./audio2text/convert_train_oracle.py --init_lr $init_lr --penalty_lambda 0.5 --decay_factor $decay_factor --mb $mb --max_step $max_step --max_inner_step $max_instep $1 $2 
