#!/bin/bash -ex

if [ $# != 4 ]; then
    echo "usage: $0 mb lr outer_step inner_step"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python3 ./audio2text/convert_train.py --init_lr $2 --penalty_lambda 0.5 --decay_factor 0.9 --mb $1 --max_step $3 --max_inner_step $4 ./embedding/sorted_audio ./embedding/sorted_text_new
