#!/bin/bash -ex

if [ $# != 5 ]; then
    echo "usage: $0 mb lr decay_factor outer_step inner_step"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python3 ./audio2text/convert_train_oracle.py --init_lr $2 --penalty_lambda 0.5 --decay_factor $3 --mb $1 --max_step $4 --max_inner_step $5 ./embedding/en_5000.vec.new ./embedding/es_5000.vec.new
