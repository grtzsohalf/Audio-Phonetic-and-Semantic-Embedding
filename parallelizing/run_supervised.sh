#!/bin/bash -ex


if [ $# != 2]; then
  echo "$0 audio_emb text_emb"
  exit 1
fi


text=$1
audio=$2

./convert_train_oracle.py --init_lr 0.01 --mb 20 --max_step 1000 $audio $text
