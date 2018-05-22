import os
import numpy as np
from collections import Counter
from collections import deque

gram_num = 5
min_count = 5
dim = 128
num_words = 5000
# data_type = '2_layer_sigmoid/0.001_128'
data_type = 'bag_2_layer_sigmoid/0.001_128'
embedding_file = '/nfs/Caishen/grtzsohalf/yeeee/English/embedding/'+data_type+'/train_embeddings_5_with_words'
freq_embedding_file = '/nfs/Caishen/grtzsohalf/yeeee/English/embedding/'+data_type+'/train_freq_embeddings_5_with_words'
word_count_file = '/nfs/Caishen/grtzsohalf/yeeee/English/word_count_all'

freq_words = []
count = 0
with open(word_count_file, 'r') as f_count:
    for line in f_count:
        if count == 5000:
            break
        line = line[:-1].split(', ')
        word = line[0]
        freq_words.append(word)
        count += 1

with open(embedding_file, 'r') as f_emb:
    with open(freq_embedding_file, 'w') as f_freq:
        line = f_emb.readline()
        f_freq.write(str(num_words)+' '+str(128)+'\n')
        for line in f_emb:
            word = line[:-1].split()[0]
            if not word in freq_words:
                continue
            f_freq.write(line)
