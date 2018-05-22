import os, sys
import numpy as np

dim = 128
data_type = 'text_sigmoid/0.001_128'
embedding_file = '/nfs/Caishen/grtzsohalf/yeeee/English/embedding/'+data_type+'/train_embeddings_word_9'
embedding_with_word_file = '/nfs/Caishen/grtzsohalf/yeeee/English/embedding/'+data_type+'/train_embeddings_word_9_with_words'
dic_file = '/nfs/Caishen/grtzsohalf/yeeee/English/embedding/'+data_type+'/label2embedding'
# word_file = '/nfs/Mazu/grtzsohalf/yeeee/English/words.txt' 

# label2word = {}
# with open(word_file, 'r') as f_word:
    # for line in f_word:
        # line = line[:-1].split()
        # word = line[0]
        # label = line[1]
        # label2word[label] = word.lower()
idx2label = {}
with open(dic_file, 'r') as f_dic:
    f_dic.readline()
    for line in f_dic:
        line = line[:-1].split()
        label = line[0]
        idx = line[1]
        idx2label[idx] = label

with open(embedding_file, 'r') as f_emb:
    with open(embedding_with_word_file, 'w') as f_emb_with_word:
        f_emb_with_word.write(str(len(idx2label))+' '+str(dim)+'\n')
        for i, line in enumerate(f_emb):
            try:
                f_emb_with_word.write(idx2label[str(i)]+' '+line)
            except:
                continue
