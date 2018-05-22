import os
import numpy as np
from collections import Counter
from collections import deque
from tqdm import tqdm

gram_num = 5
min_count = 5
dim=128
data_type = '2_layer_sigmoid/0.0001_128'
embedding_file = '/home/grtzsohalf/Desktop/LibriSpeech/embedding_all/'+data_type+'/train_embeddings_20'
embedding_with_word_file = '/home/grtzsohalf/Desktop/LibriSpeech/embedding_all/'+data_type+'/train_embeddings_20_with_words'


embs = {}
counts = Counter()
total_count = 0
with open(embedding_file, 'r') as f_emb:
    for line in tqdm(f_emb):
        line = line[:-1].split()
        word = line[0]
        try:
            emb = np.array(list(map(float, line[1:])))
        except:
            print (total_count)
            print (line)
        if not word in embs:
            embs[word] = emb
            counts[word] += 1
        else:
            embs[word] += emb
            counts[word] += 1
        total_count += 1
print ('Total #: '+str(total_count))

with open(embedding_with_word_file, 'w') as f_out:
    f_out.write(str(len(embs))+' '+str(dim)+'\n')
    for word in embs:
        emb = embs[word] / counts[word]
        f_out.write(word+' '+' '.join(emb.astype(str))+'\n')

