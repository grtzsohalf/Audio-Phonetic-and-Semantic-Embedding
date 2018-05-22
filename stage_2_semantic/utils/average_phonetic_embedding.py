import os
import numpy as np
from tqdm import tqdm

min_count = 5
dim = 256
data_type = 'all'
phonetic_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_examples_norm'
word_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_words'
word_count = '/nfs/Caishen/grtzsohalf/yeeee/English/word_count_all'
average_file = '/nfs/Caishen/grtzsohalf/yeeee/English/phonetic_embedding_'+data_type

words = []
count = 0
with open(word_count, 'r') as f_count:
    for line in f_count:
        line = line[:-1].split(',')
        word = line[0]
        num = line[1]
        if int(num) < min_count:
            break
        words.append(word)
        count += 1
print (count)

embs = {}
counts = {}
for phonetic_file, word_file in zip(os.listdir(phonetic_dir), os.listdir(word_dir)):
    with open(os.path.join(phonetic_dir, phonetic_file), 'r') as f_pho:
        with open(os.path.join(word_dir, word_file), 'r') as f_word:
            for emb, word in tqdm(zip(f_pho, f_word)):
                emb = emb[:-1].split()
                word = word[:-1]
                emb = np.array(list(map(float, emb)))
                if not word in words:
                    continue
                if not word in embs:
                    embs[word] = emb
                    counts[word] = 1
                else:
                    embs[word] += emb
                    counts[word] += 1
with open(average_file, 'w') as f_vec:
    f_vec.write(str(len(embs))+' '+str(dim)+'\n')
    for word in embs:
        emb = embs[word] / counts[word]
        f_vec.write(word+' '+' '.join(emb.astype(str))+'\n')
