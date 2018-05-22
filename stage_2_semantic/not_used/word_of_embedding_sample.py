import os
import numpy as np
from collections import Counter
from collections import deque

gram_num = 4
min_count = 60 
dim=256
data_type = '6_layer_sigmoid/0.0001_256'
embedding_file = '/nfs/YueLao/grtzsohalf/yeeee/English/embedding/'+data_type+'/train_embeddings_17'
embedding_with_word_file = '/nfs/YueLao/grtzsohalf/yeeee/English/embedding/'+data_type+'/train_embeddings_17_with_words'

# word_dir = '/nfs/YueLao/grtzsohalf/yeeee/English/all_words'
# word_files = os.listdir(word_dir)

# word_freq = Counter()
# for word_file in word_files:
    # with open(os.path.join(word_dir, word_file), 'r') as f_w:
        # for line in f_w:
            # word = line[:-1]
            # word_freq[word] += 1

# embs = {}
# counts = {}
# total_count = 0
# with open(embedding_file, 'r') as f_emb:
    # for word_file in word_files:
        # print (word_file)
        # words = []
        # with open(os.path.join(word_dir, word_file), 'r') as f_word:
            # for word in f_word:
                # words.append(word[:-1])
        # que = deque()
        # for word in words:
            # que.append(word)
            # if len(que) < 2*gram_num+1:
                # continue
            # continue_bool = False
            # for qq in que:
                # if word_freq[qq] < min_count:
                    # continue_bool = True
                    # break
            # if continue_bool == True:
                # que.popleft()
                # continue
            # emb = f_emb.readline()
            # emb = np.array(list(map(float, emb[:-1].split())))
            # ww = que[gram_num]
            # if not ww in embs:
                # embs[ww] = emb
                # counts[ww] = 1
            # else:
                # embs[ww] += emb
                # counts[ww] += 1
            # que.popleft()
            # total_count += 1
# print ('# of feats after: '+str(total_count))
# print ('# of different words after: '+str(len(embs)))


embs = {}
counts = Counter()
total_count = 0
with open(embedding_file, 'r') as f_emb:
    for line in f_emb:
        line = line[:-1].split()
        word = line[0]
        emb = np.array(list(map(float, line[1:])))
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

