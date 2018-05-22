import os
import numpy as np
from tqdm import tqdm
from collections import deque
from multiprocessing import Pool

dim=400
data_type = 'original'
phonetic_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_examples'
# word_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_words'
# utter_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_utters'
normed_phonetic_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_examples_norm'
# normed_word_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_words_norm'
# normed_utter_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/'+data_type+'_utters_norm'
if not os.path.exists(normed_phonetic_dir):
    os.mkdir(normed_phonetic_dir)
# if not os.path.exists(normed_word_dir):
    # os.mkdir(normed_word_dir)
# if not os.path.exists(normed_utter_dir):
    # os.mkdir(normed_utter_dir)

# word_files = os.listdir(word_dir)
# utter_files = os.listdir(utter_dir)
# for i in range(len(word_files)):
    # with open(os.path.join(word_dir, word_files[i]), 'r') as f_in_word, \
            # open(os.path.join(utter_dir, utter_files[i]), 'r') as f_in_utter:
        # with open(os.path.join(normed_word_dir, word_files[i]), 'w') as f_word, \
                # open(os.path.join(normed_utter_dir, utter_files[i]), 'w') as f_utter:
            # for line in f_in_word:
                # f_word.write(line)
            # for line in f_in_utter:
                # f_utter.write(line)

def read_file(phonetic_file):
    embs = []
    count = 0
    with open(os.path.join(phonetic_dir, phonetic_file), 'r') as f_pho:
        for emb in tqdm(f_pho):
            emb = emb[:-1].split()
            emb = np.array(list(map(float, emb)))
            embs.append(emb)
            count += 1
    return embs, count


p = Pool(3)
embs = []
count = [0]
# for emb_f, count_f in p.map(read_file, os.listdir(phonetic_dir)):
    # embs += emb_f
    # count.append(count[-1]+count_f)
for phonetic_file in os.listdir(phonetic_dir):
    emb_f, count_f = read_file(phonetic_file)
    embs += emb_f
    count.append(count[-1]+count_f)

mean = None
std = None
print ('Start computing means and stds...')
mean = np.mean(embs, axis=0)
std = np.std(embs, axis=0)
print ('End computing means and stds.')
# embs = (embs - mean) / std
embs -= mean
embs /= std

phonetic_files = os.listdir(phonetic_dir)
for i in range(len(count)-1):
    with open(os.path.join(normed_phonetic_dir, phonetic_files[i]), 'w') as f_norm:
        for emb in tqdm(embs[count[i]:count[i+1]]):
            f_norm.write(' '.join(map(str, emb)))
            f_norm.write('\n')
