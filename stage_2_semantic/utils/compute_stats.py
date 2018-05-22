import os
import numpy as np
from tqdm import tqdm
from collections import deque
from multiprocessing import Pool

dim=256
data_type = 'all'
phonetic_dir = '/nfs/YueLao/grtzsohalf/yeeee/English/'+data_type+'_examples'
word_dir = '/nfs/YueLao/grtzsohalf/yeeee/English/'+data_type+'_words'
word_count = '/nfs/YueLao/grtzsohalf/yeeee/English/word_count_all'
mean_file = '/nfs/YueLao/grtzsohalf/yeeee/English/mean_'+data_type
std_file = '/nfs/YueLao/grtzsohalf/yeeee/English/std_'+data_type
diff_file = '/nfs/YueLao/grtzsohalf/yeeee/English/diff_'+data_type

words = []
count = 0
with open(word_count, 'r') as f_count:
    for line in f_count:
        if count == 23406:
            break
        word = line[:-1].split(',')[0]
        words.append(word)
        count += 1


def read_file(phonetic_file, word_file):
    embs = {}
    counts = {}
    with open(os.path.join(phonetic_dir, phonetic_file), 'r') as f_pho:
        with open(os.path.join(word_dir, word_file), 'r') as f_word:
            for emb, word in tqdm(zip(f_pho, f_word)):
                emb = emb[:-1].split()
                word = word[:-1]
                emb = np.array(list(map(float, emb)))
                if not word in words:
                    continue
                if not word in embs:
                    embs[word] = [emb]
                    counts[word] = 1
                else:
                    embs[word].append(emb)
                    counts[word] += 1
    return embs, counts


p = Pool(6)
embs = {}
counts = {}
for emb_f, count_f in  p.starmap(read_file, list(zip(os.listdir(phonetic_dir), os.listdir(word_dir)))):
    for word in emb_f:
        if word in embs:
            embs[word] += emb_f[word]
            counts[word] += count_f[word]
        else:
            embs[word] = emb_f[word]
            counts[word] = count_f[word]

# for phonetic_file, word_file in zip(os.listdir(phonetic_dir), os.listdir(word_dir)):
    # with open(os.path.join(phonetic_dir, phonetic_file), 'r') as f_pho:
        # with open(os.path.join(word_dir, word_file), 'r') as f_word:
            # for emb, word in tqdm(zip(f_pho, f_word)):
                # emb = emb[:-1].split()
                # word = word[:-1]
                # emb = np.array(list(map(float, emb)))
                # if not word in words:
                    # continue
                # if not word in embs:
                    # embs[word] = [emb]
                    # counts[word] = 1
                # else:
                    # embs[word].append(emb)
                    # counts[word] += 1

means = {}
stds = {}
print ('Start computing means and stds...')
for word in embs:
    means[word] = np.mean(embs[word], axis=0)
    stds[word] = np.linalg.norm(np.std(embs[word], axis=0))
print ('End computing means and stds.')

with open(mean_file, 'w') as f_mean:
    for word in means:
        f_mean.write(word+' '+' '.join(means[word].astype(str))+'\n')
with open(std_file, 'w') as f_std:
    for word in stds:
        f_std.write(word+' '+str(stds[word])+'\n')


def rmse(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).mean())

mean_list = list(means.keys())

mean_diff = {}
for word in means:
    mean_diff[word] = deque()
print ('Start computing differences between means...')
for i, word1 in tqdm(enumerate(mean_list)):
    for word2 in mean_list[i+1:]:
        diff = rmse(means[word1], means[word2])
        if len(mean_diff[word1] ) < 3:
            mean_diff[word1].append(diff)
            sorted(mean_diff[word1])
        else:
            if diff < mean_diff[word1][-1]:
                mean_diff[word1][-1] = diff
                sorted(mean_diff[word1])
        if len(mean_diff[word2] ) < 3:
            mean_diff[word2].append(diff)
            sorted(mean_diff[word2])
        else:
            if diff < mean_diff[word2][-1]:
                mean_diff[word2][-1] = diff
                sorted(mean_diff[word2])


with open(diff_file, 'w') as f_diff:
    for word in mean_diff:
        f_diff.write(word+' '+' '.join(mean_diff[word].astype(str))+'\n')
