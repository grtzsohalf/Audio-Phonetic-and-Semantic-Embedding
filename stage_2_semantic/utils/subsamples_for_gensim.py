import os
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from tqdm import tqdm

min_count=5
dim = 128

word_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/all_words'
utter_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/all_utters'
sentence_file = '/nfs/Caishen/grtzsohalf/yeeee/English/sentence'
word2vec_file =  '/nfs/Caishen/grtzsohalf/yeeee/English/subsamples_for_gensim_iter5_min'+str(min_count)

sentences = []
print ('Load words...')
count = 0
last_utter = None
sentence = []
for word_file, utter_file in zip(os.listdir(word_dir), os.listdir(utter_dir)):
    print (word_file)
    print (utter_file)
    with open(os.path.join(word_dir, word_file), 'r') as f_word:
        with open(os.path.join(utter_dir, utter_file), 'r') as f_utter:
            for word, utter in tqdm(zip(f_word, f_utter)):
                word = word[:-1]
                utter = utter[:-1]
                if utter != last_utter:
                    last_utter = utter
                    sentences.append(list(sentence))
                    sentence = []
                sentence.append(word)
                count += 1
print (count)

with open(sentence_file, 'w') as f_sentence:
    for sentence in sentences:
        f_sentence.write(' '.join(sentence)+'\n')

sentences = LineSentence(sentence_file)
model = Word2Vec(sentences=sentences, size=dim, min_count=min_count, window=5, sg=1, sample=0.001, negative=5, iter=5)
# model = Word2Vec.load('/nfs/Caishen/grtzsohalf/yeeee/English/Word2Vec_model')
print ('Write Word2Vec...')
model.save('/nfs/Caishen/grtzsohalf/yeeee/English/gensim_model')
model.wv.save_word2vec_format(word2vec_file, binary=False)

# with open(word2vec_file, 'r') as f_in:
    # with open(word2vec_file[:-4], 'w') as f_out:
        # f_in.readline()
        # word = None
        # vec = None
        # f_out.write(str(num_of_vectors)+' '+str(dim)+'\n')
        # for i, line in enumerate(f_in):
            # if i % 2 == 0: 
                # word = line[:-1]
            # else:
                # vec = line[:-1]
                # f_out.write(word+vec+'\n')
