import os

word_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/all_words'
seq_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/all_seqs'

if not os.path.exists(seq_dir):
    os.mkdir(seq_dir)

dic_file = '/nfs/Caishen/grtzsohalf/yeeee/English/seq_of_phone_word'

dic = {}
with open(dic_file, 'r') as f_dic:
    for line in f_dic:
        line = line[:-1].split()
        word = line[0]
        seq = line[1:]
        dic[word] = seq

for word_file in os.listdir(word_dir):
    with open(os.path.join(word_dir, word_file), 'r') as f_word:
        with open(os.path.join(seq_dir, word_file), 'w') as f_seq:
            for word in f_word:
                word = word[:-1]
                f_seq.write(' '.join(dic[word])+'\n')
