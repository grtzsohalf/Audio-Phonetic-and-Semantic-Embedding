import os

label_dir = '/home/grtzsohalf/Desktop/LibriSpeech/all_labels'
word_dir = '/home/grtzsohalf/Desktop/LibriSpeech/all_words'
if not os.path.exists(word_dir):
    os.mkdir(word_dir)

dic_file = '/home/grtzsohalf/Desktop/LibriSpeech/words.txt' 
label_files = os.listdir(label_dir)

idx2word = {}
with open(dic_file, 'r') as f_dic:
    for line in f_dic:
        line = line[:-1].split()
        word = line[0]
        idx = line[1]
        idx2word[idx] = word.lower()

for label_file in label_files:
    with open(os.path.join(label_dir, label_file), 'r') as f_label:
        with open(os.path.join(word_dir, 'word'+label_file[5:]), 'w') as f_word:
            for line in f_label:
                label = line[:-1]
                word = idx2word[label]
                f_word.write(word+'\n')
