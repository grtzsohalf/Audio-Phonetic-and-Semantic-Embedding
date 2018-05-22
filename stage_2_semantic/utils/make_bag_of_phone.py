import os

word_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/all_words'
bag_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/all_bags'

if not os.path.exists(bag_dir):
    os.mkdir(bag_dir)

dic_file = '/nfs/Caishen/grtzsohalf/yeeee/English/bag_of_phone_word'

dic = {}
with open(dic_file, 'r') as f_dic:
    for line in f_dic:
        line = line[:-1].split()
        word = line[0]
        bag = line[1:]
        dic[word] = bag

for word_file in os.listdir(word_dir):
    with open(os.path.join(word_dir, word_file), 'r') as f_word:
        with open(os.path.join(bag_dir, word_file), 'w') as f_bag:
            for word in f_word:
                word = word[:-1]
                f_bag.write(' '.join(dic[word])+'\n')
