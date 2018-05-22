import os
import numpy as np
import operator
from collections import Counter

epoch = 50

feat_dir = "/nfs/Caishen/grtzsohalf/yeeee/English"
prons_file = os.path.join(feat_dir, "test_prons")
embedding_dir = os.path.join(feat_dir, "embedding/0.0005_256")
idx_file = os.path.join(embedding_dir, "embedding_idx_"+str(epoch))
mapped_prons_file = os.path.join(embedding_dir, "mapped_prons2idx_"+str(epoch))
filtered_idx_file = os.path.join(embedding_dir, "filtered_embedding_idx_"+str(epoch))
filtered_mapped_prons_file = os.path.join(embedding_dir, "filtered_mapped_prons2idx_"+str(epoch))

V_size = 25146
filtered_size = 100

def load_idx(filename):
    dic = Counter()
    count = 0
    with open(filename, 'r') as fin:
        for line in fin:
            dic[line[:-1]] += 1
            count += 1
    return dic

def load_prons(filename):
    dic = Counter()
    count = 0
    with open(filename, 'r') as fin:
        for line in fin:
            line = line.split()
            dic[line[3]] += 1
            count += 1
    return dic

def write_mapped_prons(prons_file, mapped_prons_file, prons2idx):
    with open(prons_file, 'r') as fin:
        with open(mapped_prons_file, 'w') as fout:
            for line in fin:
                line = line.split()
                if line[3] in prons2idx:
                    fout.write(prons2idx[line[3]] + str('\n'))
                else:
                    fout.write('X\n')

def write_filtered_mapped_prons(prons_file, filtered_mapped_prons_file, filtered_prons2idx):
    with open(prons_file, 'r') as fin:
        with open(filtered_mapped_prons_file, 'w') as fout:
            for line in fin:
                line = line.split()
                if line[3] in filtered_prons2idx:
                    fout.write(filtered_prons2idx[line[3]] + str('\n'))
                else:
                    fout.write('X\n')

def write_filtered_idx(idx_file, filtered_mapped_prons_file, filtered_idx_file):
    with open(idx_file, 'r') as fin1:
        with open(filtered_mapped_prons_file, 'r') as fin2:
            with open(filtered_idx_file, 'w') as fout:
                for line1, line2 in zip(fin1, fin2):
                    line1 = line1[:-1]
                    line2 = line2[:-1]
                    if line2 != 'X':
                        fout.write(line1 + str('\n'))
                    else:
                        fout.write('X\n')

def main():
    idx_dic = load_idx(idx_file)
    prons_dic = load_prons(prons_file)
    sorted_idx = sorted(idx_dic.items(), key=operator.itemgetter(1), reverse=True)
    print (sorted_idx[:10])
    sorted_prons = sorted(prons_dic.items(), key=operator.itemgetter(1), reverse=True)
    print (sorted_prons[:10])
    idx2prons = {}
    for i, idx in enumerate(sorted_idx):
        idx2prons[idx[0]] = sorted_prons[i][0]
    print ('# of idx2prons: ' + str(len(idx2prons)))
    prons2idx = {p:i for i,p in idx2prons.items()}
    print ('# of prons2idx: ' + str(len(prons2idx)))
    write_mapped_prons(prons_file, mapped_prons_file, prons2idx)

    filtered_idx = sorted_idx[:filtered_size]
    filtered_prons = sorted_prons[:filtered_size]
    filtered_idx2prons = {}
    for i, idx in enumerate(filtered_idx):
        filtered_idx2prons[idx[0]] = filtered_prons[i][0]
    filtered_prons2idx = {p:i for i,p in filtered_idx2prons.items()}
    write_filtered_mapped_prons(prons_file, filtered_mapped_prons_file, filtered_prons2idx)
    write_filtered_idx(idx_file, filtered_mapped_prons_file, filtered_idx_file)

if __name__=='__main__':
    main()


