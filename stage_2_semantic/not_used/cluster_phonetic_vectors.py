from sklearn.cluster import MiniBatchKMeans as KMeans
import os

feat_dir = "/home_local/grtzsohalf/yeeee"
phonetic_file = os.path.join(feat_dir, "phonetic_train")
centroid_file =  os.path.join(feat_dir, "phonetic_train_centroids")

V_size = 25467

def load_data(data_file):
    phonetic_vectors = []
    count = 0
    with open(data_file, 'r') as fin:
        for line in fin:
            line = line.split()[:-1]
            phonetic_vectors.append(list(map(float, line)))
            count += 1
    print ('# of phonetic vectors: ' + str(count))
    return phonetic_vectors

def write_data(data_file, centroids):
    with open(data_file, 'w') as fout:
        for c in centroids:
            for i in c[:-1]:
                fout.write(str(i) + ' ')
            fout.write(str(c[-1]) + '\n')

def main():
    phonetic_vectors = load_data(phonetic_file)
    clf = KMeans(n_clusters=V_size, verbose=True)
    clusters = clf.fit(phonetic_vectors)
    centroids = clf.cluster_centers_
    print (centroids.shape)
    write_data(centroid_file, centroids)

if __name__=='__main__':
    main()
