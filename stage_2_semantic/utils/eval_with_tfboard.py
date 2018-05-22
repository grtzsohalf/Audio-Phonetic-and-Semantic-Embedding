import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

emb_file = '/nfs/Caishen/grtzsohalf/yeeee/English/embedding/bag_2_layer_sigmoid/0.001_128/train_freq_embeddings_5_with_words'
log_dir = '/nfs/Caishen/grtzsohalf/yeeee/English/embedding/bag_2_layer_sigmoid/0.001_128/log_train_freq_embeddings_5'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

embedding = []
words = []
with open(emb_file, 'r') as f_emb:
    f_emb.readline()
    for line in f_emb:
        line = line[:-1].split()
        word = line[0]
        feat = line[1:]
        feat = list(map(float, feat))
        embedding.append(feat)
        words.append(word)
embedding = np.array(embedding)

# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
   for word in words:
        f.write(word + '\n')

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save model
saver = tf.train.Saver()
saver.save(sess, os.path.join(log_dir, "model.ckpt"))
