import tensorflow as tf
import numpy as np
import time
import os
import random
from datetime import datetime 
from model_text import AudioWord2Vec
from utils_text import *

next_random = 1

class Solver(object):
    def __init__(self, dic_file, labels, utters, batch_size, feat_dim, gram_num, memory_dim, 
                 init_lr, log_dir, model_dir, n_epochs, neg_sample_num, min_count, sampling_factor):
        self.dic_file = dic_file
        self.label_dir = labels
        self.utter_dir = utters
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.gram_num = gram_num
        self.memory_dim = memory_dim
        self.init_lr = init_lr
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.n_epochs = n_epochs
        self.neg_sample_num = neg_sample_num
        self.min_count = min_count
        self.sampling_factor = sampling_factor
        # self.model = AudioWord2Vec(memory_dim, feat_dim, gram_num, neg_sample_num)
        self.model = None

        self.generate_op = None
        self.discriminate_op = None
        
        self.n_feats = None
        self.feats = None
        self.skip_feats = None
        # self.labels = None
        # self.spk2idx = None
        # self.idx2spk = None
        self.n_batches = None
        self.dic = None
        self.idx2word = None

    def generate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: generate_op              ###
        
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=momentum)
        # gvs = optimizer.compute_gradients(loss, var_list=var_list)
        # capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs)

        train_op = optimizer.minimize(loss, var_list=var_list)
        return train_op

    def discriminate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: discriminate_op              ###
        
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=momentum)
        # gvs = optimizer.compute_gradients(loss, var_list=var_list)
        # capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs)

        train_op = optimizer.minimize(loss)
        return train_op

    def save_embedding(self, embedding_file, e_w, e_c, global_step):
        """Getting Embedding Vectors"""
        with open(embedding_file+'_word_'+str(global_step), 'w') as fout:
            for e in e_w:
                for i in e[:-1]:
                    fout.write(str(i) + ' ')
                fout.write(str(e[-1]) + '\n')
        with open(embedding_file+'_context_'+str(global_step), 'w') as fout:
            for e in e_c:
                for i in e[:-1]:
                    fout.write(str(i) + ' ')
                fout.write(str(e[-1]) + '\n')

    def compute_train_loss(self, sess, summary_writer, summary_op, epoch, loss, 
                           first_products, enc, neighbor_encs, negs_word):
        feat_order = list(range(self.n_feats))
        random.shuffle(feat_order)
        n_batches = self.n_batches
        feats = self.feats
        skip_feats = self.skip_feats
        dic = self.dic
        total_loss_value = 0.
        for step in range(n_batches+1):
            start_time = time.time()
            start_idx = step * self.batch_size
            end_idx = start_idx + self.batch_size
            feat_indices = feat_order[start_idx:end_idx]
            if step == n_batches:
                feat_indices = feat_order[step * self.batch_size:]
            batch_size = len(feat_indices)
            if batch_size == 0:
                continue
            batch_pos_feat, batch_neg_feats, batch_skip_feats \
                = batch_pair_data(feats, skip_feats, feat_indices, self.neg_sample_num)
            batch_skip_feats = batch_skip_feats.reshape((-1, 2*self.gram_num))
            batch_neg_feats = batch_neg_feats.reshape((-1, self.neg_sample_num))
            batch_masks = batch_skip_feats.astype(float)
            for i in range(len(batch_masks)):
                for j in range(len(batch_masks[0])):
                    if batch_masks[i][j] != 0.:
                        batch_masks[i][j] = 1.

            _, summary, loss_value, dot_value, enc_value, neighbor_encs_value, negs_word_value = \
                sess.run([self.generate_op, summary_op, loss, first_products, enc, neighbor_encs, negs_word], 
                         feed_dict={self.model.pos_feat: batch_pos_feat,
                                    self.model.neg_feats: batch_neg_feats,
                                    self.model.neighbors: batch_skip_feats, 
                                    self.model.masks: batch_masks})
            total_loss_value += loss_value
            print_step = 1000
            if step % print_step == 0 and step != 0:
                duration = time.time() - start_time
                example_per_sec = batch_size / duration
                print ('trained word:')
                print (self.idx2word[batch_pos_feat[0]])
                print ('mask:')
                print (batch_masks[0])
                print ('context:')
                print ([self.idx2word[idx] for idx in batch_skip_feats[0]])
                print ('neg sample:')
                print ([self.idx2word[idx] for idx in batch_neg_feats[0]])
                print ('word vector:')
                print (enc_value[0][:20])
                print ('context vector:')
                print (neighbor_encs_value[0][0][:20])
                print ('neg vector:')
                print (negs_word_value[0][0][:20])
                # print ('neg context vector:')
                # print (negs_context_value[0][0][:20])
                print ('dot products:')
                print (dot_value[0][:2*self.gram_num])
                print (dot_value[0][2*self.gram_num:])
                # print ('softmax:')
                # print (softmax_value[:20, 0])
                format_str = ('%s: epoch %d, step %d, loss=%.5f')
                print (format_str % (datetime.now(), epoch, step, total_loss_value/print_step))
                total_loss_value = 0.
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()

    def subsample(self, feats, skip_feats, word_freq):
        sampling = self.sampling_factor
        def subsampling_bool(freq):
            # global next_random
            # next_random = (next_random * 25214903917 + 11) & 0xFFFF
            prob = (math.sqrt(sampling/freq)+sampling/freq)
            return (math.sqrt(sampling/freq)+sampling/freq) - np.random.rand()  #next_random/65536.0 

        subsample_bool = []
        for feat in feats:
            subsample_bool.append(subsampling_bool(word_freq[feat]) > 0.)
        subsample_bool = np.array(subsample_bool)
        # print (subsample_bool[:100])
        sub_feats = feats[subsample_bool==True]
        sub_skip_feats = skip_feats[subsample_bool==True, :]
        return len(sub_feats), sub_feats, sub_skip_feats

    def train(self):
        """ Training for Audio-WordVec."""
        ### Load data  ###
        label_files = os.listdir(self.label_dir)
        print (label_files)
        n_feats, feats, skip_feats, self.dic, self.word_freq \
            = load_data(self.gram_num, self.label_dir, self.utter_dir, self.dic_file, self.min_count)
        print ('number of feats: '+str(n_feats))
        print ('dic size: '+str(len(self.dic)))
        self.idx2word = {i: w for w, i in self.dic.items()}

        self.model = AudioWord2Vec(self.memory_dim, len(self.dic), self.gram_num, self.neg_sample_num)
        loss, first_products, enc, neighbor_encs, negs_word = self.model.build_model()
        # enc_test, similarity_loss_test = self.model.build_test()

        # Variables
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'generator' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        # print ("G_VAR:")
        # for v in g_vars:
            # print (v)
        # print ("D_VAR:")
        # for v in d_vars:
            # print (v)
        
        self.generate_op = self.generate_opt(loss, self.init_lr, 0.9, t_vars)
        # self.generate_op = self.generate_opt(similarity_loss-discrimination_loss, self.init_lr, 0.9, g_vars)
        # self.discriminate_op = self.discriminate_opt(discrimination_loss+10*GP_loss, self.init_lr, 0.9, d_vars)

        # Build and initialization operation to run below
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

        summary_train = [tf.summary.scalar("loss", loss)]
                        # tf.summary.scalar("discrimination loss", discrimination_loss),
                        # tf.summary.scalar("GP loss", GP_loss)]
        # summary_test = [tf.summary.scalar("similarity loss eval", similarity_loss),
                        # tf.summary.scalar("discrimination loss eval", discrimination_loss),
                        # tf.summary.scalar("GP loss eval", GP_loss)]
        summary_op_train = tf.summary.merge(summary_train)
        # summary_op_test = tf.summary.merge(summary_test)

        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        ### Restore the model ###
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        global_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print ("Model restored.")
        else:
            print ('No checkpoint file found.')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ### Start training ###
        print ("Start batch training.")
        for epoch in range(self.n_epochs):
            e = epoch + global_step + 1
            print ("Start of Epoch: " + str(e) + "!")

            self.n_feats, self.feats, self.skip_feats \
                = self.subsample(feats, skip_feats, self.word_freq)
            print ('number of subsampled feats: '+str(self.n_feats))
            print ('dic size: '+str(len(self.dic)))
            self.n_batches = self.n_feats // self.batch_size
            print ('# of batches: ' + str(self.n_batches))

            self.compute_train_loss(sess, summary_writer, summary_op_train, e, 
                                    loss, first_products, enc, neighbor_encs, negs_word)
            # self.compute_test_loss(sess, summary_writer, summary_op_test, e, similarity_loss, None, None)

            ckpt = self.model_dir + '/model.ckpt'
            saver.save(sess, ckpt, global_step=e)
            print ("End of Epoch: " + str(e) + "!")
        summary_writer.flush()

    def test(self, embedding_file):
        """ Testing for Audio-Word2Vec."""
        self.dic = {}
        with open(self.dic_file, 'r') as f_dic:
            f_dic.readline()
            for line in f_dic:
                line = line[:-1].split()
                self.dic[line[0]] = line[1]
        self.model = AudioWord2Vec(self.memory_dim, len(self.dic), self.gram_num, self.neg_sample_num)
        word_embedding, context_embedding = self.model.build_test()

        # Build and initialization operation to run below
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

        ### Restore the model ###
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        global_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print ("Model restored.")
        else:
            print ('No checkpoint file found.')
            exit()

        e_w, e_c = sess.run([word_embedding, context_embedding], 
                       feed_dict={self.model.pos_feat: np.ones((self.batch_size)),
                                  self.model.neg_feats: np.ones((self.batch_size, self.neg_sample_num)),
                                  self.model.neighbors: np.ones((self.batch_size, 2*self.gram_num))})
        self.save_embedding(embedding_file, e_w, e_c, global_step)
