import tensorflow as tf
import numpy as np
import time
import os
import random
from datetime import datetime 
from model_cos import AudioWord2Vec
from utils_cos import *
import operator

class Solver(object):
    def __init__(self, examples, labels, utters, batch_size, feat_dim, gram_num, memory_dim, 
                 init_lr, log_dir, model_dir, n_epochs, neg_sample_num, min_count, sampling_factor):
        # self.train_num = train_num
        self.example_file = examples
        self.label_file = labels
        self.utter_file = utters
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
        self.model = AudioWord2Vec(memory_dim, feat_dim, gram_num, neg_sample_num)

        self.generate_op = None
        self.discriminate_op = None
        
        self.n_feats = None
        self.feats = None
        self.feat_idx = None
        self.skip_feat_idx = None
        self.labels = None
        self.spk2idx = None
        self.idx2spk = None
        self.n_batches = None
        self.masks = None

    def generate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: generate_op              ###
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=momentum)
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = optimizer.minimize(loss, var_list=var_list)
        return train_op

    def discriminate_opt(self, loss, learning_rate, momentum, var_list):
        ### Optimizer building              ###
        ### variable: discriminate_op              ###
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=momentum)
        gvs = optimizer.compute_gradients(loss, var_list=var_list)
        capped_gvs = [(grad if grad is None else tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = optimizer.minimize(loss)
        return train_op

    def save_embedding(self, embedding_file, embedding_vectors, global_step, batch_labels):
        """Getting Embedding Vectors"""
        with open(embedding_file+'_'+str(global_step), 'a') as fout:
            for e, l in zip(embedding_vectors, batch_labels):
                fout.write(l + ' ')
                for i in e[:-1]:
                    fout.write(str(i) + ' ')
                fout.write(str(e[-1]) + '\n')

    def compute_train_loss(self, sess, summary_writer, summary_op, epoch,
                           loss, first_products, enc, neighbor_encs, target_encs):
        feat_order = list(range(self.n_feats))
        random.shuffle(feat_order)
        n_batches = self.n_batches
        feats = self.feats
        feat_idx = self.feat_idx
        skip_feat_idx = self.skip_feat_idx
        labels = self.labels
        spk2idx = self.spk2idx
        idx2spk = self.idx2spk
        masks = self.masks
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
            # batch_pos_feat, batch_neg_feats, batch_skip_feats, batch_labels, batch_masks \
                # = batch_pair_data(feats, feat_idx, skip_feat_idx, labels, spk2idx, idx2spk, masks, 
                                  # feat_indices, self.neg_sample_num)
            batch_pos_feat, batch_neg_feats, batch_skip_feats, batch_labels, batch_skip_labels, batch_neg_labels \
                = batch_pair_data(feats, feat_idx, skip_feat_idx, labels, spk2idx, idx2spk, masks, 
                                  feat_indices, self.neg_sample_num)
            # batch_neg_feats = batch_neg_feats.reshape((batch_size, 2*self.gram_num, self.neg_sample_num, self.feat_dim))
            batch_neg_feats = batch_neg_feats.reshape((-1, self.neg_sample_num, self.feat_dim))
            # batch_skip_feats = batch_skip_feats.reshape((batch_size, 2*self.gram_num, self.feat_dim))
            batch_skip_feats = batch_skip_feats.reshape((-1, self.feat_dim))
            # batch_masks = batch_masks.reshape((batch_size, 2*self.gram_num))
            batch_neg_labels = batch_neg_labels.reshape((-1, self.neg_sample_num))

            _, summary, loss_value, dot_value, enc_value, neighbor_encs_value, target_encs_value = \
                sess.run([self.generate_op, summary_op, loss, first_products, enc, neighbor_encs, target_encs], 
                         feed_dict={self.model.pos_feat: batch_pos_feat,
                                    self.model.neg_feats: batch_neg_feats,
                                    self.model.neighbors: batch_skip_feats}) # self.model.masks: batch_masks})
            total_loss_value += loss_value
            print_step = 1000
            if step % print_step == 0 and step != 0:
                duration = time.time() - start_time
                example_per_sec = batch_size / duration
                print ('pos label:')
                print (batch_labels[0])
                print ('skip label:')
                print (batch_skip_labels[0])
                print ('neg labels:')
                print (batch_neg_labels[0])
                print ('enc phonetic vector:')
                print (batch_pos_feat[0][:20])
                print ('enc embedding vector:')
                print (enc_value[0][:20])
                # print ('mask:')
                # print (batch_masks[0])
                print ('context phoetic vector:')
                print (batch_skip_feats[0][:20])
                print ('context context embedding vector:')
                print (neighbor_encs_value[0][:20])
                print ('neg phonetic vector:')
                print (batch_neg_feats[0][0][:20])
                print ('neg context embedding vector:')
                print (target_encs_value[0][0][:20])
                print ('cosine similarities:')
                print (dot_value[0])
                format_str = ('%s: epoch %d, part %d, step %d, loss=%.5f')
                print (format_str % (datetime.now(), epoch, part, step, total_loss_value/(print_step)))
                total_loss_value = 0.
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()

    def compute_test_loss(self, sess, summary_writer, summary_op, epoch, loss,
                     embedding_vectors, embedding_file, global_step):
        feat_order = list(range(self.n_feats))
        random.shuffle(feat_order)
        n_batches = self.n_batches
        feats = self.feats
        feat_idx = self.feat_idx
        skip_feat_idx = self.skip_feat_idx
        labels = self.labels
        spk2idx = self.spk2idx
        idx2spk = self.idx2spk
        masks = self.masks
        total_loss_value = 0.
        for step in range(n_batches+1):
            start_idx = step * self.batch_size
            end_idx = start_idx + self.batch_size
            feat_indices = feat_order[start_idx:end_idx]
            if step == n_batches:
                feat_indices = feat_order[step * self.batch_size:]
            batch_size = len(feat_indices)
            if batch_size == 0:
                continue
            # batch_pos_feat, batch_neg_feats, batch_skip_feats, batch_labels, batch_masks \
                # = batch_pair_data(feats, feat_idx, skip_feat_idx, labels, spk2idx, idx2spk, masks, 
                                  # feat_indices, self.neg_sample_num)
            batch_pos_feat, batch_neg_feats, batch_skip_feats, batch_labels \
                = batch_pair_data(feats, feat_idx, skip_feat_idx, labels, spk2idx, idx2spk, masks, 
                                  feat_indices, self.neg_sample_num)
            # batch_neg_feats = batch_neg_feats.reshape((batch_size, 2*self.gram_num, self.neg_sample_num, self.feat_dim))
            batch_neg_feats = batch_neg_feats.reshape((-1, self.neg_sample_num, self.feat_dim))
            # batch_skip_feats = batch_skip_feats.reshape((batch_size, 2*self.gram_num, self.feat_dim))
            batch_skip_feats = batch_skip_feats.reshape((-1, self.feat_dim))
            # batch_masks = batch_masks.reshape((batch_size, 2*self.gram_num))

            if summary_writer == None:
                loss_value, e_v = \
                    sess.run([loss, embedding_vectors], 
                             feed_dict={self.model.pos_feat: batch_pos_feat,
                                        self.model.neg_feats: batch_neg_feats,
                                        self.model.neighbors: batch_skip_feats}) # self.model.masks: batch_masks})
                # format_str = ('%s: step %d, sim_l=%.5f')
                # print (format_str % (datetime.now(), step, sim_l))
                self.save_embedding(embedding_file, e_v, global_step, batch_labels)
            # print ('pos: '+str(pos_l))
            total_loss_value += loss_value
        avg_loss = total_loss_value / n_batches
        if summary_writer != None:
            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()
        print ('%s: average loss for eval = %.5f' % (datetime.now(), avg_loss))

    def train(self):
        """ Training for Audio-WordVec."""
        ### Count words
        # label_dir = self.label_dir
        # label_files = os.listdir(label_dir)
        word_freq = Counter()
        total_count = 0
        # for label_file in label_files:
            # with open(os.path.join(label_dir, label_file), 'r') as f_l:
        with open(self.label_file, 'r') as f_l:
            for line in f_l:
                word = line[:-1]
                word_freq[word] += 1
                total_count += 1
        sorted_words = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)
        with open(self.label_file+'_word_count', 'w') as fout:
            for word in sorted_words:
                fout.write(word[0] + ', ' + str(word[1]) + '\n')
        print ('number of total words: '+ str(total_count))

        enc, loss, first_products, neighbor_encs, target_encs = self.model.build_model()
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
        
        self.generate_op = self.generate_opt(loss, self.init_lr, 0.9, g_vars)
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
        # example_files = os.listdir(self.example_dir)
        # label_files = os.listdir(self.label_dir)
        # utter_files = os.listdir(self.utter_dir)
        # print (example_files)
        # print (label_files)
        # print (utter_files)
        self.feats, self.labels, utters, spks \
            = load_data(self.example_file, self.label_file, self.utter_file)
        print ("Start batch training.")
        for epoch in range(self.n_epochs):
            e = epoch + global_step + 1
            print ("Start of Epoch: " + str(e) + "!")

            ### Load data  ###
            # tmp = list(zip(example_files, label_files, utter_files))
            # random.shuffle(tmp)
            # example_files, label_files, utter_files = zip(*tmp)
            count = 0
            # file_num = len(label_files)
            # for i, (example_file, label_file, utter_file) in enumerate(zip(example_files, label_files, utter_files)):
            self.n_feats, self.feat_idx, self.skip_feat_idx, \
                self.spk2idx, self.idx2spk, self.masks \
                = subsample_data(self.feats, self.labels, utters, spks, 
                                       word_freq, total_count, self.min_count, 
                                       self.gram_num, self.sampling_factor)
            self.n_batches = self.n_feats // self.batch_size
            # print ('Part: '+str(count))
            print ('# of batches: ' + str(self.n_batches))

            self.compute_train_loss(sess, summary_writer, summary_op_train, e,
                                    loss, first_products, enc, neighbor_encs, target_encs)
            # self.compute_test_loss(sess, summary_writer, summary_op_test, e, similarity_loss, None, None)
            count += 1

            ckpt = self.model_dir + '/model.ckpt'
            # saver.save(sess, ckpt, global_step=(e-1)*file_num+(i+1))
            saver.save(sess, ckpt, global_step=e)
            
            print ("End of Epoch: " + str(e) + "!")
        summary_writer.flush()

    def test(self, embedding_file):
        """ Testing for Audio-Word2Vec."""
        ### Count words
        # label_dir = self.label_dir
        # label_files = os.listdir(label_dir)
        word_freq = Counter()
        total_count = 0
        # for label_file in label_files:
            # with open(os.path.join(label_dir, label_file), 'r') as f_l:
        with open(self.label_file, 'r') as f_l:
            for line in f_l:
                word = line[:-1]
                word_freq[word] += 1
                total_count += 1
        # sorted_words = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)
        # with open(os.path.join(self.label_dir, '../word_count'), 'w') as fout:
            # for word in sorted_words:
                # fout.write(word[0] + ', ' + str(word[1]) + '\n')
        print ('number of total words: '+ str(total_count))

        enc, loss, first_products, neighbor_encs, target_encs = self.model.build_model()

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

        ### Load data  ###
        print ("Start testing.")
        # example_files = os.listdir(self.example_dir)
        # label_files = os.listdir(self.label_dir)
        # utter_files = os.listdir(self.utter_dir)
        # print (example_files)
        # print (label_files)
        # print (utter_files)
        self.feats, self.labels, utters, spks \
            = load_data(self.example_file, self.label_file, self.utter_file)
        count = 0
        # for example_file, label_file, utter_file in zip(example_files, label_files, utter_files):
        self.n_feats, self.feat_idx, self.skip_feat_idx, \
            self.spk2idx, self.idx2spk, self.masks \
            = subsample_data(self.feats, self.labels, utters, spks, 
                                   word_freq, total_count, self.min_count, 
                                   self.gram_num, self.sampling_factor)
        # self.n_feats, self.feats, self.feat_idx, self.skip_feat_idx, \
            # self.labels, self.spk2idx, self.idx2spk, self.masks \
            # = load_subsampled_data(self.example_file, 
                                   # self.label_file,
                                   # self.utter_file, 
                                   # word_freq, total_count, self.min_count, 
                                   # self.gram_num, self.sampling_factor)#, subsample=False)
        self.n_batches = self.n_feats // self.batch_size
        # print ('Part :'+str(count))
        print ('# of testing batches: ' + str(self.n_batches))
        count += 1

        ### Start testing ###
        self.compute_test_loss(sess, None, None, None, loss, enc, embedding_file, global_step)
