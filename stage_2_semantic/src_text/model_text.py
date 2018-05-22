import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes 
import copy
        
class AudioWord2Vec(object):
    def __init__(self, memory_dim, feat_dim, gram_num, neg_sample_num):
        self.memory_dim = memory_dim
        self.feat_dim = feat_dim
        self.gram_num = gram_num
        self.neg_sample_num = neg_sample_num

        self.pos_feat = tf.placeholder(tf.int64, [None])
        self.neighbors = tf.placeholder(tf.int64, [None, 2*self.gram_num])
        self.neg_feats = tf.placeholder(tf.int64, [None, self.neg_sample_num])
        self.masks = tf.placeholder(tf.float32, [None, 2*self.gram_num])
        with tf.device('/cpu:0'):
            with tf.name_scope('word_embedding'):
                self.word_embedding = tf.Variable(tf.random_uniform([feat_dim, memory_dim], -1.0, 1.0))
            with tf.name_scope('context_embedding'):
                self.context_embedding = tf.Variable(tf.random_uniform([feat_dim, memory_dim], -1.0, 1.0))
                # self.dot_bias = tf.Variable(tf.random_uniform([feat_dim], -1.0, 1.0))

    def leaky_relu(self, x, alpha=0.01):
        return tf.maximum(x, alpha*x)

    def cosine_similarity(self, x, y):
        norm_x = tf.nn.l2_normalize(x, -1)
        norm_y = tf.nn.l2_normalize(y, -1)
        return tf.reduce_sum(norm_x*norm_y, -1)

    def skip_encode(self, neighbor, reuse=False):
        with tf.variable_scope('skip_encode', reuse=reuse) as scope_skip_enc:
            W_enc_1 = tf.get_variable("skip_enc_w_1", [self.memory_dim, self.memory_dim])
            b_enc_1 = tf.get_variable("skip_enc_b_1", shape=[self.memory_dim])
            # W_enc_2 = tf.get_variable("skip_enc_w_2", [self.memory_dim, self.memory_dim])
            # b_enc_2 = tf.get_variable("skip_enc_b_2", shape=[self.memory_dim])
            # W_enc_3 = tf.get_variable("skip_enc_w_3", [self.memory_dim, self.memory_dim])
            # b_enc_3 = tf.get_variable("skip_enc_b_3", shape=[self.memory_dim])
            # W_enc_4 = tf.get_variable("skip_enc_w_4", [self.memory_dim, self.memory_dim])
            # b_enc_4 = tf.get_variable("skip_enc_b_4", shape=[self.memory_dim])

            # neighbor = tf.contrib.layers.batch_norm(neighbor)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_2) + b_enc_2)
            enc = tf.matmul(neighbor, W_enc_1) + b_enc_1
            # enc = self.leaky_relu(tf.matmul(neighbor, W_enc_1) + b_enc_1)
            # enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_2) + b_enc_2)
            # enc = tf.matmul(enc, W_enc_2) + b_enc_2
            # enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_3) + b_enc_3)
            # enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_4) + b_enc_4)
            # return neighbor
            return enc

    def encode(self, feat, reuse=False):
        with tf.variable_scope('encode', reuse=reuse) as scope_enc:
            W_enc_1 = tf.get_variable("enc_w_1", [self.memory_dim, self.memory_dim])
            b_enc_1 = tf.get_variable("enc_b_1", shape=[self.memory_dim])
            # W_enc_2 = tf.get_variable("enc_w_2", [self.memory_dim, self.memory_dim])
            # b_enc_2 = tf.get_variable("enc_b_2", shape=[self.memory_dim])
            # W_enc_3 = tf.get_variable("enc_w_3", [self.memory_dim, self.memory_dim])
            # b_enc_3 = tf.get_variable("enc_b_3", shape=[self.memory_dim])
            # W_enc_4 = tf.get_variable("enc_w_4", [self.memory_dim, self.memory_dim])
            # b_enc_4 = tf.get_variable("enc_b_4", shape=[self.memory_dim])

            # feat = tf.contrib.layers.batch_norm(feat)
            # enc = self.leaky_relu(tf.matmul(feat, W_enc_1) + b_enc_1)
            enc = tf.matmul(feat, W_enc_1) + b_enc_1
            # enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_2) + b_enc_2)
            # enc = tf.matmul(enc, W_enc_2) + b_enc_2
            # enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_3) + b_enc_3)
            # enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_4) + b_enc_4)
            # return feat
            return enc

    '''
    def adversarial_training(self, pos_pair, neg_pair):
        W_adv_1 = tf.get_variable("adv_w_1", [(2*self.gram_num+1)*self.feat_dim, self.feat_dim])
        b_adv_1 = tf.get_variable("adv_b_1", shape=[self.feat_dim])
        W_adv_2 = tf.get_variable("adv_w_2", [self.feat_dim, self.feat_dim])
        b_adv_2 = tf.get_variable("adv_b_2", shape=[self.feat_dim])
        W_bin = tf.get_variable("bin_w", [self.feat_dim, 1])
        b_bin = tf.get_variable("bin_b", shape=[1])

        # WGAN gradient penalty
        with tf.variable_scope('gradient_penalty') as scope_2_1:
            alpha = tf.random_uniform(shape=[self.batch_size, (2*self.gram_num+1)*self.feat_dim], minval=0., maxval=1.)
            pos_pair_stop = tf.stop_gradient(pos_pair)
            neg_pair_stop = tf.stop_gradient(neg_pair)
            pair_hat = alpha * pos_pair_stop + (1 - alpha) * neg_pair_stop
            # pair_hat_norm = tf.contrib.layers.layer_norm(pair_hat)
            pair_hat_l1 = self.leaky_relu(tf.matmul(pair_hat, W_adv_1) + b_adv_1)
            pair_hat_l2 = self.leaky_relu(tf.matmul(pair_hat_l1, W_adv_2) + b_adv_2)
            bin_hat = self.leaky_relu(tf.matmul(pair_hat_l2, W_bin) + b_bin)

            GP_loss = tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.gradients(bin_hat, pair_hat)[0]**2, axis=1)) - 1.)**2)

        # discrimination loss
        with tf.variable_scope('discriminate') as scope_2_2:
            # pair_pos_norm = tf.contrib.layers.layer_norm(pair_pos)
            pos_pair_l1 = self.leaky_relu(tf.matmul(pos_pair, W_adv_1) + b_adv_1)
            pos_pair_l2 = self.leaky_relu(tf.matmul(pos_pair_l1, W_adv_2) + b_adv_2)
            bin_pos = self.leaky_relu(tf.matmul(pos_pair_l2, W_bin) + b_bin)

            # pair_neg_norm = tf.contrib.layers.layer_norm(pair_neg)
            neg_pair_l1 = self.leaky_relu(tf.matmul(neg_pair, W_adv_1) + b_adv_1)
            neg_pair_l2 = self.leaky_relu(tf.matmul(neg_pair_l1, W_adv_2) + b_adv_2)
            bin_neg = self.leaky_relu(tf.matmul(neg_pair_l2, W_bin) + b_bin)

            discrimination_loss = - tf.losses.mean_squared_error(bin_pos, bin_neg)
        return GP_loss, discrimination_loss
    '''

    def build_model(self):
        pos_feat = self.pos_feat
        neg_feats = self.neg_feats
        neighbors = self.neighbors
        masks = self.masks
        
        with tf.variable_scope('generator') as scope_g:
            # Encoding & decoding of input feats
            enc = self.encode(tf.nn.embedding_lookup(self.word_embedding, pos_feat))
            # enc = tf.clip_by_norm(enc, 1., axes=[-1])
        
            # neighbor_encs = tf.unstack(neighbors, axis=1)
            # for i, neighbor in enumerate(neighbor_encs):
                # neighbor_encs[i] = (tf.nn.embedding_lookup(self.context_embedding, neighbor), tf.nn.embedding_lookup(self.dot_bias, neighbor))
            # neighbor_encs, neighbor_bias = list(zip(*(neighbor_encs)))
            # neighbor_encs = tf.reshape(tf.stack(neighbor_encs, axis=1), [-1, self.memory_dim])
            neighbor_encs = tf.nn.embedding_lookup(self.context_embedding, tf.reshape(neighbors, [-1]))
            neighbor_encs = self.skip_encode(neighbor_encs)
            neighbor_encs = tf.unstack(tf.reshape(neighbor_encs, [-1, 2*self.gram_num, self.memory_dim]), axis=1)

            # negs = tf.unstack(neg_feats, axis=1)
            # targets = []
            # targets_bias = []
            # for i, neg in enumerate(negs):
                # targets.append(tf.nn.embedding_lookup(self.context_embedding, neg))
                # targets_bias.append(tf.nn.embedding_lookup(self.dot_bias, neg))
            # targets = tf.reshape(tf.stack(targets, axis=1), [-1, self.memory_dim])
            # negs_word = tf.nn.embedding_lookup(self.word_embedding, tf.reshape(neg_feats, [-1]))
            # negs_word = self.encode(negs_word, reuse=True)
            # negs_word = tf.unstack(tf.reshape(negs_word, 
                                              # [-1, self.neg_sample_num, self.memory_dim]), axis=1)
            negs_context = tf.nn.embedding_lookup(self.context_embedding, tf.reshape(neg_feats, [-1]))
            negs_context = self.skip_encode(negs_context, reuse=True)
            negs_context = tf.unstack(tf.reshape(negs_context, 
                                              [-1, self.neg_sample_num, self.memory_dim]), axis=1)
            # targets_word = []
            # for i, neg in enumerate(negs):
                # targets_word.append(tf.nn.embedding_lookup(self.word_embedding, neg))
            # targets_word = tf.reshape(tf.stack(targets_word, axis=0), [-1, self.memory_dim])
            # targets_word = self.encode(targets_word, reuse=True)
            # targets_word = tf.clip_by_norm(targets_word, 1., axes=[-1])
            # targets_word = tf.unstack(tf.reshape(targets_word, [self.neg_sample_num, -1, self.memory_dim]), axis=0)
            # targets += targets_word
            
            masks = tf.unstack(masks, axis=1)
            loss = tf.constant(0.)
            first_products = []
            norms = []
            # softmax = None
            # print(len(neighbor_encs))
            # print(len(neighbor_bias))
            # for i, (neighbor, bias) in enumerate(zip(neighbor_encs, neighbor_bias)):
            # norm_value = tf.reduce_suim(enc*neighbor, -1)
            for i, neighbor in enumerate(neighbor_encs):

                product = tf.reduce_sum(enc*neighbor, -1) # + bias
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(product, dtype=tf.float32), 
                    logits=product) * masks[i])
                # loss -= tf.reduce_mean(tf.log(tf.sigmoid(product)))
                first_products.append(product)
                norms.append(tf.norm(enc, axis=-1)*tf.norm(neighbor, axis=-1))

            # dot_products_word = []
            # for i, neg in enumerate(negs_word):
                # # for neg, neg_bias in zip(targets, targets_bias):
                # product = tf.reduce_sum(neg*neighbor, -1) # + neg_bias
                # # dot_products.append(tf.maximum(product, tf.zeros_like(product)))
                # dot_products_word.append(product)
                # first_products.append(product)
                # norms.append(tf.norm(neg, axis=-1)*tf.norm(neighbor, axis=-1))
            # dot_products_word = tf.stack(dot_products_word, axis=1)

            # dot_products_context = []
            for i, neg in enumerate(negs_context):
                # for neg, neg_bias in zip(targets, targets_bias):
                product = tf.reduce_sum(neg*enc, -1) # + neg_bias
                # loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    # labels=tf.zeros_like(dot_products_word, dtype=tf.float32), 
                    # logits=product) * tf.expand_dims(masks[i], axis=1))
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(product, dtype=tf.float32), 
                    logits=product) * masks[i])
                # dot_products.append(tf.maximum(product, tf.zeros_like(product)))
                # dot_products_context.append(product)
                first_products.append(product)
                norms.append(tf.norm(neg, axis=-1)*tf.norm(enc, axis=-1))

            # dot_products_context = tf.stack(dot_products_context, axis=1)
            norms = tf.stack(norms, axis=1)
            first_products = tf.stack(first_products, axis=1) / norms
                # softmax = tf.nn.softmax(dot_products)
            # loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                # labels=tf.zeros([tf.shape(enc)[0]], dtype=tf.int64), 
                # logits=dot_products))
            # loss -= tf.reduce_mean(tf.log(tf.sigmoid(-dot_products_word))) * self.neg_sample_num

            # loss -= tf.nn.sigmoid_cross_entropy_with_logits(
                # labels=tf.one_hot(tf.zeros_like(product, dtype=tf.int64), self.neg_sample_num+1), 
                # logits=10*tf.one_hot(tf.zeros_like(product, dtype=tf.int64), self.neg_sample_num+1))

        '''    
        # BCE loss
            pos_loss = tf.constant(0.)
            for neighbor_enc in neighbor_encs:
                # pos_loss += tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([tf.shape(enc)[0]]), predictions=self.cosine_similarity(enc, neighbor_enc)))
                pos_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones([tf.shape(pos_feat)[0]]), 
                    logits=tf.reduce_sum(tf.nn.embedding_lookup(self.word_embedding, enc)*
                                         tf.nn.embedding_lookup(self.context_embedding, neighbor_enc), -1)))
            # Negative sampling
            neg_loss = tf.constant(0.)
            for neg_sample in negs:
                for neighbor_enc in neighbor_encs:
                    # neg_loss += tf.reduce_sum(tf.losses.mean_squared_error(labels=tf.zeros([tf.shape(enc)[0]]), predictions=self.cosine_similarity(neighbor_enc, neg_sample)))
                    neg_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.zeros([tf.shape(pos_feat)[0]]), 
                        logits=tf.reduce_sum(tf.nn.embedding_lookup(self.context_embedding, neighbor_enc)*
                                             tf.nn.embedding_lookup(self.word_embedding, neg_sample), -1)))
                    # similarity_loss += tf.reduce_mean(tf.maximum(self.cosine_similarity(enc, neg_sample), tf.constant(0.)))
            # Neighbor loss
            # neighbor_loss = tf.constant(0.)
            # for neighbor_enc in neighbor_encs:
                # # neighbor_loss += tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([tf.shape(enc)[0]]), predictions=self.cosine_similarity(neighbor_enc, enc)))
                # neighbor_loss += tf.reduce_mean(tf.maximum(tf.reduce_sum(enc*neighbor_enc, -1), tf.constant(0.)))
        '''
        '''
        with tf.variable_scope('discriminator') as scope_d:
            pos_pair = tf.reshape(tf.concat([neighbors, tf.expand_dims(pos_feat, 1)], 1), 
                                [-1, (2*self.gram_num+1)*self.feat_dim])
            neg_pairs = [tf.reshape(tf.concat([neighbors, tf.expand_dims(dec, 1)], 1), 
                                [-1, (2*self.gram_num+1)*self.feat_dim])]
            # for neg_feat in neg_feats:
                # neg_pairs.append(tf.reshape(tf.concat([neighbors, tf.expand_dims(neg_feat, 1)], 1), 
                                    # [-1, (2*self.gram_num+1)*self.feat_dim]))
            GP_loss, discrimination_loss = self.adversarial_training(pos_pair, neg_pairs[0])
        '''
        # return enc, pos_loss, neg_loss#, neighbor_loss#, discrimination_loss, GP_loss
        # return enc, pos_loss/(2*self.gram_num+1) , neg_loss/(2*self.gram_num+1)
        # return tf.reduce_mean(softmax_losses), tf.reshape(softmax, [-1, self.neg_sample_num+1]), first_products
        return loss / (2*self.gram_num), first_products, enc, neighbor_encs, negs_context#, softmax

    def build_test(self):
        with tf.variable_scope('generator') as scope_g:
            word_embedding = self.encode(self.word_embedding)
            # word_embedding = tf.clip_by_norm(word_embedding, 1., axes=[-1])
            context_embedding = self.skip_encode(self.context_embedding)
            # context_embedding = tf.clip_by_norm(context_embedding, 1., axes=[-1])
        return word_embedding, context_embedding
