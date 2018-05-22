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

        self.pos_feat = tf.placeholder(tf.float32, [None, feat_dim])
        self.neg_feats = tf.placeholder(tf.float32, [None, 2*self.gram_num, self.neg_sample_num, feat_dim])
        # self.neg_feats = tf.placeholder(tf.float32, [None, self.neg_sample_num, feat_dim])
        self.neighbors = tf.placeholder(tf.float32, [None, 2*self.gram_num, feat_dim])
        # self.neighbors = tf.placeholder(tf.float32, [None, feat_dim])
        self.masks = tf.placeholder(tf.float32, [None, 2*self.gram_num])

    def leaky_relu(self, x, alpha=0.01):
        return tf.maximum(x, alpha*x)

    def cosine_similarity(self, x, y):
        norm_x = tf.nn.l2_normalize(x, -1)
        norm_y = tf.nn.l2_normalize(y, -1)
        return tf.reduce_sum(norm_x*norm_y, -1)

    def skip_encode(self, neighbor, reuse=False):
        with tf.variable_scope('skip_encode', reuse=reuse) as scope_skip_enc:
            W_enc_1 = tf.get_variable("skip_enc_w_1", [self.feat_dim, self.memory_dim*2])
            b_enc_1 = tf.get_variable("skip_enc_b_1", shape=[self.memory_dim*2])
            W_enc_2 = tf.get_variable("skip_enc_w_2", [self.memory_dim*2, self.memory_dim*2])
            b_enc_2 = tf.get_variable("skip_enc_b_2", shape=[self.memory_dim*2])
            W_enc_3 = tf.get_variable("skip_enc_w_3", [self.memory_dim*2, self.memory_dim])
            b_enc_3 = tf.get_variable("skip_enc_b_3", shape=[self.memory_dim])
            W_enc_4 = tf.get_variable("skip_enc_w_4", [self.memory_dim, self.memory_dim])
            b_enc_4 = tf.get_variable("skip_enc_b_4", shape=[self.memory_dim])
            W_enc_5 = tf.get_variable("skip_enc_w_5", [self.memory_dim, self.memory_dim])
            b_enc_5 = tf.get_variable("skip_enc_b_5", shape=[self.memory_dim])
            W_enc_6 = tf.get_variable("skip_enc_w_6", [self.memory_dim, self.memory_dim])
            b_enc_6 = tf.get_variable("skip_enc_b_6", shape=[self.memory_dim])

            # neighbor = tf.contrib.layers.batch_norm(neighbor)
            enc = self.leaky_relu(tf.matmul(neighbor, W_enc_1) + b_enc_1)
            # enc = tf.matmul(neighbor, W_enc_1) + b_enc_1
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_2) + b_enc_2)
            # enc = tf.matmul(enc, W_enc_2) + b_enc_2
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_3) + b_enc_3)
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_4) + b_enc_4)
            # enc = tf.matmul(enc, W_enc_4) + b_enc_4
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_5) + b_enc_5)
            enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_6) + b_enc_6)
            enc = tf.matmul(enc, W_enc_6) + b_enc_6
        return enc

    def encode(self, feat, reuse=False):
        with tf.variable_scope('encode', reuse=reuse) as scope_enc:
            W_enc_1 = tf.get_variable("enc_w_1", [self.feat_dim, self.memory_dim*2])
            b_enc_1 = tf.get_variable("enc_b_1", shape=[self.memory_dim*2])
            W_enc_2 = tf.get_variable("enc_w_2", [self.memory_dim*2, self.memory_dim*2])
            b_enc_2 = tf.get_variable("enc_b_2", shape=[self.memory_dim*2])
            W_enc_3 = tf.get_variable("enc_w_3", [self.memory_dim*2, self.memory_dim])
            b_enc_3 = tf.get_variable("enc_b_3", shape=[self.memory_dim])
            W_enc_4 = tf.get_variable("enc_w_4", [self.memory_dim, self.memory_dim])
            b_enc_4 = tf.get_variable("enc_b_4", shape=[self.memory_dim])
            W_enc_5 = tf.get_variable("enc_w_5", [self.memory_dim, self.memory_dim])
            b_enc_5 = tf.get_variable("enc_b_5", shape=[self.memory_dim])
            W_enc_6 = tf.get_variable("enc_w_6", [self.memory_dim, self.memory_dim])
            b_enc_6 = tf.get_variable("enc_b_6", shape=[self.memory_dim])

            # feat = tf.contrib.layers.batch_norm(feat)
            enc = self.leaky_relu(tf.matmul(feat, W_enc_1) + b_enc_1)
            # enc = tf.matmul(feat, W_enc_1) + b_enc_1
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_2) + b_enc_2)
            # enc = tf.matmul(enc, W_enc_2) + b_enc_2
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_3) + b_enc_3)
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_4) + b_enc_4)
            # enc = tf.matmul(enc, W_enc_4) + b_enc_4
            enc = tf.contrib.layers.batch_norm(enc)
            enc = self.leaky_relu(tf.matmul(enc, W_enc_5) + b_enc_5)
            enc = tf.contrib.layers.batch_norm(enc)
            # enc = self.leaky_relu(tf.matmul(enc, W_enc_6) + b_enc_6)
            enc = tf.matmul(enc, W_enc_6) + b_enc_6
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
            enc = self.encode(pos_feat)
            enc = tf.nn.l2_normalize(enc, dim=-1)
            # enc = tf.clip_by_norm(enc, 1., axes=[-1])

            neighbor_encs = tf.reshape(neighbors, [-1, self.feat_dim])
            neighbor_encs = self.skip_encode(neighbor_encs)
            neighbor_encs = tf.nn.l2_normalize(neighbor_encs, dim=-1)
            neighbor_encs = tf.unstack(tf.reshape(neighbor_encs, [-1, 2*self.gram_num, self.memory_dim]), axis=1)
            # neighbor_encs = tf.clip_by_norm(neighbor_encs, 1., axes=[-1])

            targets = []
            targets = tf.reshape(neg_feats, [-1, self.feat_dim])
            target_encs = self.skip_encode(targets, reuse=True)
            target_encs = tf.nn.l2_normalize(target_encs, dim=-1)
            target_encs = tf.unstack(tf.reshape(target_encs, 
                                                [-1, 2*self.gram_num, self.neg_sample_num, self.memory_dim]), axis=1)
            # target_encs = tf.clip_by_norm(target_encs, 1., axes=[-1])
            # target_encs = tf.reshape(target_encs, [-1, self.neg_sample_num, self.memory_dim])

            # target_encs = self.encode(targets, reuse=True)
            # target_encs = tf.clip_by_norm(target_encs, 1., axes=[-1])
            # target_encs = tf.unstack(tf.reshape(target_encs, [-1, self.neg_sample_num, self.memory_dim]), axis=1)
            # negs += target_encs

            masks = tf.unstack(masks, axis=1)
            loss = tf.constant(0.)
            first_products = []
            # norms = []

            for i, neighbor in enumerate(neighbor_encs):
                # norm_value = tf.reduce_sum(enc*neighbor, -1)
                product = tf.reduce_sum(enc*neighbor, -1)
                if i == 0:
                    first_products.append(product)
                # norms.append(tf.norm(enc, axis=-1)*tf.norm(neighbor_encs, axis=-1))
                # loss += tf.reduce_mean(-tf.log(tf.sigmoid(product))) / (self.neg_sample_num+1)
                loss += tf.reduce_mean((tf.constant(1.)-product) ** 2 * masks[i])

                # targets = []
                # targets = tf.reshape(neg_feats[i], [-1, self.feat_dim])
                # target_encs = self.skip_encode(targets, reuse=True)
                # # target_encs = tf.clip_by_norm(target_encs, 1., axes=[-1])
                dot_products = []
                # target_enc = tf.unstack(tf.reshape(target_encs, [-1, self.neg_sample_num, self.memory_dim]), axis=1)
                for neg in tf.unstack(target_encs[i], axis=1):
                    product = tf.reduce_sum(neg*neighbor, -1)
                    dot_products.append(product)
                    if i == 0:
                        first_products.append(product)
                    # norms.append(tf.norm(enc, axis=-1)*tf.norm(neg, axis=-1))
                    # loss += tf.reduce_mean(-tf.log(1-tf.sigmoid(product))) / (self.neg_sample_num+1)
                
                dot_products = tf.stack(dot_products, axis=1)
                # norms = tf.stack(norms, axis=1)
                if i == 0:
                    first_products = tf.stack(first_products, axis=1) #/ norms

                # softmax_losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    # labels=tf.zeros([tf.shape(enc)[0]], dtype=tf.int64), 
                    # logits=dot_products))
                # softmax.append(tf.nn.softmax(dot_products))

                loss += tf.reduce_mean((dot_products-tf.constant(-1.)) ** 2 * tf.expand_dims(masks[i], axis=1))
                # loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    # labels=tf.one_hot(tf.zeros_like(product, dtype=tf.int64), self.neg_sample_num+1), 
                    # logits=dot_products) * tf.expand_dims(masks[i], axis=1))

            '''
            # BCE loss
            pos_loss = tf.constant(0.)
            for neighbor_enc in neighbor_encs:
                # pos_loss += tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([tf.shape(enc)[0]]), predictions=self.cosine_similarity(enc, neighbor_enc)))
                pos_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones([tf.shape(pos_feat)[0]]), logits=tf.reduce_sum(enc*neighbor_enc, -1)))
            # Negative sampling
            neg_c_loss = tf.constant(0.)
            for neg_sample in negs:
                for neighbor_enc in neighbor_encs:
                    # similarity_loss += tf.reduce_mean(tf.maximum(self.cosine_similarity(enc, neg_sample), tf.constant(0.)))
                    # neg_loss += tf.reduce_sum(tf.losses.mean_squared_error(labels=tf.zeros([tf.shape(enc)[0]]), predictions=self.cosine_similarity(neighbor_enc, neg_sample)))
                    neg_c_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.zeros([tf.shape(pos_feat)[0]]), logits=tf.reduce_sum(neighbor_enc*neg_sample, -1)))
            # Neighbor loss
            neg_w_loss = tf.constant(0.)
            for neg_sample in negs:
            # for neighbor_enc in neighbor_encs:
                # # neighbor_loss += tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([tf.shape(enc)[0]]), predictions=self.cosine_similarity(neighbor_enc, enc)))
                neg_w_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros([tf.shape(pos_feat)[0]]), logits=tf.reduce_sum(neg_sample*enc, -1)))
                # neg_w_loss += tf.reduce_mean(tf.maximum(tf.reduce_sum(enc*neg_sample, -1), tf.constant(0.)))
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
        # return enc, pos_loss/(2*self.gram_num+1) , neg_c_loss/(self.neg_sample_num*(2*self.gram_num+1)), neg_w_loss/(self.neg_sample_num*(2*self.gram_num+1))
        # return enc, pos_loss, neg_c_loss, neg_w_loss
        # return enc, tf.reduce_mean(softmax_losses), tf.reshape(softmax, [-1, self.neg_sample_num+1]), first_products
        return enc, loss / (2*self.gram_num), first_products, neighbor_encs, target_encs
