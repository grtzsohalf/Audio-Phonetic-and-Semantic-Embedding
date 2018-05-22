#!/usr/bin/env python3

import tensorflow as tf
from typing import Array
import numpy as np
import logging
import math
import pickle as pkl
LOGGER = logging.getLogger(__name__)


class audio2text_GAN(object):
    AUDIO_HOLDER = 'audio_holder'
    TEXT_HOLDER = 'text_holder'
    LR_HOLDER = 'lr_holder'
    OP_D_TRAIN = 'op_d_train'
    OP_G_TRAIN = 'op_g_train'

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        D_struct: Array[int],
        batch_size: int,
        logger=LOGGER,
    ):
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.D_struct = D_struct
        self.logger = logger
        with self.graph.as_default():
            self._build_graph()
        # assert audio_k_way == text_n_vocab
        assert audio_dim == text_dim
        self.dim = audio_dim
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=self.config)
        self.sess.run(tf.variables_initializer(
            varlist=self.graph.get_collection('variables')))
        self.saver = tf.train.Saver(
            varlist=self.graph.get_collection('variables'),
            max_to_keep=None)

    def _build_graph(self):
        # None stands for the batch
        self.text_holder = tf.placeholder(
            tf.float32,
            [None, self.text_dim],
            name=self.AUDIO_HOLDER,
            )
        self.audio_holder = tf.placeholder(
            tf.float32,
            [None, self.audio_dim],
            name=self.TEXT_HOLDER,
            )
        self.lr = tf.placeholder(
            tf.float32,
            [1],
            name=self.LR_HOLDER,
            )

        self.x_trans = self.Generator(self.text_holder)
        # instead of sampling, we concate the x_trans and audio_holder
        # together to generate the Discriminator's input
        batch_size = tf.shape(self.x_trans)[0]
        label_zero = tf.zeros([batch_size, 1], dtype=tf.float32)
        label_one = tf.ones([batch_size, 1], dtype=tf.float32)
        D_input = tf.concat(
                    [self.x_trans, self.audio_holder],
                    axis=0,
                    name='input_concat')
        D_label = tf.concat([label_zero, label_one], axis=0, name='D_label')
        G_label = tf.concat([label_one, label_zero], axis=0, name='G_label')
        D_out = self.Discriminator(D_input)
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'D_' in var.name]
        self.g_vars = [var for var in t_vars if 'G_' in var.name]

        self.D_Loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=D_label,
            logits=D_out,
            name='D_loss')
        self.G_Loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=G_label,
            logits=D_out,
            name='G_loss')

        self.train_D_op = tf.train.AdamOptimizer(self.lr).minimize(
            self.D_Loss,
            var_list=self.d_vars,
            name=self.OP_D_TRAIN,
            )

        self.train_G_op = tf.train.AdamOptimizer(self.lr).minimize(
            self.G_Loss,
            var_list=self.g_vars,
            name=self.OP_G_TRAIN,
            )
        return

    # Generator holds the Matrix W such that WX = Y
    def Generator(self, X):
        with tf.variable_scope('generator') as scope: # noqa
            W = tf.get_variable(
                'G_W',
                shape=[None, self.text_dim, self.audio_dim],
                initializer=tf.keras.initializers.lecun_normal,
                )
            self.W = W
            self._update_w(0.99)
            x_trans = self.W @ X
            return x_trans

    # Discriminator holds the NN to discriminate two different sources
    def Discriminator(self, X):
        res = None
        with tf.variable_scope('discriminator') as scope: # noqa
            for order in range(len(self.D_struct)):
                if order == 0:
                    W = tf.get_variable(
                        'D_W1',
                        shape=[self.text_dim, self.D_struct[order]],
                        initializer=tf.keras.initializers.lecun_normal,
                        )
                    b = tf.get_variable(
                        'D_b1',
                        shape=[self.D_struct[order]],
                        initializer=tf.keras.initializers.lecun_normal,
                        )
                    res = tf.nn.selu(W @ X + b)
                else:
                    W = tf.get_variable(
                        'D_W'+str(order),
                        shape=[self.D_struct[order-1],
                               self.D_struct[order]],
                        initializer=tf.keras.initializers.lecun_normal,
                        )
                    b = tf.get_variable(
                        'D_b'+str(order),
                        shape=[self.D_struct[order]],
                        initializer=tf.keras.initializers.lecun_normal,
                        )
                    res = tf.nn.selu(W @ res + b)

            W = tf.get_variable(
                'D_W_out',
                shape=[self.D_struct[-1], 1],
                initializer=tf.keras.initializers.lecun_normal,
                )
            b = tf.get_variable(
                'D_b_out', shape=[1],
                initializer=tf.keras.initializers.lecun_normal,
                )
            res = W @ res + b
        return res

    def _update_w(self, beta):
        self.W = (1. + beta) * self.W - beta * (self.W @ self.W) @ self.W

    def gen_batch(self):
        np.random.shuffle(self.text_embeds)
        np.random.shuffle(self.audio_embeds)
        cnt = 0
        num = math.ceil(float(self.text_embeds) / self.batch_size)
        while True:
            if cnt == num:
                cnt = 0
                np.random.shuffle(self.text_embeds)
                np.random.shuffle(self.audio_embeds)
            start = cnt * self.batch_size
            end = start + self.batch_size
            yield self.text_embeds[start:end], self.audio_embeds[start:end]
            cnt += 1

    def train(
        self,
        audio_embeds: np.ndarray,
        text_embeds: np.ndarray,
        lr: float,
        batch_size: int,
        epoch: int,
        G_num_per_batch: int=1,
        D_num_per_batch: int=1,
        early_stopping: bool=True,
    ):

        self.text_embeds = text_embeds
        self.audio_embeds = audio_embeds
        batch_per_epoch = len(text_embeds) / batch_size
        batch_generator = self.gen_batch()
        audio_place = self.graph.get_tensor_by_name(self.AUDIO_HOLDER + ':0')
        text_place = self.graph.get_tensor_by_name(self.TEXT_HOLDER + ':0')
        lr_place = self.graph.get_tensor_by_name(self.LR_HOLDER + ':0')
        G_loss_tensor = self.graph.get_tensor_by_name('G_Loss' + ':0')
        D_loss_tensor = self.graph.get_tensor_by_name('D_Loss' + ':0')
        G_OP = self.graph.get_operation_by_name(self.OP_G_TRAIN)
        D_OP = self.graph.get_operation_by_name(self.OP_D_TRAIN)
        G_loss_op = self.graph.get_operation_by_name('G_Loss')
        D_loss_op = self.graph.get_operation_by_name('D_Loss')
        for epoch_num in range(epoch):
            for batch_cnt in batch_per_epoch:
                batch_text_embeds, batch_audio_embeds = next(batch_generator)
                for G_num in range(G_num_per_batch):
                    _, G_batch_loss = self.sess.run(
                        [G_OP, G_loss_tensor],
                        feed_dict={
                            audio_place: batch_audio_embeds,
                            text_place: batch_text_embeds,
                            lr_place: lr,
                        })
                for D_num in range(D_num_per_batch):
                    _, D_batch_loss = self.sess.run(
                        [D_OP, D_loss_tensor],
                        feed_dict={
                            audio_place: batch_audio_embeds,
                            text_place: batch_text_embeds,
                            lr_place: lr,
                        })
                self.logger.debug('batch train G_loss:{}, D_loss:{}'.format(
                    G_batch_loss, D_batch_loss))

            _, G_all_loss = self.sess.run(
                [G_loss_op, G_loss_tensor],
                feed_dict={
                    audio_place: self.audio_embeds,
                    text_place: self.text_embeds,
                })
            _, D_all_loss = self.sess.run(
                [D_loss_op, D_loss_tensor],
                feed_dict={
                    audio_place: self.audio_embeds,
                    text_place: self.text_embeds,
                })
            self.logger.info('Epoch {}, G loss:{}, D loss:{}'.format(
                epoch_num,
                G_all_loss,
                D_all_loss,
                ))

    @staticmethod
    def gen_hyper_model_path(path: str):
        hyper_path = path + '_hyper.pkl'
        var_path = path + '_var.mdl'
        return hyper_path, var_path

    def save(self, path: str):
        hyper_path, var_path = self.gen_hyper_model_path(path)
        with open(hyper_path, 'wb') as hpb:
            params = {
                'audio_dim': self.audio_dim,
                'text_dim': self.text_dim,
                'D_struct': self.D_struct,
                'batch_size': self.batch_size,
            }
            pkl.dump(params, hpb)
        self.saver.save(self.sess, path)

    @classmethod
    def load(cls, path: str):
        hyper_path, var_path = cls.gen_hyper_model_path(path)
        with open(hyper_path, 'rb') as f:
            params = pkl.load(f)
        mdl = cls(**params)
        mdl.saver.restore(mdl.sess, var_path)
        return mdl
