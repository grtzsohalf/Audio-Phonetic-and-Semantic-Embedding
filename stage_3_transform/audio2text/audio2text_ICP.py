#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import logging
import math
import pickle as pkl
LOGGER = logging.getLogger(__name__)


class audio2text_ICP(object):
    AUDIO_HOLDER = 'audio_holder'
    TEXT_HOLDER = 'text_holder'
    LR_HOLDER = 'lr_holder'

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        batch_size: int,
        penalty_lambda: float,
        logger=LOGGER,
    ):
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.penalty_lambda = penalty_lambda
        self.logger = logger
        # assert audio_k_way == text_n_vocab
        assert audio_dim == text_dim
        self.dim = audio_dim
        with self.graph.as_default():
            self._build_graph()
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=self.config)
        self.sess.run(tf.variables_initializer(
            var_list=self.graph.get_collection('variables')))
        self.saver = tf.train.Saver(
            var_list=self.graph.get_collection('variables'),
            max_to_keep=None)

    def _build_graph(self):
        # None stands for the batch size
        self.t2a_text_holder = tf.placeholder(
            tf.float32,
            [None, self.dim],
            name='t2a_'+self.TEXT_HOLDER,
            )
        self.t2a_audio_holder = tf.placeholder(
            tf.float32,
            [None, self.dim],
            name='t2a_'+self.AUDIO_HOLDER,
            )
        self.a2t_audio_holder = tf.placeholder(
            tf.float32,
            [None, self.dim],
            name='a2t_'+self.AUDIO_HOLDER,
            )
        self.a2t_text_holder = tf.placeholder(
            tf.float32,
            [None, self.dim],
            name='a2t_'+self.TEXT_HOLDER,
            )

        self.lr = tf.placeholder(
            tf.float32,
            [],
            name=self.LR_HOLDER,
            )
        self.M_text2audio = tf.Variable(tf.eye(self.dim), name='t2a_matrix')
        self.B_t2a = tf.Variable(
            tf.random_normal([self.dim], stddev=0.2),
            name='t2a_bias')
        self.M_audio2text = tf.Variable(tf.eye(self.dim), name='a2t_matrix')
        self.B_a2t = tf.Variable(
            tf.random_normal([self.dim], stddev=0.2),
            name='a2t_bias')
        y_hat, xy_hat = self.t2a_trans(self.t2a_text_holder)

        x_hat, yx_hat = self.a2t_trans(self.a2t_audio_holder)

        t2a_loss = tf.losses.mean_squared_error(
            labels=self.t2a_audio_holder,
            predictions=y_hat,
            )
        tf.identity(t2a_loss, name='t2a_loss')

        a2t_loss = tf.losses.mean_squared_error(
            labels=self.a2t_text_holder,
            predictions=x_hat,
            )
        tf.identity(a2t_loss, name='a2t_loss')
        t2a2t_loss = tf.losses.mean_squared_error(
            labels=self.t2a_text_holder,
            predictions=xy_hat,
            weights=self.penalty_lambda,
            )
        tf.identity(t2a2t_loss, name='t2a2t_loss')
        a2t2a_loss = tf.losses.mean_squared_error(
            labels=self.a2t_audio_holder,
            predictions=yx_hat,
            weights=self.penalty_lambda,
            )
        tf.identity(a2t2a_loss, name='a2t2a_loss')
        all_loss = tf.identity(
            t2a_loss + a2t_loss + t2a2t_loss + a2t2a_loss,
            name='all_loss')
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(
            all_loss,
            var_list=[
                self.M_text2audio,
                self.B_t2a,
                self.M_audio2text,
                self.B_a2t],
            name='train_op')

        return

    # Wx @ X = Y_hat, Wy @ y_hat = X_hat_hat
    def t2a_trans(self, X):
        '''
        Args:
          x should be the text embedding
          y should be the audio embedding
        '''
        y_hat_tmp = tf.matmul(
            self.M_text2audio,
            tf.transpose(X))
        y_hat = tf.transpose(y_hat_tmp)
        y_hat = tf.add(y_hat, self.B_t2a, name='audio_hat')

        xy_hat_tmp = tf.matmul(
            self.M_audio2text,
            tf.transpose(y_hat))

        xy_hat = tf.transpose(xy_hat_tmp)
        xy_hat = tf.add(xy_hat, self.B_a2t, name='text_hat_hat')
        return y_hat, xy_hat

    # Wy @ Y = X_hat, Wx @ X_hat = Y_hat_hat
    def a2t_trans(self, Y):
        '''
        Args:
          x should be the text embedding
          y should be the audio embedding
        '''
        x_hat_tmp = tf.matmul(
            self.M_audio2text,
            tf.transpose(Y))
        x_hat = tf.transpose(x_hat_tmp)
        x_hat = tf.add(x_hat, self.B_a2t, name='text_hat')
        yx_hat_tmp = tf.matmul(
            self.M_text2audio,
            tf.transpose(x_hat))
        yx_hat = tf.transpose(yx_hat_tmp)
        yx_hat = tf.add(yx_hat, self.B_t2a, name='audio_hat_hat')
        return x_hat, yx_hat

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
        t2a_text: np.ndarray,
        t2a_audio: np.ndarray,
        a2t_text: np.ndarray,
        a2t_audio: np.ndarray,
        lr: float,
        epoch: int,
    ):
        t2a_text_place = self.graph.get_tensor_by_name(
            't2a_'+self.TEXT_HOLDER+':0')
        t2a_audio_place = self.graph.get_tensor_by_name(
            't2a_'+self.AUDIO_HOLDER + ':0')
        a2t_text_place = self.graph.get_tensor_by_name(
            'a2t_'+self.TEXT_HOLDER+':0')
        a2t_audio_place = self.graph.get_tensor_by_name(
            'a2t_'+self.AUDIO_HOLDER+':0')
        lr_place = self.graph.get_tensor_by_name(self.LR_HOLDER + ':0')

        t2a_loss_tensor = self.graph.get_tensor_by_name(
            't2a_loss:0')
        a2t_loss_tensor = self.graph.get_tensor_by_name(
            'a2t_loss:0')
        t2a2t_loss_tensor = self.graph.get_tensor_by_name(
            't2a2t_loss:0')
        a2t2a_loss_tensor = self.graph.get_tensor_by_name(
            'a2t2a_loss:0')
        all_loss = self.graph.get_tensor_by_name(
            'all_loss:0')
        train_op = self.graph.get_operation_by_name('train_op')
        loss = 0.
        for epoch_num in range(epoch):
            _, loss, t2a_loss, a2t_loss, t2a2t_loss, a2t2a_loss = self.sess.run(# noqa
                [
                    train_op,
                    all_loss,
                    t2a_loss_tensor,
                    a2t_loss_tensor,
                    t2a2t_loss_tensor,
                    a2t2a_loss_tensor,
                ],
                feed_dict={
                    t2a_text_place: t2a_text,
                    t2a_audio_place: t2a_audio,
                    a2t_audio_place: a2t_audio,
                    a2t_text_place: a2t_text,
                    lr_place: lr,
                })
            if epoch_num == 0:
                start_loss = (loss, t2a_loss, a2t_loss, t2a2t_loss, a2t2a_loss)
            if (epoch_num+1)%50 == 0:
                lr *= 0.95
            self.logger.info('Epoch {}, loss:{}'.format(
                epoch_num,
                loss,
                ))

        self.logger.info('Epoch {}, loss:{}'.format(
            epoch,
            loss,
            ))
        end_loss = (loss, t2a_loss, a2t_loss, t2a2t_loss, a2t2a_loss)
        return end_loss, start_loss

    def get_matrix_and_bias(self):
        a2t = self.graph.get_tensor_by_name(
            'a2t_matrix:0')
        t2a = self.graph.get_tensor_by_name(
            't2a_matrix:0')
        t2a_bias = self.graph.get_tensor_by_name(
            't2a_bias:0')
        a2t_bias = self.graph.get_tensor_by_name(
            'a2t_bias:0')
        a2t_matrix = self.sess.run([a2t])
        a2t_bias_ret = self.sess.run([a2t_bias])
        t2a_matrix = self.sess.run([t2a])
        t2a_bias_ret = self.sess.run([t2a_bias])
        return a2t_matrix, a2t_bias_ret, t2a_matrix, t2a_bias_ret

    def save(self, path: str):
        hyper_path, var_path = self.gen_hyper_model_path(path)
        with open(hyper_path, 'wb') as hpb:
            params = {
                'audio_dim': self.audio_dim,
                'text_dim': self.text_dim,
                'batch_size': self.batch_size,
                'penalty_lambda': self.penalty_lambda,
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
