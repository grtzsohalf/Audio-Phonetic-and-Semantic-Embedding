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

    def _build_input_graph(self):
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
        return

    def _build_Mb_graph(self):
        self.M_text2audio = tf.Variable(tf.eye(self.dim), name='t2a_matrix')
        self.b_text2audio = tf.Variable(
            tf.random_normal([self.dim]),
            name='t2a_bias',
            )
        self.M_audio2text = tf.Variable(tf.eye(self.dim), name='a2t_matrix')
        self.b_audio2text = tf.Variable(
            tf.random_normal([self.dim]),
            name='a2t_bias',
            )
        return

    def _build_Loss(self, y_hat, xy_hat, x_hat, yx_hat):
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
        return all_loss

    def _build_graph(self):
        # None stands for the batch size
        self._build_input_graph()
        self._build_Mb_graph()
        y_hat, xy_hat = self.t2a_trans(
            self.t2a_text_holder, self.t2a_audio_holder)
        x_hat, yx_hat = self.a2t_trans(
            self.a2t_audio_holder, self.a2t_text_holder)
        
        self.f1score = self.getFscore(1, self.a2t_text_holder, x_hat, 'f1score')
        self.f10score = self.getFscore(10, self.a2t_text_holder, x_hat, 'f10score')
        
        all_loss = self._build_Loss(y_hat, xy_hat, x_hat, yx_hat)
        # self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
            all_loss,
            var_list=[self.M_text2audio, self.M_audio2text, self.b_audio2text,
                self.b_text2audio],
            name='train_op')
        return

    def findNN(self, y, y_hat):
        '''
        Args:
            y: the source to be compared
            y_hat: the target comparing
        Returns:
           inds: the indices of the result y hat
        '''
        # using the x_hat to find the nearest neighbour


        return

    def findKNN(self, y, y_hat, k):
        '''
        Args:
            y: the source to be compared
            y_hat: the target comparing
        Returns:
           y_ave: the average over the kNN
        '''
        return
    # Wx @ X + by = Y_hat, Wy @ y_hat + bx = X_hat_hat


    def t2a_trans(self, X, Y):
        '''
        Args:
          x should be the text embedding
          y should be the audio embedding
        '''
        y_hat_tmp = tf.matmul(
            self.M_text2audio,
            tf.transpose(X))
        y_hat_tmp = tf.transpose(y_hat_tmp)
        y_hat_tmp = tf.add(y_hat_tmp, self.b_text2audio)
        # y_hat_tmp = tf.nn.sigmoid(y_hat_tmp)
        y_hat = tf.identity(y_hat_tmp, name='audio_hat')

        xy_hat_tmp = tf.matmul(
            self.M_audio2text,
            tf.transpose(y_hat_tmp))
        xy_hat_tmp = tf.transpose(xy_hat_tmp)
        xy_hat_tmp = tf.add(xy_hat_tmp, self.b_audio2text)
        # xy_hat_tmp = tf.nn.sigmoid(xy_hat_tmp)
        xy_hat = tf.identity(xy_hat_tmp, name='text_hat_hat')

        return y_hat, xy_hat

    # Wy @ Y = X_hat, Wx @ X_hat = Y_hat_hat
    def a2t_trans(self, Y, X):
        '''
        Args:
          x should be the text embedding
          y should be the audio embedding
        '''
        x_hat_tmp = tf.matmul(
            self.M_audio2text,
            tf.transpose(Y))
        x_hat_tmp = tf.transpose(x_hat_tmp)
        x_hat_tmp = x_hat_tmp + self.b_audio2text
        # x_hat_tmp = tf.nn.sigmoid(x_hat_tmp)
        x_hat = tf.identity(x_hat_tmp, name='text_hat')

        yx_hat_tmp = tf.matmul(
            self.M_text2audio,
            tf.transpose(x_hat_tmp))
        yx_hat_tmp = tf.transpose(yx_hat_tmp)
        yx_hat_tmp = yx_hat_tmp + self.b_text2audio
        # yx_hat_tmp = tf.nn.sigmoid(yx_hat_tmp)
        yx_hat = tf.identity(yx_hat_tmp, name='audio_hat_hat')
        return x_hat, yx_hat

    def gen_batch(self):
        order_t2a = np.arange(self.text_embeds.shape[0])
        np.random.shuffle(order_t2a)
        order_a2t = np.arange(self.audio_embeds.shape[0])
        np.random.shuffle(order_a2t)
        self.t2a_text = self.text_embeds[order_t2a]
        self.t2a_audio = self.audio_embeds[order_t2a]
        self.a2t_text = self.text_embeds[order_a2t]
        self.a2t_audio = self.audio_embeds[order_a2t]
        cnt = 0
        num = math.ceil(float(self.text_embeds.shape[0]) / self.batch_size)
        while True:
            if cnt == num:
                self.new_epoch = True
                cnt = 0
                order_t2a = np.arange(self.text_embeds.shape[0])
                np.random.shuffle(order_t2a)
                order_a2t = np.arange(self.audio_embeds.shape[0])
                np.random.shuffle(order_a2t)
                self.t2a_text = self.text_embeds[order_t2a]
                self.t2a_audio = self.audio_embeds[order_t2a]
                self.a2t_text = self.text_embeds[order_a2t]
                self.a2t_audio = self.audio_embeds[order_a2t]
            start = cnt * self.batch_size
            end = start + self.batch_size
            ret = (
                self.t2a_text[start:end],
                self.t2a_audio[start:end],
                self.a2t_audio[start:end],
                self.a2t_text[start:end],
                )
            yield ret
            cnt += 1
            self.new_epoch = False

    def setTextAudio(self, t2a_text, t2a_audio, a2t_audio, a2t_text):
        self.text_embeds = t2a_text
        self.audio_embeds = a2t_audio
        self.t2a_audio = t2a_audio
        self.t2a_text = t2a_text
        self.a2t_audio = a2t_audio
        self.a2t_text = a2t_text
        return

    def getFscore(self, top_N, emb1, emb2, scope='Fscore'):
        # Trying to use tensorflow top k function to build
        with tf.variable_scope(scope):
            norm1 = tf.nn.l2_normalize(emb1, axis=1)
            shape = tf.shape(norm1)
            re_norm1 = tf.reshape(norm1, shape=(shape[0], 1, shape[1]))
            norm2 = tf.nn.l2_normalize(emb2, axis=1)
            # needs broadcast
            cos_sim = tf.reduce_sum(tf.multiply(re_norm1, norm2), axis=2)
            # shape should be (num, num)
            # _, indices = tf.nn.top_k(cos_sim, k=top_N)
            targets = tf.range(shape[0])
            top_Ks = tf.nn.in_top_k(cos_sim, targets, top_N)
            f_score = tf.reduce_mean(tf.cast(top_Ks, tf.float32))
        return f_score

    def getInputTensors(self):
        t2a_text_place = self.graph.get_tensor_by_name(
            't2a_'+self.TEXT_HOLDER+':0')
        t2a_audio_place = self.graph.get_tensor_by_name(
            't2a_'+self.AUDIO_HOLDER + ':0')
        a2t_text_place = self.graph.get_tensor_by_name(
            'a2t_'+self.TEXT_HOLDER+':0')
        a2t_audio_place = self.graph.get_tensor_by_name(
            'a2t_'+self.AUDIO_HOLDER+':0')
        return t2a_text_place, t2a_audio_place, a2t_audio_place, a2t_text_place

    def getFscoreOps(self):
        return self.f1score, self.f10score

    def trainOracle(
        self,
        lr = 0.1,
        epoch = 1000,
        decay_factor = 0.95,
    ):
        t2a_text_place, t2a_audio_place, a2t_audio_place, a2t_text_place = self.getInputTensors()
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
        batch_iter = self.gen_batch()
        loss = 0.
        self.new_epoch = False
        for epoch_num in range(epoch):
            while not self.new_epoch:
                t2a_text, t2a_audio, a2t_audio, a2t_text = next(batch_iter)
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
            self.new_epoch = False
            self.logger.info('Epoch {}, loss:{}'.format(
                epoch_num,
                loss,
                ))
            loss_str = 'Epoch {}, loss: {}, t2a:{}, a2t:{}, t2a2t:{}, a2t2a:{} lr:{}'
            # print(loss_str.format(epoch_num, loss, t2a_loss, a2t_loss, t2a2t_loss,
            #     a2t2a_loss, lr))
            if (epoch_num+1) % 100 == 0:
                lr *= decay_factor
            if (epoch_num+1) % 1000 == 0:
                f1_score, f10_score = self.calcFscore(self.text_embeds, self.audio_embeds)
                print('F1 Score:{}, F10 Score:{}'.format(f1_score, f10_score))
        return (loss, t2a_loss, a2t_loss, t2a2t_loss, a2t2a_loss)

    def trainUnsupersiced(
        self,
        lr = 0.1,
        epoch = 1000,
        decay_factor = 0.95,
    ):
        t2a_text_place, t2a_audio_place, a2t_audio_place, a2t_text_place = self.getInputTensors()
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
        batch_iter = self.gen_batch()
        loss = 0.
        self.new_epoch = False
        for epoch_num in range(epoch):
            while not self.new_epoch:
                t2a_text, t2a_audio, a2t_audio, a2t_text = next(batch_iter)
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
            self.new_epoch = False
            self.logger.info('Epoch {}, loss:{}'.format(
                epoch_num,
                loss,
                ))
            loss_str = 'Epoch {}, loss: {}, t2a:{}, a2t:{}, t2a2t:{}, a2t2a:{} lr:{}'
            # print(loss_str.format(epoch_num, loss, t2a_loss, a2t_loss, t2a2t_loss,
            #     a2t2a_loss, lr))
            if (epoch_num+1) % 100 == 0:
                lr *= decay_factor
            if (epoch_num+1) % 1000 == 0:
                f1_score, f10_score = self.calcFscore(self.text_embeds, self.audio_embeds)
                print('F1 Score:{}, F10 Score:{}'.format(f1_score, f10_score))
        return (loss, t2a_loss, a2t_loss, t2a2t_loss, a2t2a_loss)
    def calcFscore(self, text_emb, audio_emb):
        f1_op, f10_op = self.getFscoreOps()
        t2a_text, t2a_audio, a2t_audio, a2t_text = self.getInputTensors()
        f1_score, f10_score = self.sess.run(
            [f1_op, f10_op],
            feed_dict={
                t2a_text: text_emb,
                t2a_audio: audio_emb,
                a2t_audio: audio_emb,
                a2t_text: text_emb,
                }
            )
        return f1_score, f10_score

    def get_matrix(self):
        a2t = self.graph.get_tensor_by_name(
            'a2t_matrix:0')
        t2a = self.graph.get_tensor_by_name(
            't2a_matrix:0')
        a2t_matrix = self.sess.run([a2t])
        t2a_matrix = self.sess.run([t2a])
        return a2t_matrix, t2a_matrix

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
