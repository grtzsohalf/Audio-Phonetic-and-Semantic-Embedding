#!/usr/bin/env python3

import audio2text_ICP as ICP
import data_parser as dp
import argparse
from sklearn.decomposition import PCA
import numpy as np
import math

FLAG = None


def find_pair(x, Y):
    '''
    Args:
      x is a vector
      Y is a bunch of vectors

    Return:
      The indice of the closest Y to the vector x
    '''
    return np.argmax(np.linalg.norm(x-Y, axis=1))


def gen_batch(embed):
    cnt = 0
    max_cnt = math.floor(float(embed.shape[0]) / FLAG.mb)
    while True:
        start = cnt * FLAG.mb
        end = (cnt+1) * FLAG.mb
        yield embed[start:end]

        if cnt >= max_cnt:
            cnt = 0


def generate_pair(X, Y):
    y_ = np.zeros(shape=(X.shape[0], Y.shape[1]))
    for i, x in enumerate(X):
        y_[i] = Y[find_pair(x, Y)]
    return y_


def generate_pair_map(X, Y):
    m = []
    for i, x in enumerate(X):
        m.append(find_pair(x, Y))
    return m


def ICP_train(text, audio):
    text_copy = np.copy(text)
    audio_copy = np.copy(audio)

    np.random.shuffle(text_copy)
    np.random.shuffle(audio_copy)

    g_text = gen_batch(text_copy)
    g_audio = gen_batch(audio_copy)
    train_core = ICP.audio2text_ICP(
        audio.shape[1],
        text.shape[1],
        FLAG.mb,
        FLAG.penalty_lambda)
    dim = audio.shape[1]
    a2t_mat = np.identity(dim)
    t2a_mat = np.identity(dim)
    print('\n')
    for i in range(10000):
        batch_t2a_text = next(g_text)
        batch_t2a_text_tr = np.transpose(batch_t2a_text)
        batch_text = np.transpose(np.matmul(t2a_mat, batch_t2a_text_tr))
        #batch_t2a_audio = generate_pair(batch_text, audio)
        batch_a2t_audio = next(g_audio)
        batch_a2t_audio_tr = np.transpose(batch_a2t_audio)
        batch_audio = np.transpose(np.matmul(a2t_mat, batch_a2t_audio_tr))
        batch_a2t_text = generate_pair(batch_audio, text)
        losses = train_core.train(
            t2a_text=batch_t2a_text,
            t2a_audio=batch_t2a_audio,
            a2t_text=batch_a2t_text,
            a2t_audio=batch_a2t_audio,
            lr=FLAG.init_lr,
            epoch=1,
            )
        print('iter {}:'.format(i))
        print(losses)
        # log_str = 'total_loss: {0}, t2a: {1}, a2t:{2}, t2a2t:{3}, a2t2a:{4}'
        # print(log_str.format(losses))
        tmp = train_core.get_matrix()
        a2t_mat = np.reshape(np.array(tmp[0]), (dim, dim))
        t2a_mat = np.reshape(np.array(tmp[1]), (dim, dim))


def ICP_train_full(text_t2a, audio_t2a, audio_a2t, text_a2t):
    '''
    Args:
      text_t2a: the source of doing t2a ,the text embedding
      audio_t2a: the target of doing t2a, the paired audio embedding
      audio_a2t: the source of doing a2t, the audio embedding
      text_a2t: the target of doing a2t, the paired text embedding
    Return:
      at2_mat: the a2t transform matrix
      t2a_mat: the t2a transform matrix
    '''

    order_t2a = np.shuffle(np.arange(text_t2a.shape[0]))
    order_a2t = np.shuffle(np.arnage(audio_a2t.shape[0]))
    np.shuffle(order_t2a)
    np.shuffle(order_a2t)

    text_t2a_copy = text_t2a[order_t2a]
    audio_t2a_copy = audio_t2a[order_t2a]
    audio_a2t_copy = audio_a2t[order_a2t]
    text_a2t_copy = text_t2a[order_a2t]

    train_core = ICP.audio2text_ICP(
        audio_t2a.shape[1],
        text_a2t.shape[1],
        FLAG.mb,
        FLAG.penalty_lambda)
    dim = audio_t2a.shape[1]
    a2t_mat = np.identity(dim)
    t2a_mat = np.identity(dim)

    g_audio_a2t = gen_batch(audio_a2t_copy)
    g_audio_t2a = gen_batch(audio_t2a_copy)
    g_text_a2t = gen_batch(text_a2t_copy)
    g_text_t2a = gen_batch(text_t2a_copy)

    for i in range(100000):
        batch_a2t_audio = next(g_audio_a2t)
        batch_a2t_text = next(g_text_a2t)
        batch_t2a_audio = next(g_audio_t2a)
        batch_t2a_text = next(g_text_t2a)

        train_core.train(
            t2a_text=batch_t2a_text,
            t2a_audio=batch_t2a_audio,
            a2t_text=batch_a2t_text,
            a2t_audio=batch_a2t_audio,
            lr=FLAG.lr,
            epoch=1,
            )
        tmp = train_core.get_matrix()
        a2t_mat = np.reshape(np.array(tmp[0]), (dim, dim))
        t2a_mat = np.reshape(np.array(tmp[1]), (dim, dim))
    return a2t_mat, t2a_mat


def PCA_transform(text, audio):
    pca = PCA(n_components=50)
    text_pca = np.array(pca.fit_transform(text))
    audio_pca = np.array(pca.fit_transform(audio))
    return text_pca, audio_pca


def main():
    text_emb, text_labs = dp.read_csv_file(FLAG.text_embeds, ' ')
    audio_emb, audio_labs = dp.read_csv_file(FLAG.audio_embeds, ' ')
    text_pca, audio_pca = PCA_transform(text_emb, audio_emb)

    a2t_mat, t2a_mat = ICP_train(text_pca, audio_pca)

    t2a_text = np.transpose(text_pca)
    tmp_text = np.matmul(t2a_mat, t2a_text)

    # t2a_audio_map = generate_pair(tmp_text, audio_pca)

    a2t_audio = np.transpose(audio_pca)
    tmp_audio = np.matmul(a2t_mat, a2t_audio)
    # a2t_text_map = generate_pair(tmp_audio, text_pca)

    np_audio = np.array(audio_emb)
    t2a_audio_emb = np_audio[t2a_audio_map]
    np_text = np.array(text_emb)
    a2t_text_emb = np_text[a2t_text_map]

    a2t_mat, t2a_mat = ICP_train_full(
        text_emb,
        t2a_audio_emb,
        audio_emb,
        a2t_text_emb)


def addParser():
    parser = argparse.ArgumentParser(
        prog="PROG",
        description='Audio2vec Training Script')
    parser.add_argument(
        '--init_lr',
        type=float,
        default=0.1,
        metavar='<--initial learning rate>')
    parser.add_argument(
        '--penalty_lambda',
        type=float,
        default=0.5,
        metavar='<--lambda of penalty>')
    parser.add_argument(
        '--decay_factor',
        type=float,
        default=0.95,
        metavar='<decay factor of learning rate>')
    parser.add_argument(
        '--mb',
        type=int,
        default=500,
        metavar='--<mini batch size>',
        help='The mini batch size while training')
    parser.add_argument(
        '--max_step',
        type=int,
        default=20000,
        metavar='--<max step>',
        help='The max step for training')
    parser.add_argument(
        'text_embeds',
        metavar='<text embedding>')
    parser.add_argument(
        'audio_embeds',
        metavar='<audio embedding>')
    return parser


if __name__ == '__main__':
    FLAG = addParser().parse_args()
    main()
