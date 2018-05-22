#!/usr/bin/env python3

import audio2text_ICP as ICP
import data_parser as dp
import sys
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
    return np.argmin(np.linalg.norm(x-Y, axis=1))


def find_top_N(x, Y, N=10):
    '''
    Args:
      x: the comparing feature, shape = [FEAT_DIM]
      Y: the feature data to be compaired shape = [NUM, FEAT_DIM]
      N: the number of the top N selection
    Returns:
      top_N: N args, shape = [N]
    '''
    norm = np.linalg.norm(x-Y, axis=1)
    top_N = np.argsort(norm)[-10:][::-1]
    return top_N


def gen_batch(embed):
    cnt = 0
    max_cnt = math.floor(float(embed.shape[0]) / FLAG.mb)
    while True:
        start = cnt * FLAG.mb
        end = (cnt+1) * FLAG.mb
        yield embed[start:end]
        cnt += 1
        if cnt >= max_cnt:
            cnt = 0


def generate_pair(X, Y):
    y_ = np.zeros(shape=(X.shape[0], Y.shape[1]))
    for i, x in enumerate(X):
        y_[i] = Y[find_pair(x, Y)]
    return y_


def generate_pair_list(X, Y):
    m = []
    for i, x in enumerate(X):
        m.append(find_top_N(x, Y, 10))
    return m


def print_loss(start_losses, end_losses, iter_num):
    if iter_num == 0:
        print('\n\n\n\n\n')
    sys.stdout.write('\033[F'*6)
    print('iter {} with lr {:.4f} :'.format(iter_num+1, FLAG.lr))
    header = ('T_LOSS', 't2a', 'a2t', 't2a2t', 'a2t2a')

    str_fmt = '{0: ^10}|{1: ^10}|{2: ^10}|{3: ^10}|{4: ^10}'
    loss_fmt = '{0: ^10.6f}|{1: ^10.6f}|{2: ^10.6f}|{3: ^10.6f}|{4: ^10.6f}'

    print(str_fmt.format(*header))
    print('Start Loss:')
    loss = (i for i in start_losses)
    print(loss_fmt.format(*loss))
    print('End Loss:')
    loss = (i for i in end_losses)
    print(loss_fmt.format(*loss))
    return


def evaluate(a2t_map, t2a_map, a2t_list, t2a_list):
    # calc a2t
    a2t_acc = 0.
    for i, t_i in enumerate(a2t_list):
        if a2t_map[i] in t_i:
            a2t_acc += 1
    a2t_acc /= len(a2t_list)
    # calc t2a
    t2a_acc = 0.
    for i, a_i in enumerate(t2a_list):
        if t2a_map[i] in a_i:
            t2a_acc += 1
    t2a_acc /= len(t2a_list)
    return a2t_acc, t2a_acc


def ICP_train(text, audio):
    text_copy = np.copy(text)
    audio_copy = np.copy(audio)
    t2a_m = [i for i in range(len(text))]
    a2t_m = [i for i in range(len(audio))]
    # np.random.shuffle(text_copy)
    # np.random.shuffle(audio_copy)

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
    lr = FLAG.init_lr

    tmp_text = np.matmul(a2t_mat, np.transpose(audio))
    tmp_text = np.transpose(tmp_text)
    tmp_audio = np.matmul(t2a_mat, np.transpose(text))
    tmp_audio = np.transpose(tmp_audio)
    t_l = generate_pair_list(text, tmp_text)
    a_l = generate_pair_list(audio, tmp_audio)
    a2t_acc, t2a_acc = evaluate(a2t_m, t2a_m, t_l, a_l)
    print('a2t_acc: {}, t2a_acc: {}'.format(a2t_acc, t2a_acc))
    print('\n'*5)

    for i in range(FLAG.max_step):
        ''' Oracle Setup'''
        batch_t2a_text = next(g_text)
        batch_a2t_audio = next(g_audio)
        batch_a2t_text = batch_t2a_text
        batch_t2a_audio = batch_a2t_audio

        if (i+1) % 40 == 0:
            lr *= FLAG.decay_factor
        if lr < 0.0005:
            lr = 0.0005
            FLAG.decay_factor = 1
            FLAG.max_inner_step += 1000

        end_losses, start_losses = train_core.train(
            t2a_text=batch_t2a_text,
            t2a_audio=batch_t2a_audio,
            a2t_text=batch_a2t_text,
            a2t_audio=batch_a2t_audio,
            lr=lr,
            epoch=FLAG.max_inner_step,
            )

        FLAG.lr = lr
        print_loss(start_losses, end_losses, i)
        tmp = train_core.get_matrix_and_bias()
        a2t_mat = np.reshape(np.array(tmp[0]), (dim, dim))
        a2t_b = np.reshape(np.array(tmp[1]), (dim))
        t2a_mat = np.reshape(np.array(tmp[2]), (dim, dim))
        t2a_b = np.reshape(np.array(tmp[3]), (dim))
        if i % 20 == 0:
            tmp_text = np.matmul(a2t_mat, np.transpose(audio))
            tmp_text = np.transpose(tmp_text)
            tmp_text = np.add(tmp_text, a2t_b)
            tmp_audio = np.matmul(t2a_mat, np.transpose(text))
            tmp_audio = np.transpose(tmp_audio)
            tmp_audio = np.add(tmp_audio, t2a_b)
            t_l = generate_pair_list(text, tmp_text)
            a_l = generate_pair_list(audio, tmp_audio)
            a2t_acc, t2a_acc = evaluate(a2t_m, t2a_m, t_l, a_l)
            print('a2t_acc: {}, t2a_acc: {}'.format(a2t_acc, t2a_acc))
            print('\n'*5)
    return a2t_mat, t2a_mat


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

    order_t2a = np.arange(text_t2a.shape[0])
    order_a2t = np.arnage(audio_a2t.shape[0])
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

    for i in range(1000):
        batch_a2t_audio = next(g_audio_a2t)
        batch_a2t_text = next(g_text_a2t)
        batch_t2a_audio = next(g_audio_t2a)
        batch_t2a_text = next(g_text_t2a)

        end_losses, start_losses = train_core.train(
            t2a_text=batch_t2a_text,
            t2a_audio=batch_t2a_audio,
            a2t_text=batch_a2t_text,
            a2t_audio=batch_a2t_audio,
            lr=FLAG.init_lr,
            epoch=FLAG.max_step,
            )
        print_loss(start_losses, end_losses, i)

        tmp = train_core.get_matrix()
        a2t_mat = np.reshape(np.array(tmp[0]), (dim, dim))
        t2a_mat = np.reshape(np.array(tmp[1]), (dim, dim))
    return a2t_mat, t2a_mat


def PCA_transform(text, audio):
    pca = PCA(n_components=50)
    text_pca = np.array(pca.fit_transform(text))
    audio_pca = np.array(pca.fit_transform(audio))
    return text_pca, audio_pca


def gaussian_norm(embs):
    '''
    Args:
      embs: embeddings with shape (num, dim)
    Return:
      new_emb: dimensional-wise normalized embeddings
    '''
    dim_mean = np.mean(embs, axis=0)
    dim_std = np.std(embs, axis=0)
    return (embs-dim_mean)/dim_std


def main():
    text_emb, text_labs = dp.read_csv_file(FLAG.text_embeds, ' ')
    audio_emb, audio_labs = dp.read_csv_file(FLAG.audio_embeds, ' ')
    
    text_emb = gaussian_norm(text_emb)
    audio_emb = gaussian_norm(audio_emb)
    text_pca, audio_pca = PCA_transform(text_emb, audio_emb)

    # normalization both dimension-wise
    # text_pca = gaussian_norm(text_pca)
    # audio_pca = gaussian_norm(audio_pca)
    a2t_mat, t2a_mat = ICP_train(text_pca, audio_pca)

    t2a_text = np.transpose(text_pca)
    tmp_text = np.transpose(np.matmul(t2a_mat, t2a_text))

    t2a_audio_map = generate_pair(tmp_text, audio_pca)

    a2t_audio = np.transpose(audio_pca)
    tmp_audio = np.transpose(np.matmul(a2t_mat, a2t_audio))
    a2t_text_map = generate_pair(tmp_audio, text_pca)

    text_emb = gaussian_norm(text_emb)
    audio_emb = gaussian_norm(audio_emb)

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
        default=80000,
        metavar='--<max step>',
        help='The max step for training')
    parser.add_argument(
        '--max_inner_step',
        type=int,
        default=500,
        metavar='--<max step for inner batch>',
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
