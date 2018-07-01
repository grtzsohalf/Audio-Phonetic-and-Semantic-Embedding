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
    # assume the text and audio are matched
    text_copy = np.copy(text)
    audio_copy = np.copy(audio)

    #np.random.shuffle(text_copy)
    #np.random.shuffle(audio_copy)

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
    for i in range(80000):
        batch_t2a_text = next(g_text)
        # batch_t2a_text_tr = np.transpose(batch_t2a_text)
        # batch_text = np.transpose(np.matmul(t2a_mat, batch_t2a_text_tr))
        # batch_t2a_audio = generate_pair(batch_text, audio)
        batch_a2t_audio = next(g_audio)
        batch_t2a_audio = batch_a2t_audio
        # batch_a2t_audio_tr = np.transpose(batch_a2t_audio)
        # batch_audio = np.transpose(np.matmul(a2t_mat, batch_a2t_audio_tr))
        # batch_a2t_text = generate_pair(batch_audio, text)
        batch_a2t_text = batch_t2a_text
        losses = train_core.train(
            t2a_text=batch_t2a_text,
            t2a_audio=batch_t2a_audio,
            a2t_text=batch_a2t_text,
            a2t_audio=batch_a2t_audio,
            lr=FLAG.init_lr,
            epoch=1,
            )
        if i % 100 == 0:
            print('iter {}:'.format(i))
            print(losses)
        # log_str = 'total_loss: {0}, t2a: {1}, a2t:{2}, t2a2t:{3}, a2t2a:{4}'
        # print(log_str.format(losses))
        tmp = train_core.get_matrix()
        a2t_mat = np.reshape(np.array(tmp[0]), (dim, dim))
        t2a_mat = np.reshape(np.array(tmp[1]), (dim, dim))


def icpTrainOracle(text_t2a, audio_t2a, audio_a2t, text_a2t):
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
    train_core = ICP.audio2text_ICP(
        audio_t2a.shape[1],
        text_a2t.shape[1],
        FLAG.mb,
        FLAG.penalty_lambda)
    train_core.setTextAudio(text_t2a, audio_t2a, audio_a2t, text_a2t)
    train_core.trainOracle(lr=FLAG.init_lr, epoch=FLAG.max_step,
        decay_factor=FLAG.decay_factor)
    dim = audio_t2a.shape[1]
    a2t_mat, t2a_mat = train_core.get_matrix()
    a2t_mat = np.reshape(np.array(a2t_mat), (dim, dim))
    t2a_mat = np.reshape(np.array(t2a_mat), (dim, dim))
    return a2t_mat, t2a_mat, train_core


def PCA_transform(text, audio):
    pca = PCA(n_components=100)
    text_tmp = np.array(text)
    text_norm = ( text_tmp - np.mean(text_tmp, axis=0)) / np.std(text_tmp,
        axis=0)
    text_pca = pca.fit_transform(text_norm)

    audio_tmp = np.array(audio)
    audio_norm = ( audio_tmp - np.mean(audio_tmp, axis=0)) / np.std(audio_tmp,
        axis=0)
    audio_pca = pca.fit_transform(audio_norm)

    return text_pca, audio_pca


def main():
    text_emb, text_labs = dp.read_csv_file(FLAG.text_embeds, ' ')
    audio_emb, audio_labs = dp.read_csv_file(FLAG.audio_embeds, ' ')
    text_pca, audio_pca = PCA_transform(text_emb, audio_emb)
    # # train 1000, test 3000
    # a2t_mat, t2a_mat, train_core = icpTrainOracle(text_pca[:1000][:],
    #     audio_pca[:1000][:], audio_pca[:1000][:], text_pca[:1000][:])
    # f1, f10 = train_core.calcFscore(text_pca, audio_pca)
    # print(f1, f10)
    # # train 2000, test 3000
    # a2t_mat, t2a_mat, train_core = icpTrainOracle(text_pca[:2000][:],
    #     audio_pca[:2000][:], audio_pca[:2000][:], text_pca[:2000][:])
    # f1, f10 = train_core.calcFscore(text_pca, audio_pca)
    # print(f1, f10)
    # train 3000, test 3000
    a2t_mat, t2a_mat, train_core = icpTrainOracle(text_pca,
        audio_pca, audio_pca, text_pca)
    f1, f10 = train_core.calcFscore(text_pca, audio_pca)
    print(f1, f10)

    np.save(open('a2t_mat','wb'), a2t_mat)
    np.save(open('t2a_mat','wb'), t2a_mat)


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
