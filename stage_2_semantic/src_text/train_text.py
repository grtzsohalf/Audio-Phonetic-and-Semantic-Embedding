import numpy as np
import time
import os
import argparse
from solver_text import Solver

FLAG = None

def addParser():
    parser = argparse.ArgumentParser(prog="PROG", 
        description='Audio2vec Training Script')
    parser.add_argument('--init_lr',  type=float, default=0.0005,
        metavar='<--initial learning rate>')
    parser.add_argument('--hidden_dim',type=int, default=256,
        metavar='<--hidden dimension>',
        help='The hidden dimension of a neuron')
    parser.add_argument('--batch_size',type=int, default=64,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--feat_dim',type=int, default=256,
        metavar='--<feat dim>',
        help='feature dimension')
    parser.add_argument('--gram_num',type=int, default=2,
        metavar='--<gram num>',
        help='skip gram num')
    parser.add_argument('--n_epochs',type=int, default=20,
        metavar='--<# of epochs for training>',
        help='The number of epochs for training')
    parser.add_argument('--neg_sample_num',type=int, default=5,
        metavar='--<# of negative sampling>',
        help='The number of negative sampling')
    parser.add_argument('--min_count',type=int, default=5,
        metavar='--<min count>',
        help='min count')
    parser.add_argument('--sampling_factor',type=float, default=0.0001,
        metavar='--<sampling_factor>',
        help='sampling factor')

    parser.add_argument('log_dir', 
        metavar='<log directory>')
    parser.add_argument('model_dir', 
        metavar='<model directory>')
    parser.add_argument('train_labels', 
        metavar='<training labels file>')    
    parser.add_argument('train_utters', 
        metavar='<training utters file>')    
    parser.add_argument('dic_file', 
        metavar='<dic file>')    
    return parser

def main():
    solver = Solver(FLAG.dic_file, FLAG.train_labels, FLAG.train_utters, FLAG.batch_size, FLAG.feat_dim, 
                    FLAG.gram_num, FLAG.hidden_dim, FLAG.init_lr, FLAG.log_dir, FLAG.model_dir, 
                    FLAG.n_epochs, FLAG.neg_sample_num, FLAG.min_count, FLAG.sampling_factor)
    print ("Solver constructed!")
    solver.train()

if __name__ == '__main__':
    parser = addParser()
    FLAG = parser.parse_args()
    main()


