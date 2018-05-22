import numpy as np
import time
import os
import argparse
from solver import Solver

FLAG = None

def addParser():
    parser = argparse.ArgumentParser(prog="PROG", 
        description='Audio2vec Training Script')
    parser.add_argument('--init_lr',  type=float, default=0.001,
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
    parser.add_argument('test_examples', 
        metavar='<testing example dir>')    
    parser.add_argument('test_labels', 
        metavar='<testing label dir>')    
    parser.add_argument('test_utters', 
        metavar='<testing utter dir>')    
    parser.add_argument('embedding_file', 
        metavar='<audio semantic embedding file>')    
    return parser

def main():
    solver = Solver(FLAG.test_examples, FLAG.test_labels, FLAG.test_utters, FLAG.batch_size, FLAG.feat_dim, 
                    FLAG.gram_num, FLAG.hidden_dim, FLAG.init_lr, FLAG.log_dir, FLAG.model_dir, 
                    None, FLAG.neg_sample_num, FLAG.min_count, FLAG.sampling_factor)
    print ("Solver constructed!")
    solver.test(FLAG.embedding_file)

if __name__ == '__main__':
    parser = addParser()
    FLAG = parser.parse_args()
    main()


