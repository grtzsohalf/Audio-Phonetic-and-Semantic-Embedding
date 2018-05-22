#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script calculates embedding results against all available fast running
 benchmarks in the repository and saves results as single row csv table.

 Usage: ./evaluate_on_all -f <path to file> -o <path to output file>

 NOTE:
 * script doesn't evaluate on WordRep (nor its subset) as it is non standard
 for now and long running (unless some nearest neighbor approximation is used).

 * script is using CosAdd for calculating analogy answer.

 * script is not reporting results per category (for instance semantic/syntactic) in analogy benchmarks.
 It is easy to change it by passing category parameter to evaluate_analogy function (see help).
"""
from optparse import OptionParser
import logging
import os
from web.embeddings import fetch_GloVe, load_embedding
from web.datasets.utils import _get_dataset_dir

from web.evaluate import evaluate_correlation_on_all


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

parser = OptionParser()
parser.add_option("-f1", "--file1", dest="filename_1",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-f2", "--file2", dest="filename_2",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-p", "--format", dest="format",
                  help="Format of the embedding, possible values are: word2vec, word2vec_bin, dict and glove.",
                  default='glove')

parser.add_option("-o", "--output", dest="output",
                  help="Path where to save results.",
                  default=None)

parser.add_option("-c", "--clean_words", dest="clean_words",
                  help="Clean_words argument passed to load_embedding function. If set to True will remove"
                       "most of the non-alphanumeric characters, which should speed up evaluation.",
                  default=False)

if __name__ == "__main__":
    (options, args) = parser.parse_args()

    # Load embeddings
    fname_1 = options.filename_1
    fname_2 = options.filename_2
    if not fname_1 or not fname_2:
        w_1 = fetch_GloVe(corpus="wiki-6B", dim=300)
        w_2 = fetch_GloVe(corpus="wiki-6B", dim=300)
    else:
        if not os.path.isabs(fname_1) or not os.path.isabs(fname_2):
            fname_1 = os.path.join(_get_dataset_dir(), fname_1)
            fname_2 = os.path.join(_get_dataset_dir(), fname_2)

        format = options.format

        # if not format:
            # _, ext = os.path.splitext(fname)
            # if ext == ".bin":
                # format = "word2vec_bin"
            # elif ext == ".txt":
                # format = "word2vec"
            # elif ext == ".pkl":
                # format = "dict"

        assert format in ['word2vec_bin', 'word2vec', 'glove', 'bin'], "Unrecognized format"

        load_kwargs_1 = {}
        if format == "glove":
            load_kwargs_1['vocab_size'] = sum(1 for line in open(fname_1))
            load_kwargs_1['dim'] = len(next(open(fname_1)).split()) - 1

        w_1 = load_embedding(fname_1, format=format, normalize=True, lower=True, clean_words=options.clean_words,
                           load_kwargs=load_kwargs_1)

        load_kwargs_2 = {}
        if format == "glove":
            load_kwargs_2['vocab_size'] = sum(1 for line in open(fname_2))
            load_kwargs_2['dim'] = len(next(open(fname_2)).split()) - 1

        w_2 = load_embedding(fname_1, format=format, normalize=True, lower=True, clean_words=options.clean_words,
                           load_kwargs=load_kwargs_2)

    out_fname = options.output if options.output else "results.csv"

    results = evaluate_correlation_on_all(w_1, w_2)

    logger.info("Saving results...")
    print(results)
    results.to_csv(out_fname)
