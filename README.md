## Note: Upgraded to TF 1.6

This is the implementation of the paper: [Towards Unsupervised Automatic Speech
Recognition Trained by Unaligned Speech and Text
only](https://arxiv.org/abs/1803.10952).

# Stage One: Disentangle

Specify the paths in the `path.sh`.

We assume that in an audio collection, each utterance is already segmented into
word-level segments. We pad all segments to the same length. Each segment is represented by the features with shape of (sequence length x feature dimension). For example, if we use MFCC features, and a word is padded to 50 time frames, the shape is (50 x 39).

To train the stage one, `cd` into `stage_1_disentangle` and run `./train.sh
[options]`.
To test and produce phonetic embeddings, `cd` into `stage_1_disentangle` and run `./test.sh [options]`.

# Stage Two: Semantic

Specify the paths in the `path.sh`.

To train the stage two, `cd` into `stage_2_semantic` and run `./train.sh
[options]`.
To produce semantic embeddings, `cd` into `stage_2_semantic` and run `./test.sh [options]`.

# Stage Three: Transform

Given two kinds of embeddings, this module is to transform one embedding to the
other.

There are two kinds of strategies here. One is referencing GAN-approach to do
so, inspired by [MUSE](https://github.com/facebookresearch/MUSE).

[TODO]
+ `audio2text_GAN.py` is the application of the first approach.

Another is the approach of iterative closest point, referencing the model of
[An Iterative Closest Point Method for Unsupervised Word
Translation](https://arxiv.org/abs/1801.06126).

[DONE]
+ `audio2text_ICP.py` and `convert_train.py` belong to the second approach.

To train, please refer to the training example: `../ICP_train.sh`.

# Evaluation of Semantic Embeddings

This is modified from [github](https://github.com/kudkudak/word-embeddings-benchmarks).

To run evalution, `cd` into `word-embeddings-benchmarks` and run `python
evaluate_correlation_on_all.py [options]`.
