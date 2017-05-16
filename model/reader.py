# Some inspiration from https://github.com/carpedm20/lstm-char-cnn-tensorflow/
## ===========================================================================
from __future__ import division

import collections
import os
import sys
import time
import random
from itertools import islice

import numpy as np
import tensorflow as tf

import pickle
from nltk.util import ngrams
from tensorflow.python.platform import gfile

"""
TODO: Add beginning of sentence and end of sentences words - with their character decomposition 
"""

def _filter_vocab(vocab, threshold):  
    if threshold == 0:
        return vocab
    else:
        return dict((k, v) for k, v in vocab.iteritems() if v < threshold + 4)

class datasetQ():
    def __init__(self,
                 dir_path,
                 data_file,
                 vocab_file,
                 reps,
                 max_seq_length = None,
                 context_length = None,
                 max_word_length = None,
                 map_vocab_threshold = 0,
                 word_vocab_threshold = 0,
                 char_vocab_threshold = 0):
        """
        Caution: this iterator will open the file and process lines on the fly before
        yielding them, (to be able to work with files too big to fit in memory
        which implies there is no data shuffling. Training data must be shuffled using:
        shuf --head-count=NB_SEQ train
        """
        self.path = os.path.join(dir_path, data_file)
        self.vocab_data_path = os.path.join(dir_path, vocab_file + '.vocab.pkl')
        self.reps = reps

        # If the vocabulary is not already saved:
        if not os.path.exists(self.vocab_data_path):
            self._file_to_vocab()

        with open(self.vocab_data_path, 'r') as _file:
            [word_to_id, word_counts, char_to_id, tot] = pickle.load(_file)

        self.eval_word_to_id = _filter_vocab(word_to_id, map_vocab_threshold)
        self.word_to_id = _filter_vocab(word_to_id, word_vocab_threshold)
        self.char_to_id = _filter_vocab(char_to_id, char_vocab_threshold)
        
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length

        self.wid_to_charid = {i: np.ndarray(shape=(self.max_word_length,),
                                            buffer=np.array([2] + [self.char_to_id.get(c,1) for j, c in enumerate(w) if j < (self.max_word_length - 2)] + [3] + [0]*(self.max_word_length - len(w) - 2)),
                                            dtype=int) for w, i in self.eval_word_to_id.iteritems()}
        #[2] + [self.char_to_id.get(c,1) for j, c in enumerate(w) if j < (self.max_word_length - 2)] + [3] + [0]*(self.max_word_length - len(w) - 2)
        self.wid_to_charid[0] = [2] + [3] + [0]*(self.max_word_length - 2)
        self.wid_to_charid[1] = [2] + [1] + [3] + [0]*(self.max_word_length - 3)
        self.wid_to_charid[2] = [2] + [2] + [3] + [0]*(self.max_word_length - 3)
        self.wid_to_charid[3] = [2] + [3] + [3] + [0]*(self.max_word_length - 3)
        self.wid_to_charid = collections.OrderedDict(sorted(self.wid_to_charid.items(), key = lambda t: t[0]))
        self.wid_to_charid = np.asarray(self.wid_to_charid.values())

        self.uniform = [1]*4 + [1]*word_vocab_threshold
        self.unigram = [1]*4 + list(word_counts[:word_vocab_threshold])

        self.tot = tot        

    def _file_to_vocab(self):
        # Get words, find longuest sentence and longuest word
        data = []
        temp_max_seq_length = 0
        temp_max_word_length = 0
        with open(self.path) as train_file:
            for line in train_file:
                seq = line.strip().split()
                temp_max_seq_length = max([temp_max_seq_length, len(seq)])
                temp_max_word_length = max([temp_max_seq_length] + [len(word) for word in seq])
                data.append(seq)

        # Get complete word and characters vocabulary
        word_counter = collections.Counter([word for seq in data for word in seq])
        word_pairs = sorted(word_counter.items(), key=lambda x: -x[1])
        words, w_counts = list(zip(*word_pairs))
        word_to_id = dict(zip(words, range(4, len(words)+4)))

        char_counter = collections.Counter([char for seq in data for word in seq for char in word])
        char_pairs = sorted(char_counter.items(), key=lambda x: -x[1])
        chars, c_counts = list(zip(*char_pairs))
        char_to_id = dict(zip(chars, range(4, len(chars)+4)))

        with open(self.vocab_data_path, 'w') as vocab_file:
            pickle.dump([word_to_id, w_counts, char_to_id, len(data)], vocab_file)

    def sampler_ngrams(self, n, batch_size):
        with open(self.path) as _file:
            while True:
                to_be_read = list(islice(_file, batch_size))
                if len(to_be_read) < batch_size:
                    _file.seek(0)
                    to_be_read = list(islice(_file, batch_size))
                """
                TODO: one loop for word/char faster than two loops with list comprehensions ? 
                """
                n_grams = [ngrams(sent.strip().split(),n) for sent in to_be_read]
                if self.reps[0] or self.reps[2]:
                    word_train_tensor = np.array([[self.word_to_id.get(w, 1) for w in n_gram] for sent in n_grams for n_gram in sent], dtype = 'int32')
                if self.reps[1] or self.reps[3]:
                    char_train_tensor=np.array([[np.ndarray(shape=(self.max_word_length,),
                                                            buffer=[2] + np.array([self.char_to_id.get(c,1) for c in word] + [3] + [0]*(self.max_word_length - len(word) - 2)),
                                                            dtype=int) for word in n_gram] for sent in n_grams for n_gram in sent], dtype = 'int32')    
                eval_tensor = np.array([[self.eval_word_to_id.get(w, 1) for w in n_gram] for sent in n_grams for n_gram in sent], dtype = 'int32')
                out = []
                if self.reps[0]:
                    x = word_train_tensor[:,:-1]
                    out.append(x)
                if self.reps[1]:
                    c = char_train_tensor[:,:-1,:]
                    out.append(c)
                if self.reps[2]:
                    y = word_train_tensor[:,-1]
                    out.append(y)
                if self.reps[3]:
                    yc = char_train_tensor[:,-1,:]
                    out.append(yc)
                e = eval_tensor[:,1:]
                out.append(e)
                yield out

    def sampler_seq(self, batch_size):
        with open(self.path) as _file:
            while True:
                to_be_read = list(islice(_file, batch_size))
                if len(to_be_read) < batch_size:
                    _file.seek(0)
                    to_be_read = list(islice(_file, batch_size))
                if self.reps[0] or  self.reps[2]:
                    word_train_tensor = np.ndarray(shape = (batch_size, self.max_seq_length+1), dtype=int)
                if self.reps[1] or  self.reps[3]:
                    char_train_tensor = np.zeros(shape = (batch_size, self.max_seq_length+1, self.max_word_length), dtype=int)
                eval_tensor = np.ndarray(shape = (batch_size, self.max_seq_length+1), dtype=int)
                for id_seq, line in enumerate(to_be_read):
                    seq = line.strip().split()
                    if self.reps[0] or self.reps[2]:
                        word_train_tensor[id_seq]=np.ndarray(shape=(self.max_seq_length+1,),
                                                             buffer=np.array([self.word_to_id.get(w, 1) for w in seq] + [0]*(self.max_seq_length + 1 - len(seq))),
                                                             dtype=int)
                    if self.reps[1] or self.reps[3]:
                        for id_word, word in enumerate(seq):
                            if id_word < self.max_seq_length:
                                char_train_tensor[id_seq,id_word]=np.ndarray(shape=(self.max_word_length,),
                                                                             buffer=np.array([2] + [self.char_to_id.get(c,1) for j, c in enumerate(word) if j < (self.max_word_length - 2)] + [3] + [0]*(self.max_word_length - len(word) - 2)),
                                                                             dtype=int)                                
                    eval_tensor[id_seq]=np.ndarray(shape=(self.max_seq_length+1,),
                                                   buffer=np.array([self.eval_word_to_id.get(w, 1) for w in seq] + [0]*(self.max_seq_length + 1 - len(seq))),
                                                   dtype=int)
                out = []
                if self.reps[0]:
                    x = word_train_tensor[:,:-1]
                    out.append(x)
                if self.reps[1]:
                    c = char_train_tensor[:,:-1,:]
                    out.append(c)
                if self.reps[2]:
                    y = word_train_tensor[:,1:]
                    out.append(y)
                if self.reps[3]:
                    yc = char_train_tensor[:,1:,:]
                    out.append(yc)
                e = eval_tensor[:,1:]
                out.append(e)
                yield out

