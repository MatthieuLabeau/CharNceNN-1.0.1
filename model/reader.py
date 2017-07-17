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

def load_w2v(w2v_file, vocab):
    print("Load word2vec file {}\n".format(w2v_file))
    with open(w2v_file, "r") as f:
        header = f.readline()
        vocab_length_r, emb_dim = map(int, header.split())
        initEmb = np.random.uniform(-1.0, 1.0, size = (max(vocab.values())+1, emb_dim))
        for line in xrange(vocab_length_r):
            vec = f.readline().split(' ', 1)
            word = vec[0]
            values = np.array(vec[1].split(), dtype='float32')
            idx = vocab.get(word)
            if idx != 0:
                initEmb[idx] = values
        print initEmb.shape
    return initEmb
                
def _filter_vocab_tags(words_vocab, tags_words_map, tags_list, freq = 0):
    words_bool = dict(zip(sorted(words_vocab.keys(), key=words_vocab.get), [True] * freq + [False]*(len(words_vocab) - freq)))
    words_true = []
    for tag in tags_list:
        words_true.extend(tags_words_map[tag])
    for w in words_true:
        words_bool[w] = True
    words = [w for w in sorted(words_vocab.keys(), key=words_vocab.get) if words_bool[w]]
    voc = dict(zip(words, range(2, len(words)+2)))
    vocs_map = [voc.get(w,1) for w in sorted(words_vocab.keys(), key=words_vocab.get)]
    return voc, vocs_map

def _filter_vocab(vocab, threshold):  
    if threshold == 0:
        return vocab
    else:
        return dict((k, v) for k, v in vocab.iteritems() if v < threshold + 4)

def _filter_vocab_lems(lems_vocab, words_vocab, lems_words_map):
    words_bool = dict(zip(words_vocab.keys(), [False]*len(words_vocab)))
    words_true = []
    for lems in sorted(lems_vocab.keys(), key=lems_vocab.get):
        words_true.extend(lems_words_map[lems])
    for w in words_true:
        words_bool[w] = True
    words = [w for w in sorted(words_vocab.keys(), key=words_vocab.get) if words_bool[w]]
    voc = dict(zip(words, range(2, len(words)+2)))
    vocs_map = [voc.get(w,1) for w in sorted(words_vocab.keys(), key=words_vocab.get)]
    return voc, vocs_map
    
def _filter_tags_toks_map(dic_tags_toks, words_vocab, tags_vocab, max_tag_number):

    for w in dic_tags_toks.keys():
        if len(dic_tags_toks[w]) > 1:
            tags_toks = dic_tags_toks[w][0]
            for t in range(max_tag_number):
                tags_toks[t] = _most_common([dic_tags_toks[w][i][t] for i in range(len(dic_tags_toks[w]))])                          
            dic_tags_toks[w] = tags_toks
        else:
            dic_tags_toks[w] = dic_tags_toks[w][0]
    
    wid_to_tagsid = {i: np.ndarray(shape=(max_tag_number,),
                                    buffer=np.array([tags_vocab[j].get(t,1) for j, t in enumerate(dic_tags_toks[w])]),
                                    dtype=int) for w, i in words_vocab.iteritems()}
    wid_to_tagsid[0] = [0]*max_tag_number
    wid_to_tagsid[1] = [1]*max_tag_number
    return wid_to_tagsid

def _most_common(lst):
    data = collections.Counter(lst)
    return data.most_common(1)[0][0]
        
class datasetQ():
    def __init__(self,
                 dir_path,
                 data_file,
                 vocab_file,
                 reps,
                 max_seq_length = None,
                 max_word_length = None,
                 max_tag_number = 12,
                 map_vocab_threshold = 0,
                 emb_vocab_threshold = 0,
                 char_vocab_threshold = 0,
                 tags_list = [],
                 w2v_init = False):
        """
        Caution: this iterator will open the file and process lines on the fly before
        yielding them, (to be able to work with files too big to fit in memory
        which implies there is no data shuffling. Training data must be shuffled using:
        shuf --head-count=NB_SEQ train
        """
        self.lem_path = os.path.join(dir_path, data_file + '.lem')
        self.tag_path = os.path.join(dir_path, data_file + '.fact')
        self.path = os.path.join(dir_path, data_file + '.tok')
        self.vocab_data_path = os.path.join(dir_path, vocab_file + '.new.vocab.pkl')
        self.w2v_init = w2v_init
        if w2v_init:
            self.w2v_path = os.path.join(dir_path, data_file + '.tok.vecs')
                    
        self.reps = reps
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
        self.max_tag_number = max_tag_number
        
        # If the vocabulary is not already saved:
        if not os.path.exists(self.vocab_data_path):
            self._file_to_vocab()

        with open(self.vocab_data_path, 'r') as _file:
            [word_to_id, word_counts, char_to_id, lems_to_id, lem_counts, tags_to_id_l, dic_toks_lems, dic_lems_toks, dic_toks_tags, dic_tags_toks, tot] = pickle.load(_file)

        if self.w2v_init:
            self.initEmb = load_w2v(self.w2v_path, word_to_id)            
            
        # Prepare the necessary vocabularies according to chosen representations
        # Lemmes
        self.lems_to_id = _filter_vocab(lems_to_id, emb_vocab_threshold)
            
        # Tags
        self.tags_to_id_l = [ _filter_vocab(tags_to_id, 0) for tags_to_id in tags_to_id_l]

        # Words
        if self.reps[8]:
            self.word_to_id, self.eval_to_word = _filter_vocab_lems(self.lems_to_id, word_to_id, dic_toks_lems)
        else:
            if len(tags_list) > 0:
                self.word_to_id, self.eval_to_word = _filter_vocab_tags(word_to_id, dic_toks_tags, tags_list, emb_vocab_threshold)
            else:
                self.word_to_id = _filter_vocab(word_to_id, emb_vocab_threshold)
                self.eval_to_word = range(2, len(self.word_to_id)+2) + [1] * (len(word_to_id) - len(self.word_to_id))

        # Evaluation: Lemmes ou mots du corpus d'entrainement:
        if self.reps[8]:
            self.eval_word_to_id = _filter_vocab(lems_to_id, map_vocab_threshold)
            self.unigram = [1]*4 + list(lem_counts[:(len(self.eval_word_to_id)-2)])
            self.uniform = [1]*(len(self.eval_word_to_id)+2)
        else:
            self.eval_word_to_id = _filter_vocab(word_to_id, map_vocab_threshold)
            self.unigram = [1]*4 + list(word_counts[:(len(self.eval_word_to_id)-2)])
            self.uniform = [1]*(len(self.eval_word_to_id)+2)
                                                    
        # Characters
        self.char_to_id = _filter_vocab(char_to_id, char_vocab_threshold)
        
        # Mots/Lemmes -> char map
        if self.reps[6]:
            self.wid_to_charid = {i: np.ndarray(shape=(self.max_word_length,),
                                                buffer=np.array([2] + [self.char_to_id.get(c,1) for j, c in enumerate(w) if j < (self.max_word_length - 2)] + [3] + [0]*(self.max_word_length - len(w) - 2)),
                                                dtype=int) for w, i in self.eval_word_to_id.iteritems()}
            self.wid_to_charid[0] = [2] + [3] + [0]*(self.max_word_length - 2)
            self.wid_to_charid[1] = [2] + [1] + [3] + [0]*(self.max_word_length - 3)
            self.wid_to_charid = collections.OrderedDict(sorted(self.wid_to_charid.items(), key = lambda t: t[0]))
            self.wid_to_charid = np.asarray(self.wid_to_charid.values())

        # Lemmes map
        if self.reps[7] and self.reps[5] and not self.reps[8]:
            self.wid_to_lemsid = {i: self.lems_to_id.get(dic_lems_toks[w],1) for w, i in self.eval_word_to_id.iteritems()}
            self.wid_to_lemsid = collections.OrderedDict(sorted(self.wid_to_lemsid.items(), key = lambda t: t[0]))
            self.wid_to_lemsid = list(self.wid_to_lemsid.values())
            
        # Tags map        
        if self.reps[7] and not self.reps[8]:
            self.wid_to_tagsid = _filter_tags_toks_map(dic_tags_toks, self.eval_word_to_id, self.tags_to_id_l, self.max_tag_number)
            self.wid_to_tagsid = collections.OrderedDict(sorted(self.wid_to_tagsid.items(), key = lambda t: t[0]))
            self.wid_to_tagsid = np.asarray(self.wid_to_tagsid.values())
            
        # Nombre total d'exemples d'entrainement
        self.tot = tot        

    def _file_to_vocab(self):
        # Get words, find longuest sentence and longuest word
        data = []
        data_lems = []
        data_tags = []
        dic_toks_lems = collections.defaultdict(set)
        dic_toks_lems['sos'].add('sos')
        dic_toks_lems['eos'].add('eos')
        dic_toks_tags = collections.defaultdict(set)
        dic_toks_tags['('].add('eos')
        dic_toks_tags[')'].add('sos')
        dic_tags_toks = collections.defaultdict(list)
        dic_tags_toks['sos'].append(['(']*self.max_tag_number)
        dic_tags_toks['eos'].append([')']*self.max_tag_number)
        dic_lems_toks = {}
        dic_lems_toks['sos'] = 'sos'
        dic_lems_toks['eos'] = 'eos'
        with open(self.path) as train_file:
            with open(self.lem_path) as lem_file:
                with open(self.tag_path) as tag_file:
                    for lems, tags, toks in zip(lem_file, tag_file, train_file):
                        seq_lems = lems.strip().split()
                        seq_tags = tags.strip().split()
                        seq_toks = toks.strip().split()
                        if not (True in [(len(tag_mute) < self.max_tag_number) for tag_mute in seq_tags]):
                            data.append(seq_toks)
                            data_lems.append(seq_lems)
                            data_tags.append(seq_tags)
                            for lem, tok, tag in zip(seq_lems, seq_toks, seq_tags):
                                dic_toks_lems[lem].add(tok)
                                dic_toks_tags[tag[0]].add(tok)
                                dic_tags_toks[tok].append(list(tag))
                                dic_lems_toks[tok] = lem
        # Get complete word and characters vocabulary
        word_counter = collections.Counter([word for seq in data for word in seq])
        word_pairs = sorted(word_counter.items(), key=lambda x: -x[1])
        words, w_counts = list(zip(*word_pairs))
        word_to_id = dict(zip(words, range(4, len(words)+4)))
        word_to_id['sos'] = 2
        word_to_id['eos'] = 3                 
        
        lem_counter = collections.Counter([lem for seq in data_lems for lem in seq])
        lem_pairs = sorted(lem_counter.items(), key=lambda x: -x[1])
        lems, l_counts = list(zip(*lem_pairs))
        lems_to_id = dict(zip(lems, range(4, len(lems)+4)))
        lems_to_id['sos'] = 2
        lems_to_id['eos'] = 3
        
        char_counter = collections.Counter([char for seq in data for word in seq for char in word])
        char_pairs = sorted(char_counter.items(), key=lambda x: -x[1])
        chars, c_counts = list(zip(*char_pairs))
        char_to_id = dict(zip(chars, range(4, len(chars)+4)))

        tags_to_id_l = []
        for i in range(self.max_tag_number):
            tags_counter = collections.Counter([tags[i] for seq in data_tags for tags in seq])
            tags_pairs = sorted(tags_counter.items(), key=lambda x: -x[1])
            tags, _ = list(zip(*tags_pairs))
            tags_to_id = dict(zip(tags, range(4, len(tags)+4)))
            tags_to_id['('] = 2
            tags_to_id[')'] = 3
            tags_to_id_l.append(tags_to_id)
            
        with open(self.vocab_data_path, 'w') as vocab_file:
            pickle.dump([word_to_id, w_counts, char_to_id, lems_to_id, l_counts, tags_to_id_l, dic_toks_lems, dic_lems_toks, dic_toks_tags, dic_tags_toks, len(data)], vocab_file)
            
    def sampler_seq(self, batch_size):
        with open(self.lem_path) as lem_file:
            with open(self.tag_path) as tag_file:
                with open(self.path) as _file:
                    while True:
                        lem_to_be_read =  list(islice(lem_file, batch_size))
                        tag_to_be_read = list(islice(tag_file, batch_size))
                        to_be_read = list(islice(_file, batch_size))
                        if len(lem_to_be_read) < batch_size:
                            lem_file.seek(0)
                            tag_file.seek(0)
                            _file.seek(0)
                            lem_to_be_read =  list(islice(lem_file, batch_size))
                            tag_to_be_read = list(islice(tag_file, batch_size))
                            to_be_read = list(islice(_file, batch_size))


                        if self.reps[0] or self.reps[4]:
                            word_tensor = np.ndarray(shape = (batch_size, self.max_seq_length+1), dtype=int)
                        if self.reps[1] or self.reps[5]:
                            lems_tensor = np.ndarray(shape = (batch_size, self.max_seq_length+1), dtype=int)
                        if self.reps[2] or (self.reps[6] and not self.reps[8]):
                            char_tensor = np.zeros(shape = (batch_size, self.max_seq_length+1, self.max_word_length), dtype=int)
                        if self.reps[6] and self.reps[8]:
                            char_lems_tensor = np.zeros(shape = (batch_size, self.max_seq_length+1, self.max_word_length), dtype=int)
                        if self.reps[3] or self.reps[7]:
                            tags_tensor = np.zeros(shape = (batch_size, self.max_seq_length+1, self.max_tag_number), dtype=int)
                        
                            
                        eval_tensor = np.ndarray(shape = (batch_size, self.max_seq_length+1), dtype=int)

                        for id_seq, (lems, tags, line) in enumerate(zip(lem_to_be_read, tag_to_be_read, to_be_read)):
                            lems_seq = ['sos'] + lems.strip().split() + ['eos']
                            seq = ['sos'] + line.strip().split() + ['eos']
                            tags_seq = ['('] * self.max_tag_number + tags.strip().split() + [')'] * self.max_tag_number
                            # Mots et/ou Lemmes
                            if self.reps[0] or self.reps[4]:
                                word_tensor[id_seq]=np.ndarray(shape=(self.max_seq_length+1,),
                                                               buffer=np.array([self.word_to_id.get(w, 1) for w in seq] + [0]*(self.max_seq_length + 1 - len(seq))),
                                                               dtype=int)
                            if self.reps[1] or self.reps[5]:
                                lems_tensor[id_seq]=np.ndarray(shape=(self.max_seq_length+1,),
                                                               buffer=np.array([self.lems_to_id.get(l, 1) for l in lems_seq] + [0]*(self.max_seq_length + 1 - len(lems_seq))),
                                                               dtype=int)
                            # Caracteres
                            if self.reps[2] or (self.reps[6] and not self.reps[8]):
                                for id_word, word in enumerate(seq):
                                    if id_word < self.max_seq_length:
                                        char_tensor[id_seq,id_word]=np.ndarray(shape=(self.max_word_length,),
                                                                               buffer=np.array([2] + [self.char_to_id.get(c,1) for j, c in enumerate(word) if j < (self.max_word_length - 2)] + [3] + [0]*(self.max_word_length - len(word) - 2)),
                                                                               dtype=int)                                
                            # Caracteres des Lemmes            
                            if self.reps[6] and self.reps[8]:
                                for id_lem, lem in enumerate(lems_seq):
                                    if id_lem < self.max_seq_length:
                                        char_lems_tensor[id_seq, id_lem]=np.ndarray(shape=(self.max_word_length,),
                                                                                    buffer=np.array([2] + [self.char_to_id.get(c,1) for j, c in enumerate(lem) if j < (self.max_word_length - 2)] + [3] + [0]*(self.max_word_length - len(lem) - 2)),
                                                                                    dtype=int)
                                        
                                        
                            # Tags
                            if self.reps[3] or self.reps[7]:
                                for id_tag, tag in enumerate(tags_seq):
                                    if id_tag < self.max_seq_length:
                                        tags_tensor[id_seq, id_tag]=np.ndarray(shape=(self.max_tag_number,),
                                                                               buffer=np.array([self.tags_to_id_l[j].get(t,1) for j, t in enumerate(tag)] + [1]*(self.max_tag_number - len(tag))),
                                                                               dtype=int)
                            # Labels - ajouter version Lemmes !
                            if self.reps[8]:
                                eval_seq = lems_seq
                            else:
                                eval_seq = seq
                            eval_tensor[id_seq]=np.ndarray(shape=(self.max_seq_length+1,),
                                                           buffer=np.array([self.eval_word_to_id.get(w, 1) for w in eval_seq] + [0]*(self.max_seq_length + 1 - len(eval_seq))),
                                                           dtype=int)
                                    
                        out = []
                        # Inputs:
                        if self.reps[0]:
                            out.append(word_tensor[:,:-1])
                        elif self.reps[1]:
                            out.append(lems_tensor[:,:-1])
                        if self.reps[2]:
                            out.append(char_tensor[:,:-1,:])
                        if self.reps[3]:
                            out.append(tags_tensor[:,:-1,:])
                            
                        # Outputs
                        if self.reps[4]:
                            out.append(word_tensor[:,1:])
                        elif self.reps[5]:
                            out.append(lems_tensor[:,1:])
                        if self.reps[6]:
                            if self.reps[8]:
                                out.append(char_lems_tensor[:,1:,:])
                            else:
                                out.append(char_tensor[:,1:,:])
                        # Labels
                        if self.reps[7]:
                            out.append(tags_tensor[:,1:,:])
                        out.append(eval_tensor[:,1:])
                            
                        yield out

