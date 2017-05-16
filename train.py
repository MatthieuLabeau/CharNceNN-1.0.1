import sys
import os
import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

#from model.model import LM
from model.runner import datarunner
from model.reader import datasetQ
from model.model import LM

#Define the class for the model Options
class Options(object):
  def __init__(self):

    #Structural choices
    """ Choose input/output representations: word, character"""
    self.reps = [False, True, False, True]
    self.max_word_length = 15
    self.max_seq_length = 30
    self.context_length = None
    self.highway_layers = 2
    self.hidden_layers = 2
    self.output_highway_layers = 0
    self.batch_norm = True
    self.reuse_character_layer = True
    self.positionEmbeddings = False
    self.wordCharGate = False
    self.charLayer = 'concat'
    
    # Vocabulary
    self.evaluation_vocab_size = 100000
    self.word_vocab_size = 100000 # Must be smaller than evaluation_vocab_size
    
    #Data
    self.path = "/vol/work2/labeau/lm/data/cs/TED/"
    self.train = 'train'
    self.test = 'test'
    self.batch_size = 512
    self.train_set = datasetQ(dir_path=self.path,
                              data_file=self.train,
                              vocab_file=self.train,
                              reps = self.reps,
                              max_seq_length = self.max_seq_length,
                              context_length = self.context_length,
                              max_word_length = self.max_word_length,
                              map_vocab_threshold = self.evaluation_vocab_size,
                              word_vocab_threshold = self.word_vocab_size,
                              char_vocab_threshold = 0)
    self.test_set = datasetQ(dir_path=self.path,
                             data_file=self.test,
                             vocab_file=self.train,
                             reps = self.reps,
                             max_seq_length = self.max_seq_length,
                             context_length = self.context_length,
                             max_word_length = self.max_word_length,
                             map_vocab_threshold = self.evaluation_vocab_size,
                             word_vocab_threshold = self.word_vocab_size,
                             char_vocab_threshold = 0)                    
    self.vocab_size = max(self.train_set.word_to_id.values())+1
    self.eval_vocab_size = max(self.train_set.eval_word_to_id.values())+1
    self.char_vocab_size = max(self.train_set.char_to_id.values())+1
    self.train_size = self.train_set.tot
    self.test_size = 2773
    self.n_training_steps = self.train_size // self.batch_size
    self.n_testing_steps = self.test_size // self.batch_size
    self.training_sub = 20

    #Structural Hyperparameters
    self.c_emb_dim = 20
    self.w_emb_dim = 150
    self.w_emb_out_dim = 150
    """ Both next lists must have the same length - final 
    dimension of the word embedding built from characters
    will be the sum of filter dimensions """
    self.window_sizes = [3, 5, 7]
    self.filter_dims = [30, 50, 70]
    self.charLSTM_dim = 50
    if self.charLayer == 'conv':
      self.emb_dim = int(self.reps[0])*self.w_emb_dim + int(self.reps[1])*sum(self.filter_dims)
      self.hidden_dim = int(self.reps[2])*self.w_emb_out_dim + int(self.reps[3])*sum(self.filter_dims)
    elif self.charLayer == 'LSTM':
      self.emb_dim = int(self.reps[0])*self.w_emb_dim + int(self.reps[1])*2*self.charLSTM_dim
      self.hidden_dim = int(self.reps[2])*self.w_emb_out_dim + int(self.reps[3])*2*self.charLSTM_dim
    else:
      self.emb_dim = int(self.reps[0])*self.w_emb_dim + int(self.reps[1])*self.c_emb_dim*self.max_word_length
      self.hidden_dim = int(self.reps[2])*self.w_emb_out_dim + int(self.reps[3])*self.c_emb_dim*self.max_word_length
    if self.wordCharGate:
      self.hidden_dim = self.hidden_dim / 2

    #Training Hyperparameters
    self.learning_rate = 0.005
    self.lr_decay = 0.9
    self.epochs = 200
    self.dropout = 0.5
    self.reg = 0.000

    #Objective: 'nce', 'target', else MLE
    self.obj = 'target'
    #Noise: for nce: 'unigram', 'uniform'
    self.noiseBool = (self.obj == 'nce' or self.obj == 'target' or self.obj == 'blackOut')
    self.noise = 'unigram'
    if self.noise == 'unigram':
      self.noiseDistrib = self.train_set.unigram
    elif self.noise == 'uniform':
      self.noiseDistrib= self.train_set.uniform
    self.distortion = 0.25
    self.unique = False
    self.k = self.batch_size * self.max_seq_length
    #self.k = 500
    self.batched_noise = False
    self.noise_length = int(self.noiseBool) * self.k * ( 1 + int(self.batched_noise) * (self.batch_size * self.max_seq_length - 1))

    #Others
    self.save_path = "saves/"
    self.display_step = 5

  def decay(self):
    self.learning_rate = self.learning_rate * self.lr_decay

#opts = Options()
with tf.Graph().as_default(), tf.Session(
    config=tf.ConfigProto(
      inter_op_parallelism_threads=16,
      intra_op_parallelism_threads=16)) as session: 

  print('Preprocessing: Building options')
  opts = Options()
  print opts.train_set.wid_to_charid.shape
  print("Vocab size: %i" % (opts.vocab_size,))
  print("Character vocab size: %i" % (opts.char_vocab_size,))

  print('Preprocessing: Creating data queue')
  train_runner = datarunner(opts.reps,
                            opts.max_seq_length,
                            opts.max_seq_length,
                            opts.max_word_length)
  test_runner = datarunner(opts.reps,
                           opts.max_seq_length,
                           opts.max_seq_length,
                           opts.max_word_length)
  train_inputs = train_runner.get_inputs(opts.batch_size)
  test_inputs = test_runner.get_inputs(opts.batch_size)
  print('Preprocessing: Creating model')
  with tf.variable_scope("model"):
    model = LM(opts, session, train_inputs)
  with tf.variable_scope("model", reuse=True):
    model_eval = LM(opts, session, test_inputs, training=False)
  tf.initialize_all_variables().run()
  print('Initialized !')

  tf.train.start_queue_runners(sess=session)
  train_gen = opts.train_set.sampler_seq(opts.batch_size)
  test_gen = opts.test_set.sampler_seq(opts.batch_size)
  train_runner.start_threads(session, train_gen)
  test_runner.start_threads(session, test_gen)

  timeFile = str(datetime.now()).replace(' ','_').replace(':','-').replace('.','_')
  results_file = open('./logs/log' + timeFile,'w')
  opts_attr = vars(opts)
  for attr in opts_attr.items():
    if not (str(attr[0]) == 'noiseDistrib'):
      results_file.write(str(attr[0]) + ' : ' + str(attr[1]) + '\n')

  print ("Epoch 0:")
  results_file.write(str(0) + '\n')
  model_eval.call(results_file)
  for ep in xrange(opts.epochs):
    print ("Epoch %i :" % (ep+1))
    results_file.write(str(ep+1) + '\n')
    model.call(results_file)
    model_eval.call(results_file)
    model._options.decay()
  results_file.close()

  
