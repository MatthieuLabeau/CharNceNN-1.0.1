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

    self.name = 'W2VTestWCharCNN'
    #Structural choices
    """
    Permet le choix des representations utilisees:
    - Embeddings de mots en entree
    - Embeddings de Lemmes en entree (mutuellement exclusif)
    - Representations a base de caracteres en entree
    - Information sur les tags en entree

    - Embeddings de mots en sortie
    - Embeddings de lemmes en sortie (mutuellement exclusif)
    - Representations a base de caracteres en sortie, a scorer 
    
    - Ajouter a l'objectif la prediction des tags en sortie, par maximum de vraisemblance
    - Indique que l'on score le lemme a la place du mot. Si embedding de mot il y a, il doit etre du meme type
    """
    self.reps = [False, True, False, True, True, False, False, False, False]
    self.max_word_length = 15
    self.max_seq_length = 30
    self.max_tag_number = 12
    self.highway_layers = 2
    self.hidden_layers = 2
    self.output_highway_layers = 0
    self.batch_norm = True
    self.reuse_character_layer = False
    self.charLayer = 'conv'
    self.tagLayer = 'LSTM'

    self.freqLoss = True
    self.ranges = [0, 25295, 159146]
    
    # Vocabulary
    self.emb_vocab_size = 0
    self.evaluation_vocab_size = 0 # Doit etre precise imperativement, que le vocabulaire de sortie soit de lemmes ou de mots
    # Si les vocabulaires d'entree et de sortie sont de meme type, celui d'entree sera egal au vocabulaire de sortie.
    # Si c'est un vocabulaire de mots et que des lemmes sont en sortie, il sera adapte pour que les mots dont les lemmes sont OOV soient OOV.
    self.tags_list = []  #['C', 'D', 'I', 'J', 'P', 'R', 'T', 'Z', 'X']
    
    #Data
    self.path = "./data/newsco/"
    self.train = 'train.news-commentary.en2cs.cs'
    self.valid = 'valid.news-commentary.en2cs.cs'
    self.test = 'test.news-commentary.en2cs.cs'
    self.batch_size = 128
    self.train_set = datasetQ(dir_path=self.path,
                              data_file=self.train,
                              vocab_file=self.train,
                              reps = self.reps,
                              max_seq_length = self.max_seq_length,
                              max_word_length = self.max_word_length,
                              max_tag_number = self.max_tag_number,
                              map_vocab_threshold = self.evaluation_vocab_size,
                              emb_vocab_threshold = self.emb_vocab_size,
                              char_vocab_threshold = 0,
                              tags_list = self.tags_list,
                              w2v_init  = True)
    self.valid_set = datasetQ(dir_path=self.path,
                              data_file=self.valid,
                              vocab_file=self.train,
                              reps = self.reps,
                              max_seq_length = self.max_seq_length,
                              max_word_length = self.max_word_length,
                              max_tag_number = self.max_tag_number,
                              map_vocab_threshold = self.evaluation_vocab_size,
                              emb_vocab_threshold = self.emb_vocab_size,
                              char_vocab_threshold = 0,
                              tags_list = self.tags_list)
    self.test_set = datasetQ(dir_path=self.path,
                             data_file=self.test,
                             vocab_file=self.train,
                             reps = self.reps,
                             max_seq_length = self.max_seq_length,
                             max_word_length = self.max_word_length,
                             max_tag_number = self.max_tag_number,
                             map_vocab_threshold = self.evaluation_vocab_size,
                             emb_vocab_threshold = self.emb_vocab_size,
                             char_vocab_threshold = 0,
                             tags_list = self.tags_list)                    
    if self.reps[0]:
      self.input_vocab_size =  max(self.train_set.word_to_id.values())+1
    elif self.reps[1]:
      self.input_vocab_size = max(self.train_set.lems_to_id.values())+1
    if self.reps[4]:
      self.output_vocab_size =  max(self.train_set.word_to_id.values())+1
    elif self.reps[5]:
      self.output_vocab_size = max(self.train_set.lems_to_id.values())+1
    self.eval_vocab_size = max(self.train_set.eval_word_to_id.values())+1
    self.char_vocab_size = max(self.train_set.char_to_id.values())+1
    self.tags_vocab_size = [max(tag_to_id.values()) + 1 for tag_to_id in self.train_set.tags_to_id_l]

    self.train_size = self.train_set.tot
    self.valid_size = 2976
    self.test_size = 2656
    self.n_training_steps = self.train_size // self.batch_size
    self.n_valid_steps = self.valid_size // self.batch_size
    self.n_testing_steps = self.test_size // self.batch_size
    self.training_sub = 5

    #Structural Hyperparameters
    self.c_emb_dim = 20
    self.w_emb_dim = 150
    self.t_emb_dim = 12
    self.w_emb_out_dim = 150
    """ Both next lists must have the same length - final 
    dimension of the word embedding built from characters
    will be the sum of filter dimensions """
    self.window_sizes = [3, 5, 7]
    self.filter_dims = [30, 50, 70]
    self.charLSTM_dim = 75
    self.tagLSTM_dim = 75
    
    self.char_emb_dim = int(self.charLayer == 'conv')*sum(self.filter_dims) + int(self.charLayer == 'LSTM')*2*self.charLSTM_dim + int(self.charLayer == 'concat')*self.c_emb_dim*self.max_word_length
    self.tag_emb_dim =  int(self.tagLayer == 'LSTM')*2*self.tagLSTM_dim + int(self.tagLayer == 'concat')* self.t_emb_dim * self.max_tag_number

    self.emb_dim = int(self.reps[0] or self.reps[1])*self.w_emb_dim + int(self.reps[2])*self.char_emb_dim + int(self.reps[3])*self.tag_emb_dim
    self.hidden_dim = int(self.reps[4] or self.reps[5])*self.w_emb_out_dim + int(self.reps[6])*self.char_emb_dim + int(self.reps[7])*self.tag_emb_dim
    
    if not (self.reps[4] or self.reps[5] or self.reps[6]) and self.reps[7]:
      self.hidden_dim = 150
    self.tag_out_dim = self.hidden_dim
      
    #Training Hyperparameters
    self.learning_rate = 0.0005
    self.lr_decay = 0.75
    self.epochs = 50
    self.patience = 20
    self.dropout = 0.5
    self.reg = 0.0005

    #Objective: 'nce', 'target', else MLE
    self.obj = 'target'
    #Noise: for nce: 'unigram', 'uniform'
    self.noiseBool = (self.obj == 'nce' or self.obj == 'target' or self.obj == 'blackOut')
    self.noise = 'unigram'
    if self.noise == 'unigram':
      self.noiseDistrib = self.train_set.unigram
    elif self.noise == 'uniform':
      self.noiseDistrib= self.train_set.uniform
    self.distortion = 1.0
    self.unique = True
    #self.k = self.batch_size * self.max_seq_length
    self.k = 500
    self.batched_noise = False
    self.noise_length = int(self.noiseBool) * self.k * ( 1 + int(self.batched_noise) * (self.batch_size * self.max_seq_length - 1))

    #Others
    self.save_path = "saves/"
    self.display_step = 5

  def decay(self):
    self.learning_rate = self.learning_rate * self.lr_decay



for j in range(5):
  for k in range(1):
    with tf.Graph().as_default(), tf.Session(
        config=tf.ConfigProto(
          allow_soft_placement=True
        )) as session: 

      print('Preprocessing: Building options')
      opts = Options()    
      if opts.reps[5] and not opts.reps[8]:
        opts.eval_word_map = tf.constant([0, 1] + opts.train_set.wid_to_lemsid ,dtype = 'int64')
      else:
        opts.eval_word_map = tf.constant([0, 1] + opts.train_set.eval_to_word, dtype = 'int64')
    
      if opts.reps[6]:
        print opts.train_set.wid_to_charid.shape
      if opts.reps[7] and not opts.reps[8]:
        print opts.train_set.wid_to_tagsid.shape
      if opts.reps[0] or opts.reps[1]:
        print("Input vocab sizes:  %i" % (opts.input_vocab_size,))
      if opts.reps[4] or opts.reps[5]:
        print("Output vocab sizes:  %i" % (opts.output_vocab_size,))
      print("Eval vocab size:  %i" %  (opts.eval_vocab_size,))
      print("Character vocab size: %i" % (opts.char_vocab_size,))

      print('Preprocessing: Creating data queue')
      train_runner = datarunner(opts.reps,
                                opts.train_size,                            
                                opts.max_seq_length,
                                opts.max_tag_number,
                                opts.max_word_length)
      valid_runner = datarunner(opts.reps,
                                opts.batch_size * 5,
                                opts.max_seq_length,
                                opts.max_tag_number,
                                opts.max_word_length)
      test_runner = datarunner(opts.reps,
                               opts.batch_size * 5,
                               opts.max_seq_length,
                               opts.max_tag_number,
                               opts.max_word_length)
      train_inputs = train_runner.get_inputs(opts.batch_size)
      valid_inputs = valid_runner.get_inputs(32)
      test_inputs = test_runner.get_inputs(32)

      print('Preprocessing: Creating model')
      with tf.variable_scope("model"):
        model = LM(opts, session, train_inputs)
      with tf.variable_scope("model", reuse=True):
        model_valid = LM(opts, session, valid_inputs, training=False)
        model_eval = LM(opts, session, test_inputs, training=False)
      tf.global_variables_initializer().run()
      model._output_wordemb.assign(opts.train_set.initEmb)
      print('Initialized !')

      tf.train.start_queue_runners(sess=session)
      train_gen = opts.train_set.sampler_seq(opts.batch_size)
      valid_gen = opts.valid_set.sampler_seq(32)
      test_gen = opts.test_set.sampler_seq(32)
      train_runner.start_threads(session, train_gen)
      valid_runner.start_threads(session, valid_gen)
      test_runner.start_threads(session, test_gen)

      timeFile = str(datetime.now()).replace(' ','_').replace(':','-').replace('.','_')
      results_file = open('./article/'+ opts.name + timeFile,'w')
      opts_attr = vars(opts)
      for attr in opts_attr.items():
        if not (str(attr[0]) == 'noiseDistrib'):
          results_file.write(str(attr[0]) + ' : ' + str(attr[1]) + '\n')

      saver = tf.train.Saver()  
        
      print ("Epoch 0:")
      results_file.write(str(0) + '\n')
      base_valid_score = model_valid.call(opts.n_valid_steps, results_file)
      _ = model_eval.call(opts.n_testing_steps, results_file)
      saver.save(session,
                 os.path.join(opts.save_path, "model" + timeFile + ".ckpt"))

      ep = 0
      count_patience = 0

      while (ep < opts.epochs) and (count_patience < opts.patience):
        print ("Epoch %i :" % (ep+1))
        results_file.write(str(ep+1) + '\n')
        model.call(opts.n_training_steps // opts.training_sub, results_file)
        valid_score = model_valid.call(opts.n_valid_steps, results_file)
        test_score = model_eval.call(opts.n_testing_steps, results_file)
        if valid_score[0] < base_valid_score[0]:
          base_valid_score = valid_score
          saver.save(session,
                     os.path.join(opts.save_path, "model" + timeFile + ".ckpt"))
          count_patience = 0
        else:
          saver.restore(session,
                        os.path.join(opts.save_path, "model" + timeFile + ".ckpt"))
          model._options.decay()
          count_patience += 1
          results_file.write('Reversed\n')
        ep += 1
      results_file.close()

