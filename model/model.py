import sys
import os
import math
import time

import numpy as np
import tensorflow as tf

from utils import batch_norm, sequence_mask, highway, CE, CE_RNN, restrict_voc, word_char_gate

#Define the class for the Language Model 
class LM(object):
    def __init__(self, options, session, inputs, training=True):
        self._options = options
        self._session = session
        self._training = training
        if self._options.reps[0]:
            self._examples = inputs.pop(0)
        if self._options.reps[1]:
            self._examplesChar = inputs.pop(0)
        if self._options.reps[2]:
            self._labels = inputs.pop(0)
        if self._options.reps[3]:
            self._labelsChar = inputs.pop(0)
        self._evalLabels = inputs.pop(0)
        self.build_graph()

    # TODO: Add possibility of horizontal concatenation ? Possibly inside of CE - same number of parameters, just reshape the filters ?
    # TODO: process version feedforward
    def process_seq(self):
        # Getting input embeddings from inputs 
        if self._options.reps[0]:
            self._examples = tf.cast(tf.verify_tensor_all_finite(tf.cast(self._examples, 'float32'), 'Nan'), 'int64')

            self._wordemb = tf.get_variable(
                name='wordemb',
                shape=[self._options.vocab_size, self._options.w_emb_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.vocab_size))))
            
            w_embeddings = tf.nn.embedding_lookup(self._wordemb, tf.reshape(self._examples, [-1]))
            w_input_emb = tf.reshape(w_embeddings, [self._options.batch_size, self._options.max_seq_length, self._options.w_emb_dim])

            mask = sequence_mask(self._examples)

        if self._options.reps[1]:
            mask, mask_c = sequence_mask(self._examplesChar, char=True)

            if self._options.positionEmbeddings:
                self._charemb_trunc = tf.get_variable(
                    name='charemb',
                    shape=[self._options.char_vocab_size - 1, self._options.max_word_length, self._options.c_emb_dim],
                    initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))
                self._char_pad = tf.constant(0., shape=[1, self._options.max_word_length, self._options.c_emb_dim])
                self._charemb = tf.concat(0, [self._char_pad, self._charemb_trunc])

                position_indexes = tf.cast(tf.range(self._options.max_word_length),'int64')
                indexes = tf.reshape(self._examplesChar + self._options.char_vocab_size * position_indexes, [-1])
                c_embeddings = tf.gather(tf.reshape(self._charemb,
                                                    [self._options.char_vocab_size * self._options.max_word_length, self._options.c_emb_dim]),
                                         indexes)
            else:
                self._charemb_trunc = tf.get_variable(
                    name='charemb',
                    shape=[self._options.char_vocab_size - 1, self._options.c_emb_dim],
                    initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))
                self._char_pad = tf.constant(0., shape=[1, self._options.c_emb_dim])
                self._charemb = tf.concat(0, [self._char_pad, self._charemb_trunc])

                c_embeddings = tf.nn.embedding_lookup(self._charemb, tf.reshape(self._examplesChar, [-1]))

            if self._options.charLayer == "conv":
                self._convfilters = []
                for w, d in zip(self._options.window_sizes, self._options.filter_dims):
                    self._convfilters.append(
                        tf.get_variable(
                            name='filter%d' % w,
                            shape=[w, self._options.c_emb_dim, d],
                            initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(w * self._options.c_emb_dim)))
                        )
                    )
                    weight_decay = tf.nn.l2_loss(self._convfilters[-1])
                    tf.add_to_collection('losses', weight_decay)
                    c_input_emb = tf.reshape(
                        CE(tf.reshape(c_embeddings, [self._options.batch_size * self._options.max_seq_length, self._options.max_word_length, self._options.c_emb_dim]),
                           self._convfilters),
                        [self._options.batch_size, self._options.max_seq_length, -1])

            elif self._options.charLayer == "LSTM":
                self.char_cell_fw = tf.nn.rnn_cell.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                self.char_cell_bw = tf.nn.rnn_cell.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                c_input_emb = tf.reshape(
                    CE_RNN(tf.reshape(c_embeddings, [self._options.batch_size * self._options.max_seq_length, self._options.max_word_length, self._options.c_emb_dim]),
                           self.char_cell_fw,
                           self.char_cell_bw,
                           tf.reshape(tf.reduce_sum(mask_c, 2), [-1])),
                    [self._options.batch_size, self._options.max_seq_length, self._options.charLSTM_dim * 2])

            else:
                c_input_emb = tf.reshape(c_embeddings, [self._options.batch_size, self._options.max_seq_length, -1])

        if self._options.reps[0] and not self._options.reps[1]:
            input_emb = w_input_emb            
        elif self._options.reps[1] and not self._options.reps[0]:
            input_emb = c_input_emb
        elif self._options.reps[0] and self._options.reps[1]:
            input_emb = tf.concat(2, [w_input_emb, c_input_emb])

        input_emb = tf.verify_tensor_all_finite(input_emb, 'Nan')
        # Batch normalization
        if self._options.batch_norm:
            self.batch_normalizer = batch_norm()
            input_emb = self.batch_normalizer(input_emb, self._training)

            input_emb = tf.verify_tensor_all_finite(input_emb, 'Nan')

        # Highway Layer
        if self._options.highway_layers > 0:
            self._highway_w = []
            self._highway_wg = []
            self._highway_b = []
            self._highway_bg = []
            for i in range(self._options.highway_layers):                
                self._highway_w.append(
                    tf.get_variable(
                        name='highway_w%d' % i,
                        shape=[self._options.emb_dim] * 2,
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.emb_dim)))))
                weight_decay = tf.nn.l2_loss(self._highway_w[-1])
                tf.add_to_collection('losses', weight_decay)
                self._highway_b.append(
                    tf.get_variable(
                        name='highway_b%d' % i,
                        shape=[self._options.emb_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.emb_dim)))))
                weight_decay = tf.nn.l2_loss(self._highway_b[-1])
                tf.add_to_collection('losses', weight_decay)
                self._highway_wg.append(
                    tf.get_variable(
                        name='highway_wg%d' % i,
                        shape=[self._options.emb_dim] * 2,
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.emb_dim)))))
                weight_decay = tf.nn.l2_loss(self._highway_wg[-1])
                tf.add_to_collection('losses', weight_decay)
                self._highway_bg.append(
                    tf.get_variable(
                        name='highway_bg%d' % i,
                        shape=[self._options.emb_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.emb_dim)))))
                weight_decay = tf.nn.l2_loss(self._highway_bg[-1])
                tf.add_to_collection('losses', weight_decay)
            input_emb = tf.reshape(highway(tf.reshape(input_emb,
                                                      [-1, self._options.emb_dim]),
                                           self._highway_w,
                                           self._highway_b,
                                           self._highway_wg,
                                           self._highway_bg),
                                   [self._options.batch_size, self._options.max_seq_length, self._options.emb_dim])
        
        input_emb = tf.verify_tensor_all_finite(input_emb, 'Nan')    
        # LSTM
        self.cell = tf.nn.rnn_cell.LSTMCell(self._options.hidden_dim, state_is_tuple = False, activation=tf.nn.relu)
        if self._training and self._options.dropout < 1.0:
            self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self._options.dropout )
        if self._options.hidden_layers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self._options.hidden_layers)
        hidden, _ = tf.nn.dynamic_rnn(self.cell,
                                      input_emb,
                                      sequence_length= tf.reduce_sum(mask, 1),
                                      dtype='float32')

        hidden = tf.verify_tensor_all_finite(hidden, 'Nan')
        print(hidden.get_shape())
        return mask, hidden

    def process_output_seq(self, allVoc=True):    

        if self._options.reps[2]:
            self._output_wordemb = tf.get_variable(
                name="output_wordemb",
                shape= [self._options.vocab_size, self._options.w_emb_out_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.w_emb_out_dim)) ))
            if allVoc:
                w_output_embeddings = tf.nn.embedding_lookup(self._output_wordemb, restrict_voc(tf.range(self._options.eval_vocab_size), self._options.vocab_size))
            else:
                w_output_embeddings = tf.nn.embedding_lookup(self._output_wordemb, tf.reshape(self._labels, [-1])) 
            
        if self._options.reps[3]:            
            if self._options.reps[1] and self._options.reuse_character_layer:
                if allVoc:
                    if self._options.positionEmbeddings:
                        position_indexes = tf.cast(tf.range(self._options.max_word_length),'int64')
                        indexes = tf.reshape(tf.pack(self._options.train_set.wid_to_charid) + self._options.char_vocab_size * position_indexes, [-1])
                        c_output_embeddings = tf.gather(tf.reshape(self._charemb,
                                                                   [self._options.char_vocab_size * self._options.max_word_length, self._options.c_emb_dim]),
                                                        indexes)
                    else:
                        c_output_embeddings = tf.nn.embedding_lookup(self._charemb, tf.reshape(tf.pack(self._options.train_set.wid_to_charid), [-1]))

                    if self._options.charLayer == 'conv':
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                          self._convfilters)
                    elif self._options.charLayer == 'LSTM':                        
                        mask_c = tf.sign(tf.pack(self._options.train_set.wid_to_charid))
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                              self.char_cell_fw,
                                              self.char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                    else:
                        c_output_emb = tf.reshape(c_output_embeddings, [self._options.eval_vocab_size, -1])
                        
                else:
                    if self._options.positionEmbeddings:
                        position_indexes = tf.cast(tf.range(self._options.max_word_length),'int64')
                        indexes = tf.reshape(self._labelsChar + self._options.char_vocab_size * position_indexes, [-1])
                        c_output_embeddings = tf.gather(tf.reshape(self._charemb,
                                                                   [self._options.char_vocab_size * self._options.max_word_length, self._options.c_emb_dim]),
                                                        indexes)
                    else:
                        c_output_embeddings = tf.nn.embedding_lookup(self._charemb, tf.reshape(self._labelsChar, [-1]))

                    if self._options.charLayer == 'conv':
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [self._options.batch_size * self._options.max_seq_length + self._options.noise_length, self._options.max_word_length, self._options.c_emb_dim]),
                                          self._convfilters)
                    elif self._options.charLayer == 'LSTM':
                        mask_c = tf.sign(self._labelsChar)
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [self._options.batch_size * self._options.max_seq_length + self._options.noise_length,  self._options.max_word_length, self._options.c_emb_dim]),
                                              self.char_cell_fw,
                                              self.char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                    else:
                        c_output_emb = tf.reshape(c_output_embeddings, [self._options.batch_size * self._options.max_seq_length + self._options.noise_length, -1])
            else:
                if self._options.positionEmbeddings:
                    self._output_charemb_trunc = tf.get_variable(
                        name='output_charemb',
                        shape=[self._options.char_vocab_size - 1, self._options.max_word_length, self._options.c_emb_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))
                    self._char_pad_out = tf.constant(0., shape=[1, self._options.max_word_length, self._options.c_emb_dim])
                    self._output_charemb = tf.concat(0, [self._char_pad_out, self._output_charemb_trunc])
                    position_indexes = tf.cast(tf.range(self._options.max_word_length),'int64')
                    if allVoc:
                        indexes = tf.reshape(tf.pack(self._options.train_set.wid_to_charid) + self._options.char_vocab_size * position_indexes, [-1])
                        c_output_embeddings = tf.gather(tf.reshape(self._output_charemb,
                                                                   [self._options.char_vocab_size * self._options.max_word_length, self._options.c_emb_dim]),
                                                        indexes)
                    else:
                        indexes = tf.reshape(self._labelsChar + self._options.char_vocab_size * position_indexes, [-1])
                        c_output_embeddings = tf.gather(tf.reshape(self._output_charemb,
                                                                   [self._options.char_vocab_size * self._options.max_word_length, self._options.c_emb_dim]),
                                                        indexes)
                else:
                    self._output_charemb_trunc = tf.get_variable(
                        name='output_charemb',
                        shape=[self._options.char_vocab_size - 1, self._options.c_emb_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))
                    self._char_pad_out = tf.constant(0., shape=[1, self._options.c_emb_dim])
                    self._output_charemb = tf.concat(0, [self._char_pad_out, self._output_charemb_trunc])
                
                    if allVoc:
                        c_output_embeddings = tf.nn.embedding_lookup(self._output_charemb, tf.reshape(tf.pack(self._options.train_set.wid_to_charid), [-1]))
                    else:
                        c_output_embeddings = tf.nn.embedding_lookup(self._output_charemb, tf.reshape(self._labelsChar, [-1])) 

                if self._options.charLayer == 'conv':
                    self._output_convfilters = []
                    for w, d in zip(self._options.window_sizes, self._options.filter_dims):
                        self._output_convfilters.append(
                            tf.get_variable(
                                name='output_filter%d' % w,
                                shape=[w, self._options.c_emb_dim, d],
                                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(w * self._options.c_emb_dim)))
                            )
                        )
                    if allVoc:
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                          self._output_convfilters)
                    else:
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [self._options.batch_size * self._options.max_seq_length + self._options.noise_length , self._options.max_word_length, self._options.c_emb_dim]),
                                          self._output_convfilters)
            
                elif self._options.charLayer == 'LSTM':
                    self.output_char_cell_fw = tf.nn.rnn_cell.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                    self.output_char_cell_bw = tf.nn.rnn_cell.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                    if allVoc:
                        mask_c = tf.sign(tf.pack(self._options.train_set.wid_to_charid))
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                              self.output_char_cell_fw,
                                              self.output_char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                    else:
                        mask_c = tf.sign(self._labelsChar)
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [self._options.batch_size * self._options.max_seq_length + self._options.noise_length,  self._options.max_word_length, self._options.c_emb_dim]),
                                              self.output_char_cell_fw,
                                              self.output_char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                else:
                    if allVoc: 
                        c_output_emb = tf.reshape(c_output_embeddings, [self._options.eval_vocab_size, -1])
                    else:
                        c_output_emb = tf.reshape(c_output_embeddings, [self._options.batch_size * self._options.max_seq_length + self._options.noise_length, -1])

        if self._options.reps[2] and not self._options.reps[3]:
            output_emb = w_output_embeddings
        elif self._options.reps[3] and not self._options.reps[2]:
            output_emb = c_output_emb
        elif self._options.reps[2] and self._options.reps[3] and self._options.wordCharGate:
            self._gateWeight = tf.get_variable(
                name='wg',
                shape=[2 * self._options.hidden_dim, self._options.hidden_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim))))
            weight_decay = tf.nn.l2_loss(self._gateWeight)
            tf.add_to_collection('losses', weight_decay)
            self._gateBias = tf.get_variable(
                name='bg',
                shape=[self._options.hidden_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim))))
            weight_decay = tf.nn.l2_loss(self._gateBias)
            tf.add_to_collection('losses', weight_decay)
            output_emb = word_char_gate(w_output_embeddings, c_output_emb, self._gateWeight, self._gateBias)
        elif self._options.reps[2] and self._options.reps[3]:
            print(w_output_embeddings.get_shape())
            print(c_output_emb.get_shape())
            output_emb = tf.concat(1, [w_output_embeddings, c_output_emb])
        shape = tf.shape(output_emb)
       
        # Highway Layer
        if self._options.output_highway_layers > 0:
            self._output_highway_w = []
            self._output_highway_wg = []
            self._output_highway_b = []
            self._output_highway_bg = []
            for i in range(self._options.output_highway_layers):
                self._output_highway_w.append(
                    tf.get_variable(
                        name='output_highway_w%d' % i,
                        shape=[self._options.hidden_dim] * 2,
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim)))))
                weight_decay = tf.nn.l2_loss(self._output_highway_w[-1])
                tf.add_to_collection('losses', weight_decay)
                self._output_highway_b.append(
                    tf.get_variable(
                        name='output_highway_b%d' % i,
                        shape=[self._options.hidden_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim)))))
                weight_decay = tf.nn.l2_loss(self._output_highway_b[-1])
                tf.add_to_collection('losses', weight_decay)
                self._output_highway_wg.append(
                    tf.get_variable(
                        name='output_highway_wg%d' % i,
                        shape=[self._options.hidden_dim] * 2,
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim)))))
                weight_decay = tf.nn.l2_loss(self._output_highway_wg[-1])
                tf.add_to_collection('losses', weight_decay)
                self._output_highway_bg.append(
                    tf.get_variable(
                        name='output_highway_bg%d' % i,
                        shape=[self._options.hidden_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim)))))
                weight_decay = tf.nn.l2_loss(self._output_highway_bg[-1])
                tf.add_to_collection('losses', weight_decay)
            output_emb = tf.reshape(highway(output_emb,
                                            self._output_highway_w,
                                            self._output_highway_b,
                                            self._output_highway_wg,
                                            self._output_highway_bg),
                                    shape)
        return output_emb

    def tryLoss_seq(self, hidden, mask, output):
        _hidden = tf.reshape(hidden, [-1, self._options.hidden_dim])
        _logits = tf.matmul(_hidden, tf.transpose(output))

        # With masking :      
        _labels = tf.reshape(self._evalLabels, [-1])
        _cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(_logits, _labels)
        _mask = tf.reshape(mask, [-1])
        _cross_entropy = tf.mul(_cross_entropy, tf.cast(_mask, dtype='float32'))
        length = tf.maximum(tf.reduce_sum(mask, 1), tf.ones(self._options.batch_size, dtype = 'int64'))
        cross_entropy_seq = tf.reduce_sum(tf.reshape(_cross_entropy,[self._options.batch_size, self._options.max_seq_length]), 1) / tf.cast(length, dtype='float32')
        
        loss = tf.reduce_mean(cross_entropy_seq)
        return loss

    def nce_noise(self):
        labels_exp = tf.expand_dims(tf.reshape(self._evalLabels, [-1]), 1)
        (negative_samples,
         true_expected_counts,
         sampled_expected_counts) = tf.nn.fixed_unigram_candidate_sampler(labels_exp,
                                                                          1,
                                                                          self._options.noise_length,
                                                                          False,
                                                                          self._options.vocab_size,
                                                                          distortion=self._options.distortion,
                                                                          num_reserved_ids=0,
                                                                          unigrams=self._options.noiseDistrib,
                                                                          name='nce_sampling')
        negative_samplesLabels = restrict_voc(negative_samples, self._options.vocab_size)
        if self._options.reps[2]:
            self._labels = tf.concat(0, [tf.reshape(self._labels, [-1]), negative_samplesLabels])
        else:
            self._labels = tf.concat(0, [tf.reshape(self._evalLabels, [-1]), negative_samplesLabels])
        if self._options.reps[3]:
            negative_samplesChar = tf.gather(tf.pack(self._options.train_set.wid_to_charid), negative_samples)
            self._labelsChar = tf.concat(0, [tf.reshape(self._labelsChar, [-1, self._options.max_word_length]), negative_samplesChar])
        return true_expected_counts, sampled_expected_counts

    def target_noise(self):
        labels_exp = tf.expand_dims(tf.reshape(self._evalLabels, [-1]), 1)
        (negative_samples,
         true_expected_counts,
         sampled_expected_counts) = tf.nn.learned_unigram_candidate_sampler(labels_exp,
                                                                            1,
                                                                            self._options.noise_length,
                                                                            False,
                                                                            self._options.vocab_size,
                                                                            name='target_sampling')
        if self._options.reps[2]:            
            negative_samplesLabels = restrict_voc(negative_samples, self._options.vocab_size)
            self._labels = tf.concat(0, [tf.reshape(self._labels, [-1]), negative_samplesLabels])            
        if self._options.reps[3]:
            negative_samplesChar = tf.gather(tf.pack(self._options.train_set.wid_to_charid), negative_samples)
            self._labelsChar = tf.concat(0, [tf.reshape(self._labelsChar, [-1, self._options.max_word_length]), negative_samplesChar])   
        return true_expected_counts, sampled_expected_counts
        
    def blackOut_loss(self, hidden, mask, output_weight, expected_counts):
        # Here, we use the same noise as NCE - distorted unigram - but unique needs to be put to false,
        # and we need to remove accidental hits
        _hidden = tf.reshape(hidden, [-1, self._options.hidden_dim])
        true_weight = tf.slice(output_weight,
                               [0, 0],
                               [self._options.batch_size * self._options.max_seq_length, -1])
        sampled_weight = tf.slice(output_weight,
                                 [self._options.batch_size * self._options.max_seq_length, 0],
                                 [self._options.noise_length, -1])

        true_logits = tf.expand_dims(tf.nn._sum_rows(tf.mul(_hidden, true_weight)), 1)
        sampled_logits = tf.matmul(_hidden, tf.transpose(sampled_weight))

        true_logits -= tf.log(expected_counts[0])
        sampled_logits -= tf.expand_dims(tf.log(expected_counts[1]),0)

        labels = tf.expand_dims(tf.slice(tf.reshape(self._labels, [-1]),[0],[self._options.batch_size * self._options.max_seq_length]), 1)
        sampled = tf.slice(tf.reshape(self._labels, [-1]),[self._options.batch_size * self._options.max_seq_length],[self._options.noise_length])

        # Necessary - remove accidental hits
        acc_hits = tf.nn.compute_accidental_hits(labels, sampled, num_true=1)
        acc_indices, acc_ids, acc_weights = acc_hits

        print(labels.get_shape())
        print(sampled.get_shape())

        # This is how SparseToDense expects the indices.
        acc_indices_2d = tf.expand_dims(acc_indices, 1)
        acc_ids_2d_int32 = tf.expand_dims(tf.cast(acc_ids, 'int32'), 1)
        sparse_indices = tf.concat(1, [acc_indices_2d, acc_ids_2d_int32])
        
        # Create sampled_logits_shape = [batch_size, num_sampled]
        sampled_logits_shape = [self._options.batch_size * self._options.max_seq_length, self._options.noise_length]
        
        if sampled_logits.dtype != acc_weights.dtype:
            acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
        sampled_logits += tf.sparse_to_dense(
            sparse_indices,
            sampled_logits_shape,
            acc_weights,
            default_value=0.0,
            validate_indices=False)
                
        print(true_logits.get_shape())
        print(sampled_logits.get_shape())
        
        maxes = tf.reduce_max(sampled_logits, 1, keep_dims=True)
        sampled_logits_without_maxes = sampled_logits - maxes
        noise_offset = tf.expand_dims(tf.squeeze(maxes, [-1]) + tf.log(tf.nn._sum_rows(tf.exp(sampled_logits_without_maxes))), 1)
        #noise_offset = tf.expand_dims(tf.log(tf.nn._sum_rows(tf.exp(sampled_logits))), 1)

        logits = tf.concat(1, [true_logits, sampled_logits]) - noise_offset
        labels = tf.concat(1, [tf.ones_like(true_logits),  tf.zeros_like(sampled_logits)])
        
        score = tf.nn._sum_rows(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels))
        
        # With masking:
        _mask = tf.reshape(mask, [-1])
        masked_score = tf.mul(score, tf.cast(_mask, dtype='float32'))
        loss = tf.reduce_sum(masked_score) / tf.cast(tf.reduce_sum(mask), 'float32')

        return tf.reduce_sum(loss)
        

    def nce_loss(self, hidden, mask, output_weights, expected_counts):
        labels_exp = tf.expand_dims(tf.cast(tf.range(self._options.batch_size * self._options.max_seq_length), dtype='int64'), 1)
        target_score = tf.nn.nce_loss(output_weights,
                                      tf.zeros([self._options.batch_size * self._options.max_seq_length + self._options.noise_length], 'float32', name='zeros_biases'),
                                      tf.reshape(hidden, [-1, self._options.hidden_dim]),
                                      labels_exp,
                                      self._options.noise_length,
                                      self._options.vocab_size,
                                      num_true=1,
                                      sampled_values=(tf.cast(tf.range(self._options.batch_size * self._options.max_seq_length,
                                                                       self._options.batch_size * self._options.max_seq_length + self._options.noise_length), dtype='int64'),
                                                      expected_counts[0],
                                                      expected_counts[1]),
                                      remove_accidental_hits=True,
                                      partition_strategy='mod',
                                      name='sampled_softmax_loss')
        # With masking:                                                                                    
        _mask = tf.reshape(mask, [-1])
        masked_score = tf.mul(target_score, tf.cast(_mask, dtype='float32'))
        loss = tf.reduce_sum(masked_score) / tf.cast(tf.reduce_sum(mask), 'float32')
        return loss
        
    def target_loss(self, hidden, mask, output_weights, expected_counts):
        labels_exp = tf.expand_dims(tf.cast(tf.range(self._options.batch_size * self._options.max_seq_length), dtype='int64'), 1)
        target_score = tf.nn.sampled_softmax_loss(output_weights,
                                                  tf.zeros([self._options.batch_size * self._options.max_seq_length + self._options.noise_length], 'float32', name='zeros_biases'),
                                                  tf.reshape(hidden, [-1, self._options.hidden_dim]),
                                                  labels_exp,
                                                  self._options.noise_length,
                                                  self._options.vocab_size,
                                                  num_true=1,
                                                  sampled_values=(tf.cast(tf.range(self._options.batch_size * self._options.max_seq_length,
                                                                                   self._options.batch_size * self._options.max_seq_length + self._options.noise_length), dtype='int64'),
                                                                  expected_counts[0],
                                                                  expected_counts[1]),
                                                  remove_accidental_hits=True,
                                                  partition_strategy='mod',
                                                  name='sampled_softmax_loss')
        # With masking:
        _mask = tf.reshape(mask, [-1])
        masked_score = tf.mul(target_score, tf.cast(_mask, dtype='float32'))
        loss = tf.reduce_sum(masked_score) / tf.cast(tf.reduce_sum(mask), 'float32')
        return loss

    def optimize(self, loss):
          self._lr = self._options.learning_rate
          optimizer = tf.train.AdamOptimizer(self._lr)
          
          gvs = optimizer.compute_gradients(loss)          
          gvs = [(tf.verify_tensor_all_finite(grad, 'NaN with '+var.name), var) for grad, var in gvs]
          capped_gvs = [(tf.clip_by_value(grad,-0.1,0.1), var) for grad, var in gvs]
          train = optimizer.apply_gradients(capped_gvs)
          
          train = optimizer.minimize(loss + self._options.reg*tf.add_n(tf.get_collection('losses')))
          self._train = train

    def build_graph(self):
        _mask, _hidden = self.process_seq()
        if self._options.obj == 'target' and self._training:
            _expected_counts = self.target_noise()
        elif (self._options.obj == 'nce' or self._options.obj == 'blackOut') and self._training:
            _expected_counts = self.nce_noise()
        _output_emb = self.process_output_seq(not (self._options.noiseBool) or not self._training)
        if self._options.obj == 'target' and self._training:
            loss = self.target_loss(_hidden, _mask, _output_emb, _expected_counts)
        elif self._options.obj == 'nce' and self._training:
            loss = self.nce_loss(_hidden, _mask, _output_emb, _expected_counts)
        elif self._options.obj == 'blackOut' and self._training:
            loss =  self.blackOut_loss(_hidden, _mask, _output_emb, _expected_counts)
        else:
            loss = self.tryLoss_seq(_hidden, _mask, _output_emb)
        if self._training:
            self.optimize(loss)
        self._loss = loss

    def call(self, results_file = None):
        start_time = time.time()
        average_loss = 0
        if self._training:
            print('In training')
            n_steps = self._options.n_training_steps // self._options.training_sub
            op = self._train
            display = n_steps // self._options.display_step
            call = "Training:"
        else:
            n_steps = self._options.n_testing_steps
            op = tf.no_op()
            display = n_steps-1
            call = "Testing:"

        for step in xrange(n_steps):
            _, loss = self._session.run([op, self._loss])
            average_loss+= loss

            # Record monitored values                                                                               
            if self._training:
                if step % (display) == 0:
                    print(" %s Cross-entropy at batch %i : %.3f ; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                          loss, 
                                                                                                          (time.time() - start_time) / (step + 1) ))
            else:
                if step % (display) == 0 and step > 0:
                    print(" %s Perplexity, score and norm at batch %i : %.3f; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                                      average_loss/(step+1),
                                                                                                                      (time.time() - start_time) / (step + 1) ))
            if step == (n_steps - 1):
                if not results_file == None:
                    results_file.write( str(average_loss/(step+1)) + '\n')
                results_file.flush()
