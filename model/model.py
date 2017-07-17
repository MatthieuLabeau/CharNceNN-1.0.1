import sys
import os
import math
import time

import numpy as np
import tensorflow as tf

from utils import batch_norm, sequence_mask, highway, CE, CE_RNN, restrict_voc, restrict_voc_map, word_char_gate, _compute_ranged_scores

#Define the class for the Language Model 
class LM(object):
    def __init__(self, options, session, inputs, training=True):
        self._options = options
        self._session = session
        self._training = training

        if self._options.reps[0] or self._options.reps[1]:
            self._examples = inputs.pop(0)
        if self._options.reps[2]:
            self._examplesChar = inputs.pop(0)
        if self._options.reps[3]:
            self._examplesTags = inputs.pop(0)

        if self._options.reps[4] or self._options.reps[5]:
            self._labels = inputs.pop(0)
        if self._options.reps[6]:
            self._labelsChar = inputs.pop(0)
        if self._options.reps[7]:
            self._evalTags = inputs.pop(0)
            
        self._evalLabels = inputs.pop(0)

        self.reverse_gvs = []
        self.build_graph()
        
    def process_seq(self):
        # Getting input embeddings from inputs 
        # Mots ou lemmes
        if self._options.reps[0] or self._options.reps[1]:
            self._wordemb = tf.get_variable(
                name='wordemb',
                shape=[self._options.input_vocab_size, self._options.w_emb_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.input_vocab_size))))
            
            w_embeddings = tf.nn.embedding_lookup(self._wordemb, tf.reshape(self._examples, [-1]))
            w_input_emb = tf.reshape(w_embeddings, [-1, self._options.max_seq_length, self._options.w_emb_dim])

            mask = sequence_mask(self._examples)

        # Tags
        if self._options.reps[3]:
            self._tagEmbs = []
            tag_input_embs = []
            for i in range(self._options.max_tag_number):
                self._tagEmbs.append(tf.get_variable(
                    name="tagEmbs%d" % i,
                    shape= [self._options.tags_vocab_size[i], self._options.t_emb_dim],
                    initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.t_emb_dim)) )))
            
                tag_embeddings = tf.nn.embedding_lookup(self._tagEmbs[i], tf.reshape(self._examplesTags[:,:,i], [-1]))
                tag_input_embs.append(tf.reshape(tag_embeddings, [-1, self._options.max_seq_length, self._options.t_emb_dim]))

            if self._options.tagLayer == "LSTM":
                tag_embeddings = tf.concat(axis=2, values=tag_input_embs)
                self.tag_cell_fw = tf.contrib.rnn.LSTMCell(self._options.tagLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                self.tag_cell_bw = tf.contrib.rnn.LSTMCell(self._options.tagLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                if self._training and self._options.dropout < 1.0:
                    self.tag_cell_fw = tf.contrib.rnn.DropoutWrapper(self.tag_cell_fw, output_keep_prob=self._options.dropout )
                    self.tag_cell_bw = tf.contrib.rnn.DropoutWrapper(self.tag_cell_bw, output_keep_prob=self._options.dropout )
                size = int(self._training) * self._options.batch_size + int(not self._training) * 32
                tag_input_emb = tf.reshape(
                    CE_RNN(tf.reshape(tag_embeddings, [-1, self._options.max_tag_number, self._options.t_emb_dim]),
                           self.tag_cell_fw,
                           self.tag_cell_bw,
                           tf.constant(self._options.max_tag_number, dtype='int64', shape=[size * self._options.max_seq_length,])),
                    [-1, self._options.max_seq_length, self._options.charLSTM_dim * 2])
                
            else:
                tag_input_emb = tf.concat(axis=2, values=tag_input_embs)
            
        # Caracteres
        if self._options.reps[2]:
            mask, mask_c = sequence_mask(self._examplesChar, char=True)

            self._charemb_trunc = tf.get_variable(
                name='charemb',
                shape=[self._options.char_vocab_size - 1, self._options.c_emb_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))
            self._char_pad = tf.constant(0., shape=[1, self._options.c_emb_dim])
            self._charemb = tf.concat(axis=0, values=[self._char_pad, self._charemb_trunc])

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
                    CE(tf.reshape(c_embeddings, [-1, self._options.max_word_length, self._options.c_emb_dim]),
                       self._convfilters),
                    [-1, self._options.max_seq_length, self._options.char_emb_dim])

            elif self._options.charLayer == "LSTM":
                self.char_cell_fw = tf.contrib.rnn.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                self.char_cell_bw = tf.contrib.rnn.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                if self._training and self._options.dropout < 1.0:
                    self.char_cell_fw = tf.contrib.rnn.DropoutWrapper(self.char_cell_fw, output_keep_prob=self._options.dropout )
                    self.char_cell_bw = tf.contrib.rnn.DropoutWrapper(self.char_cell_bw, output_keep_prob=self._options.dropout )
                c_input_emb = tf.reshape(
                    CE_RNN(tf.reshape(c_embeddings, [-1, self._options.max_word_length, self._options.c_emb_dim]),
                           self.char_cell_fw,
                           self.char_cell_bw,
                           tf.reshape(tf.reduce_sum(mask_c, 2), [-1])),
                    [-1, self._options.max_seq_length, self._options.charLSTM_dim * 2])

            else:
                c_input_emb = tf.reshape(c_embeddings, [-1, self._options.max_seq_length, self._options.char_emb_dim])

        embs = []      
        if (self._options.reps[0] or self._options.reps[1]):
            embs.append(w_input_emb)
        if self._options.reps[2]:
            embs.append(c_input_emb)
        if self._options.reps[3]:
            embs.append(tag_input_emb)
        input_emb = tf.concat(axis=2, values=embs)

        # Batch normalization
        if self._options.batch_norm:
            self.batch_normalizer = batch_norm()
            input_emb = self.batch_normalizer(input_emb, self._training)

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
                                   [-1, self._options.max_seq_length, self._options.emb_dim])
        
        # LSTM
        self.cell = tf.contrib.rnn.LSTMCell(self._options.hidden_dim, state_is_tuple = False, activation=tf.nn.relu)
        if self._training and self._options.dropout < 1.0:
            self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self._options.dropout )
        if self._options.hidden_layers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self._options.hidden_layers)
        hidden, _ = tf.nn.dynamic_rnn(self.cell,
                                      input_emb,
                                      sequence_length= tf.reduce_sum(mask, 1),
                                      dtype='float32')

        print(hidden.get_shape())
        return mask, hidden

    def process_output_seq(self, allVoc=True):    
        # Mots
        if self._options.reps[4] or self._options.reps[5]:
            self._output_wordemb = tf.get_variable(
                name="output_wordemb",
                shape= [self._options.output_vocab_size, self._options.w_emb_out_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.w_emb_out_dim)) ))
            if allVoc:                
                w_output_embeddings = tf.nn.embedding_lookup(self._output_wordemb,
                                                             restrict_voc_map(tf.range(self._options.eval_vocab_size), self._options.eval_word_map))
            else:
                w_output_embeddings = tf.nn.embedding_lookup(self._output_wordemb, tf.reshape(self._labels, [-1])) 
            
        # Caracteres
        if self._options.reps[6]:            
            if self._options.reps[2] and self._options.reuse_character_layer:
                if allVoc:
                    c_output_embeddings = tf.nn.embedding_lookup(self._charemb, tf.reshape(tf.stack(self._options.train_set.wid_to_charid), [-1]))

                    if self._options.charLayer == 'conv':
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                          self._convfilters)
                    elif self._options.charLayer == 'LSTM':                        
                        mask_c = tf.sign(tf.stack(self._options.train_set.wid_to_charid))
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                              self.char_cell_fw,
                                              self.char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                    else:
                        c_output_emb = tf.reshape(c_output_embeddings, [self._options.eval_vocab_size, -1])
                        
                else:
                    c_output_embeddings = tf.nn.embedding_lookup(self._charemb, tf.reshape(self._labelsChar, [-1]))

                    if self._options.charLayer == 'conv':
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [-1, self._options.max_word_length, self._options.c_emb_dim]),
                                          self._convfilters)
                    elif self._options.charLayer == 'LSTM':
                        mask_c = tf.sign(self._labelsChar)
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [-1,  self._options.max_word_length, self._options.c_emb_dim]),
                                              self.char_cell_fw,
                                              self.char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                    else:
                        c_output_emb = tf.reshape(c_output_embeddings, [self._options.batch_size * self._options.max_seq_length + self._options.noise_length, -1])
            else:
                self._output_charemb_trunc = tf.get_variable(
                    name='output_charemb',
                    shape=[self._options.char_vocab_size - 1, self._options.c_emb_dim],
                    initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))
                self._char_pad_out = tf.constant(0., shape=[1, self._options.c_emb_dim])
                self._output_charemb = tf.concat(axis=0, values=[self._char_pad_out, self._output_charemb_trunc])
                
                if allVoc:
                    c_output_embeddings = tf.nn.embedding_lookup(self._output_charemb, tf.reshape(tf.stack(self._options.train_set.wid_to_charid), [-1]))
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
                        weight_decay = tf.nn.l2_loss(self._output_convfilters[-1])
                        tf.add_to_collection('losses', weight_decay)
                    if allVoc:
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                          self._output_convfilters)
                    else:
                        c_output_emb = CE(tf.reshape(c_output_embeddings,
                                                     [-1 , self._options.max_word_length, self._options.c_emb_dim]),
                                          self._output_convfilters)
            
                elif self._options.charLayer == 'LSTM':
                    self.output_char_cell_fw = tf.contrib.rnn.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                    self.output_char_cell_bw = tf.contrib.rnn.LSTMCell(self._options.charLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                    if self._training and self._options.dropout < 1.0:
                        self.output_char_cell_fw = tf.contrib.rnn.DropoutWrapper(self.output_char_cell_fw, output_keep_prob=self._options.dropout )
                        self.output_char_cell_bw = tf.contrib.rnn.DropoutWrapper(self.output_char_cell_bw, output_keep_prob=self._options.dropout )
                    if allVoc:
                        mask_c = tf.sign(tf.stack(self._options.train_set.wid_to_charid))
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [self._options.eval_vocab_size, self._options.max_word_length, self._options.c_emb_dim]),
                                              self.output_char_cell_fw,
                                              self.output_char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                    else:
                        mask_c = tf.sign(self._labelsChar)
                        c_output_emb = CE_RNN(tf.reshape(c_output_embeddings,
                                                         [-1,  self._options.max_word_length, self._options.c_emb_dim]),
                                              self.output_char_cell_fw,
                                              self.output_char_cell_bw,
                                              tf.reduce_sum(mask_c, 1),
                                              name = 'biRNNOut')
                else:
                    c_output_emb = tf.reshape(c_output_embeddings, [-1, sum(self._options.filters_dims)])
        # Tags
        if self._options.reps[7] and not self._options.reps[8]:
            self._output_tagEmbs = []
            tag_output_embs = []
            tags_map = tf.stack(self._options.train_set.wid_to_tagsid)
            for i in range(self._options.max_tag_number):
                """
                self._output_tagEmbs.append(tf.get_variable(
                    name="output_tagEmbs%d" % i,
                    shape= [self._options.tags_vocab_size[i], self._options.t_emb_dim],
                    initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.t_emb_dim)) )))
                """
                if allVoc:
                    tag_output_embeddings = tf.nn.embedding_lookup(self._tagEmbs[i], tf.reshape(tags_map[:, i], [-1]))
                else:
                    tag_output_embeddings = tf.nn.embedding_lookup(self._tagEmbs[i], tf.reshape(self._evalTags[:,i], [-1]))
                tag_output_embs.append(tf.reshape(tag_output_embeddings, [-1, self._options.t_emb_dim]))

            tag_output_emb = tf.concat(axis=1, values=tag_output_embs)
            if self._options.tagLayer == "LSTM":
                self.output_tag_cell_fw = tf.contrib.rnn.LSTMCell(self._options.tagLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                self.output_tag_cell_bw = tf.contrib.rnn.LSTMCell(self._options.tagLSTM_dim, state_is_tuple = False, activation=tf.nn.relu)
                if self._training and self._options.dropout < 1.0:
                    self.output_tag_cell_fw = tf.contrib.rnn.DropoutWrapper(self.output_tag_cell_fw, output_keep_prob=self._options.dropout )
                    self.output_tag_cell_bw = tf.contrib.rnn.DropoutWrapper(self.output_tag_cell_bw, output_keep_prob=self._options.dropout )
                size = int(self._training)*(self._options.batch_size * self._options.max_seq_length + self._options.noise_length) + int(not self._training) * self._options.eval_vocab_size
                tag_output_emb = CE_RNN(tf.reshape(tag_output_emb, [-1, self._options.max_tag_number, self._options.t_emb_dim]),
                                        self.output_tag_cell_fw,
                                        self.output_tag_cell_bw,
                                        tf.constant(self._options.max_tag_number, dtype='int64', shape=[size,]),
                                        name = 'tagBiRNNOut')

        output_embs = []
        if (self._options.reps[4] or self._options.reps[5]):
            output_embs.append(w_output_embeddings)
        if self._options.reps[6]:
            output_embs.append(c_output_emb)
        if self._options.reps[7] and not self._options.reps[8]:
            output_embs.append(tag_output_emb)
        output_emb = tf.concat(axis=1, values=output_embs)
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
        _cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=_labels)
        _mask = tf.reshape(mask, [-1])
        _cross_entropy = tf.multiply(_cross_entropy, tf.cast(_mask, dtype='float32'))
        #length = tf.maximum(tf.reduce_sum(mask, 1), tf.ones(self._options.batch_size, dtype = 'int64'))
        length = tf.reduce_sum(mask, 1)
        cross_entropy_seq = tf.reduce_sum(tf.reshape(_cross_entropy,[-1, self._options.max_seq_length]), 1) / tf.cast(length, dtype='float32')
        
        loss = [tf.reduce_mean(cross_entropy_seq)]

        if self._options.freqLoss:
            freqLosses = _compute_ranged_scores(tf.reshape(_cross_entropy,[-1, self._options.max_seq_length]),
                                                self._evalLabels,
                                                mask,
                                                self._options.ranges)
            loss = loss + freqLosses
        return loss

    def nce_noise(self):
        labels_exp = tf.expand_dims(tf.reshape(self._evalLabels, [-1]), 1)
        (negative_samples,
         true_expected_counts,
         sampled_expected_counts) = tf.nn.fixed_unigram_candidate_sampler(labels_exp,
                                                                          1,
                                                                          self._options.noise_length,
                                                                          self._options.unique,
                                                                          self._options.eval_vocab_size,
                                                                          distortion=self._options.distortion,
                                                                          num_reserved_ids=0,
                                                                          unigrams=self._options.noiseDistrib,
                                                                          name='nce_sampling')
        if self._options.reps[4] or self._options.reps[5]:
            negative_samplesLabels = restrict_voc_map(negative_samples, self._options.eval_word_map)
            self._labels = tf.concat(axis=0, values=[tf.reshape(self._labels, [-1]), negative_samplesLabels])
        else:
            self._labels = tf.concat(axis=0, values=[tf.reshape(self._evalLabels, [-1]), negative_samples])
        if self._options.reps[6]:
            negative_samplesChar = tf.gather(tf.stack(self._options.train_set.wid_to_charid), negative_samples)
            self._labelsChar = tf.concat(axis=0, values=[tf.reshape(self._labelsChar, [-1, self._options.max_word_length]), negative_samplesChar])
        if self._options.reps[7] and not self._options.reps[8]:
            negative_samplesTags = tf.gather(tf.stack(self._options.train_set.wid_to_tagsid), negative_samples)
            self._evalTags = tf.concat(axis = 0, values=[tf.reshape(self._evalTags, [-1, self._options.max_tag_number]), negative_samplesTags])
        return true_expected_counts, sampled_expected_counts

    def target_noise(self):
        labels_exp = tf.expand_dims(tf.reshape(self._evalLabels, [-1]), 1)
        (negative_samples,
         true_expected_counts,
         sampled_expected_counts) = tf.nn.learned_unigram_candidate_sampler(labels_exp,
                                                                            1,
                                                                            self._options.noise_length,
                                                                            self._options.unique,
                                                                            self._options.eval_vocab_size,
                                                                            name='target_sampling')
        if self._options.reps[4] or self._options.reps[5]:            
            negative_samplesLabels = restrict_voc_map(negative_samples, self._options.eval_word_map)
            self._labels = tf.concat(axis=0, values=[tf.reshape(self._labels, [-1]), negative_samplesLabels])            
        if self._options.reps[6]:
            negative_samplesChar = tf.gather(tf.stack(self._options.train_set.wid_to_charid), negative_samples)
            self._labelsChar = tf.concat(axis=0, values=[tf.reshape(self._labelsChar, [-1, self._options.max_word_length]), negative_samplesChar])
        if self._options.reps[7] and not self._options.reps[8]:
            negative_samplesTags = tf.gather(tf.stack(self._options.train_set.wid_to_tagsid), negative_samples)
            self._evalTags = tf.concat(axis = 0, values=[tf.reshape(self._evalTags, [-1, self._options.max_tag_number]), negative_samplesTags])
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

        true_logits = tf.expand_dims(tf.reduce_sum(tf.multiply(_hidden, true_weight), 1), 1)
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
        sparse_indices = tf.concat(axis=1, values=[acc_indices_2d, acc_ids_2d_int32])
        
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
        noise_offset = tf.expand_dims(tf.squeeze(maxes, [-1]) + tf.log(tf.reduce_sum(tf.exp(sampled_logits_without_maxes), 1)), 1)
        #noise_offset = tf.expand_dims(tf.log(tf.nn._sum_rows(tf.exp(sampled_logits))), 1)

        logits = tf.concat(axis=1, values=[true_logits, sampled_logits]) - noise_offset
        labels = tf.concat(axis=1, values=[tf.ones_like(true_logits),  tf.zeros_like(sampled_logits)])
        
        score = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), 1)
        
        # With masking:
        _mask = tf.reshape(mask, [-1])
        masked_score = tf.multiply(score, tf.cast(_mask, dtype='float32'))
        loss = tf.reduce_sum(masked_score) / tf.cast(tf.reduce_sum(mask), 'float32')

        return tf.reduce_sum(loss)
        

    def nce_loss(self, hidden, mask, output_weights, expected_counts):
        labels_exp = tf.expand_dims(tf.cast(tf.range(self._options.batch_size * self._options.max_seq_length), dtype='int64'), 1)
        target_score = tf.nn.nce_loss(output_weights,
                                      tf.zeros([self._options.batch_size * self._options.max_seq_length + self._options.noise_length], 'float32', name='zeros_biases'),
                                      labels_exp,
                                      tf.reshape(hidden, [-1, self._options.hidden_dim]),
                                      self._options.noise_length,
                                      self._options.eval_vocab_size,
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
        masked_score = tf.multiply(target_score, tf.cast(_mask, dtype='float32'))
        loss = tf.reduce_sum(masked_score) / tf.cast(tf.reduce_sum(mask), 'float32')
        return loss
        
    def target_loss(self, hidden, mask, output_weights, expected_counts):
        labels_exp = tf.expand_dims(tf.cast(tf.range(self._options.batch_size * self._options.max_seq_length), dtype='int64'), 1)
        target_score = tf.nn.sampled_softmax_loss(output_weights,
                                                  tf.zeros([self._options.batch_size * self._options.max_seq_length + self._options.noise_length], 'float32', name='zeros_biases'),
                                                  labels_exp,
                                                  tf.reshape(hidden, [-1, self._options.hidden_dim]),
                                                  self._options.noise_length,
                                                  self._options.eval_vocab_size,
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
        masked_score = tf.multiply(target_score, tf.cast(_mask, dtype='float32'))
        loss = tf.reduce_sum(masked_score) / tf.cast(tf.reduce_sum(mask), 'float32')
        return loss
    
    def tags_sum_loss(self, hidden, mask):

        self._tagOutputWeights = []
        self._tagOutputBiases = []
        for i in range(self._options.max_tag_number):
            self._tagOutputWeights.append(tf.get_variable(
                name="tagOutputEmb%d" % i,
                shape= [self._options.tag_out_dim, self._options.tags_vocab_size[i]],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.tag_out_dim)) )))
            self._tagOutputBiases.append(tf.get_variable(
                name='tagOutputBiases%d' % i,
                shape= [self._options.tags_vocab_size[i]],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.tag_out_dim)) )))              

        hidden = tf.reshape(hidden, [-1, self._options.tag_out_dim])
    
        tags_logits = []
        tags_labels = []
        tags_cross_entropy = []
        for i in range(self._options.max_tag_number):
            tags_logits.append(tf.matmul(hidden, self._tagOutputWeights[i]) + self._tagOutputBiases[i])
            tags_labels.append(tf.reshape(self._evalTags[:,:,i], [-1]))
            tags_cross_entropy.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tags_logits[i], labels=tags_labels[i]))
            
        cross_entropy = tf.stack(tags_cross_entropy, axis = 1)
        
        mask_r = tf.reshape(mask, [-1])
        cross_entropy = tf.multiply(cross_entropy, tf.expand_dims(tf.cast(mask_r, dtype='float32'), 1))
            
        length = tf.reduce_sum(mask, 1)
        cross_entropy_seq = tf.reduce_sum( tf.reshape( cross_entropy,
                                                       [- 1,
                                                        self._options.max_seq_length,
                                                        self._options.max_tag_number]),
                                           1) / tf.expand_dims(tf.cast(length, dtype='float32'), 1)
            
        losses = tf.reduce_mean(cross_entropy_seq, 0)
        return losses
        

    def optimize(self, loss):
        self._lr = self._options.learning_rate
        self.optimizer = tf.train.AdamOptimizer(self._lr)
        
        if len(tf.get_collection('losses')) > 0: 
            train = self.optimizer.minimize(loss + self._options.reg*tf.add_n(tf.get_collection('losses')))
        else:
            train = self.optimizer.minimize(loss)
        return train

    def reset_optimizer(self):
        for slot in self.optimizer.get_slot_names():
            for (var, slot_for_the_variable) in self.optimizer._slots[slot].iteritems():          
                reset = tf.assign(slot_for_the_variable, tf.zeros_like(var))
                _ = self._session.run([reset])
                
    def build_graph(self):
        _mask, _hidden = self.process_seq()
        if self._options.obj == 'target' and self._training:
            _expected_counts = self.target_noise()
        elif (self._options.obj == 'nce' or self._options.obj == 'blackOut') and self._training:
            _expected_counts = self.nce_noise()
        if (self._options.reps[4] or self._options.reps[5] or self._options.reps[6] or (self._options.reps[7] and not self._options.reps[8])):
            _output_emb = self.process_output_seq(not (self._options.noiseBool) or not self._training)
            if self._options.obj == 'target' and self._training:
                self.loss = self.target_loss(_hidden, _mask, _output_emb, _expected_counts)
            elif self._options.obj == 'nce' and self._training:
                self.loss = self.nce_loss(_hidden, _mask, _output_emb, _expected_counts)
            elif self._options.obj == 'blackOut' and self._training:
                self.loss =  self.blackOut_loss(_hidden, _mask, _output_emb, _expected_counts)
            else:
                self.loss = self.tryLoss_seq(_hidden, _mask, _output_emb)
        if self._options.reps[7] and self._options.reps[8]:
            self._tag_losses = self.tags_sum_loss(_hidden, _mask)            

        """
        if (self._options.reps[4] or self._options.reps[5] or self._options.reps[6]) and self._options.reps[7]:
            self._loss = self.loss + tf.reduce_sum(self._tag_losses)
        """

        self._loss = self.loss
        if self._options.reps[7] and self._options.reps[8]:
            self._tag_loss = tf.reduce_sum(self._tag_losses)
            if self._training:
                self._train = self.optimize(0.05 * self._tag_loss + self._loss)
        else:
            if self._training:
                self._train = self.optimize(self._loss)

            
    def call(self, n_steps, results_file = None):
        start_time = time.time()
        if self._options.reps[7] and self._options.reps[8]:
            average_tag_losses = np.zeros(self._options.max_tag_number)
        if self._training:
            print('In training')
            average_loss = 0
            op = self._train
            display = n_steps // self._options.display_step
            call = "Training:"
        else:
            average_loss = np.zeros(len(self._options.ranges))
            op = tf.no_op()
            display = n_steps-1
            call = "Validation/Testing:"

        for step in xrange(n_steps):
            if self._options.reps[7] and self._options.reps[8]:
                _, loss, tag_losses = self._session.run([op, self._loss, self._tag_losses])
                average_tag_losses += tag_losses
            else:
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
                                                                                                                      average_loss[0]/(step+1),
                                                                                                                      (time.time() - start_time) / (step + 1) ))
            if step == (n_steps - 1):
                if not results_file == None:
                    results_file.write( str(average_loss/(step+1)) + '\n')
                    if self._options.reps[7] and self._options.reps[8]:
                        results_file.write( str(average_tag_losses/(step+1)) + '\n')
                results_file.flush()

        return average_loss/(step+1)
