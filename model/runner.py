# From : https://indico.io/blog/tensorflow-data-input-part2-extensions/
## ====================================================================

import tensorflow as tf
import threading
import numpy as np

class datarunner(object):
    def __init__(self,
                 reps,
                 capacity,
                 seq_len,
                 tag_number,
                 word_len = None,
                 training = True):
        self.PhL = []
        self.ShL = []

        # Inputs
        if reps[0] or reps[1]:
            self.examples_ph = tf.placeholder(dtype=tf.int64, shape=[None, seq_len])
            self.PhL.append(self.examples_ph)
            self.ShL.append([seq_len])
        if reps[2]:
            self.examplesChar_ph = tf.placeholder(dtype=tf.int64, shape=[None, seq_len, word_len])
            self.PhL.append(self.examplesChar_ph)
            self.ShL.append([seq_len, word_len])
        if reps[3]:
            self.examplesTags_ph = tf.placeholder(dtype=tf.int64, shape=[None, seq_len, tag_number])
            self.PhL.append(self.examplesTags_ph)
            self.ShL.append([seq_len, tag_number])
        # Outputs
        if reps[4] or reps[5]:
            self.labels_ph = tf.placeholder(dtype=tf.int64, shape=[None, seq_len])
            self.PhL.append(self.labels_ph)
            self.ShL.append([seq_len])
        if reps[6]:
            self.labelsChar_ph = tf.placeholder(dtype=tf.int64, shape=[None, seq_len, word_len])
            self.PhL.append(self.labelsChar_ph)
            self.ShL.append([seq_len, word_len])
        if reps[7]:
            self.evalTags_ph = tf.placeholder(dtype=tf.int64, shape=[None, seq_len, tag_number])
            self.PhL.append(self.evalTags_ph)
            self.ShL.append([seq_len, tag_number])
        # Labels
        self.evalLabels_ph = tf.placeholder(dtype=tf.int64, shape=[None, seq_len])
        self.PhL.append(self.evalLabels_ph)
        self.ShL.append([seq_len])
        
        if training:
            self.queue = tf.RandomShuffleQueue(shapes=self.ShL,
                                               dtypes=[tf.int64] * len(self.ShL),
                                               capacity=capacity,
                                               min_after_dequeue= capacity / 5)            
        else:
            self.queue = tf.FIFOQueue(shapes=self.ShL,
                                      dtypes=[tf.int64] * len(self.ShL),
                                      capacity=capacity)
                
        self.enqueue_op = self.queue.enqueue_many(self.PhL)

    def get_inputs(self, batch_size):
        return self.queue.dequeue_many(batch_size)

    def thread_main(self, session, sampler):
        for batches in sampler:
            session.run(self.enqueue_op, feed_dict={pl:batch for pl, batch in zip(self.PhL, batches)})

    def start_threads(self, session, sampler, n_threads=1):
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(session, sampler))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

