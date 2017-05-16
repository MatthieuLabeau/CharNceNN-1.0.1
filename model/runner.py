# From : https://indico.io/blog/tensorflow-data-input-part2-extensions/
## ====================================================================

import tensorflow as tf
import threading
import numpy as np

class datarunner(object):
    def __init__(self,
                 reps,
                 in_len,
                 out_len,
                 word_len = None):
        self.PhL = []
        self.ShL = []

        if reps[0]:
            self.examples_ph = tf.placeholder(dtype=tf.int64, shape=[None, in_len])
            self.PhL.append(self.examples_ph)
            self.ShL.append([in_len])
        if reps[1]:
            self.examplesChar_ph = tf.placeholder(dtype=tf.int64, shape=[None, in_len, word_len])
            self.PhL.append(self.examplesChar_ph)
            self.ShL.append([in_len, word_len])
        if reps[2]:
            self.labels_ph = tf.placeholder(dtype=tf.int64, shape=[None, out_len])
            self.PhL.append(self.labels_ph)
            self.ShL.append([out_len])
        if reps[3]:
            self.labelsChar_ph = tf.placeholder(dtype=tf.int64, shape=[None, out_len, word_len])
            self.PhL.append(self.labelsChar_ph)
            self.ShL.append([out_len, word_len])
        self.evalLabels_ph = tf.placeholder(dtype=tf.int64, shape=[None, out_len])
        self.PhL.append(self.evalLabels_ph)
        self.ShL.append([out_len])            

        self.queue = tf.RandomShuffleQueue(shapes=self.ShL,
                                           dtypes=[tf.int64] * len(self.ShL),
                                           capacity=2000,
                                           min_after_dequeue=1000)                
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

