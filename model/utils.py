# Ideas from https://danijar.com/variable-sequence-lengths-in-tensorflow/
# and from https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/ops.py
## =======================================================================================
import tensorflow as tf

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name=name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", shape[1:],
                                         initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", shape[1:],
                                        initializer=tf.constant_initializer(0.))
            # Only normalize across the batch - implies shape change of scale and offset (gamma/beta)
            mean, variance = tf.nn.moments(x, [0])
            return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon)

def sequence_mask(sequence, char=False):
    # Character tokens
    if char:
        char_used = tf.sign(sequence)
        words_length = tf.reduce_sum(char_used, 2)
        used = tf.sign(words_length)
        return used, char_used
    # Word tokens
    else:
        used = tf.sign(sequence)
        return used 

def highway(embeddings, tr_weight_list, tr_bias_list, gate_weight_list, gate_bias_list):
    output = embeddings
    for l, (W_tr, b_tr, W_g, b_g) in enumerate(zip(tr_weight_list, tr_bias_list, gate_weight_list, gate_bias_list)):
        """
        if dropout < 1.0:
            output = tf.nn.dropout(tf.nn.relu(tf.matmul(output, W_tr) + b_tr), dropout)
            transform_gate = tf.nn.dropout(tf.sigmoid(tf.matmul(output, W_g) + b_g), dropout)
        else:
        """
        #transform_gate = tf.sigmoid(tf.matmul(output, W_g) + b_g)
        output = tf.nn.relu(tf.matmul(output, W_tr) + b_tr)
        transform_gate = tf.sigmoid(tf.matmul(output, W_g) + b_g)
        output = transform_gate * output + (1. - transform_gate) * embeddings
    return output

def word_char_gate(w_embeddings, c_embeddings, W_g, b_g):
    embeddings = tf.concat(1, [w_embeddings, c_embeddings])
    transform_gate = tf.sigmoid(tf.matmul(embeddings, W_g) + b_g)
    output = transform_gate * w_embeddings + (1. - transform_gate) * c_embeddings
    return output

def CE(char_embeddings, filter_list, pooling_f = tf.nn.max_pool):    
    outputs = []
    shape = char_embeddings.get_shape().as_list()
    # Mixes batch and sequence dimension to get only one batch of batch_size * seq_length words
    # char_embeddings = tf.reshape(char_embeddings, [shape[0] * shape[1], shape[2], shape[3]])
    # Add last dimension for "channels"
    char_embeddings = tf.expand_dims(char_embeddings, -1)
    for H in filter_list:
        # Add dimension for "channels"
        H = tf.expand_dims(H, 2)        
        # conv is of shape [batch_size * seq_length, word_length - window_size + 1, 1, filter_dim]
        conv = tf.nn.conv2d(char_embeddings, H, strides=[1, 1, 1, 1], padding='VALID')
        # Pooling is reducing the 2nd dimension - by whatever reduce function wanted
        pool = pooling_f(tf.tanh(conv), [1, conv.get_shape().as_list()[1],1,1], [1,1,1,1], 'VALID')
        # supress empty dim and reshape
        filtered = tf.reshape(tf.squeeze(pool), [shape[0], -1])
        outputs.append(filtered)
    if len(filter_list) > 1:
        output = tf.concat(1, outputs)
    else:
        output = outputs[0]
    return output

def CE_RNN(char_embeddings, cell_fw, cell_bw, mask, name = "biRNN"):
    shape = char_embeddings.get_shape().as_list()
    (c_input_emb_fw,
     c_input_emb_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                          cell_bw,
                                                          char_embeddings,
                                                          mask,
                                                          dtype='float32',
                                                          scope = name)
    batch_indexes = tf.cast(tf.range(shape[0]),'int64')
    length_indexes = tf.maximum(mask - 1, 0)
    indexes = length_indexes + shape[1] * batch_indexes
    output = tf.gather(tf.reshape(tf.concat(2,
                                            [c_input_emb_fw, c_input_emb_bw]),
                                  [shape[0] * shape[1], -1]),
                       indexes)
    return output

def restrict_voc(samples, threshold):
    samples_clipped = tf.select(tf.less(samples,threshold), samples, tf.ones_like(samples))
    return samples_clipped
