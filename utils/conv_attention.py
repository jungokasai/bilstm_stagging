import tensorflow as tf 

def get_attention_weights(name, units): # no dropout
    weights = {}
    window_size = 2
    with tf.variable_scope(name) as scope:
        weights['W-attention'] = tf.get_variable('W-attention', [window_size*2+1, units, units])
        weights['b-attention'] = tf.get_variable('b-attention', [units])
    return weights

#

def attention_equation(lstm_outputs, weights, hidden_prob): 
    shape = tf.shape(lstm_outputs)
    n, b, d = shape[0], shape[1], shape[2]
    lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2]) 
    ## [b, n, d]
    h_conv1 = tf.nn.relu(tf.nn.conv1d(lstm_outputs, weights['W-attention'], stride = 1, padding = 'SAME')+weights['b-attention'])
    h_conv1 = tf.nn.dropout(h_conv1, hidden_prob)
    h_conv1 = tf.transpose(h_conv1, [1, 0, 2])
    ## [b, n, d] => [n, b, d]
    return h_conv1
