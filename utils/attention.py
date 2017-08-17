import tensorflow as tf 

def get_attention_weights(name, units): # no dropout
    weights = {}
    with tf.variable_scope(name) as scope:
        weights['W-attention'] = tf.get_variable('W-attention', [units, units])
        weights['b-attention'] = tf.get_variable('b-attention', [units])
    return weights

def attention_equation(lstm_outputs, cells, weights): 
    shape = tf.shape(lstm_outputs)
    n, b, d = shape[0], shape[1], shape[2]
    lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2]) 
    ## [b, n, d]
    cells = tf.transpose(cells, [1, 0, 2]) 
    ## [b, n, d]
    coefs = tf.matmul(tf.reshape(lstm_outputs, [b*n, d]), weights['W-attention'])+weights['b-attention'] # [bn, d]
    coefs = tf.nn.softmax(tf.matmul(tf.reshape(coefs, [b, n, d]), tf.transpose(lstm_outputs, [0, 2, 1])))
    ## [b, n, n]
    output = cells + tf.reduce_sum(tf.expand_dims(coefs, 3)*tf.expand_dims(lstm_outputs, 1), 2) ## [b, n, d]
    output = tf.transpose(output, [1, 0, 2]) ## [n, b, d]
    return output


