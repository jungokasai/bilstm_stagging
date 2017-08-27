import tensorflow as tf 

def get_attention_weights(name, units): # no dropout
    weights = {}
    with tf.variable_scope(name) as scope:
        weights['W-attention'] = tf.get_variable('W-attention', [units, units])
        weights['W-context'] = tf.get_variable('W-context', [2*units, units])
    return weights

#def attention_equation(lstm_outputs, cells, weights): 
#    shape = tf.shape(lstm_outputs)
#    n, b, d = shape[0], shape[1], shape[2]
#    lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2]) 
#    ## [b, n, d]
#    cells = tf.transpose(cells, [1, 0, 2]) 
#    ## [b, n, d]
#    coefs = tf.matmul(tf.reshape(lstm_outputs, [b*n, d]), weights['W-attention'])+weights['b-attention'] # [bn, d]
#    coefs = tf.nn.softmax(tf.matmul(tf.reshape(coefs, [b, n, d]), tf.transpose(lstm_outputs, [0, 2, 1])))
#    ## [b, n, n]
#    output = cells + tf.reduce_sum(tf.expand_dims(coefs, 3)*tf.expand_dims(lstm_outputs, 1), 2) ## [b, n, d]
#    output = tf.transpose(output, [1, 0, 2]) ## [n, b, d]
#    return output
#

def attention_equation(forward_h, backward_hs, backward_h, weights): 
    ## forward_h [b, d]
    ## backward_hs [n, b, d]
    ## backward_h [b, d]
    shape = tf.shape(backward_hs)
    n, b, d = shape[0], shape[1], shape[2]
    backward_hs = tf.transpose(backward_hs, [1, 0, 2]) 
    ## [b, n, d]
    context_vec = tf.map_fn(lambda x: get_context(x, weights), [backward_hs, forward_h]) ## [b, d]
    output = tf.nn.tanh(tf.matmul(tf.concat([context_vec, backward_h], 1]), weights['W-context'])) 
    ### [b, 2d] x [2d, d] => [b, d]
    return output

def get_context(inputs, weights):
    backward_hs_sent = inputs[0]  ## [n, d]
    forward_h_sent = inputs[1] # [d]
    ## lstm_outputs_sent [n, d]
    coefs = tf.matmul(tf.matmul(forward_h_sent, weights['W-attention']), tf.transpose(backward_hs_sent, [1, 0])) # [n, 1]
    ### [d] x [d, d] => [d]
    ### [d] x [d, n] => [n]
    coefs = tf.nn.softmax(coefs) ## [n]
    #output_sent = tf.reduce_sum(coefs*tf.expand_dims(lstm_outputs_sent, 0), 1) # [n, d]
    context_vec = tf.matmul(coefs, backward_hs_sent)
    ### [n] x [n, d] => d
    return context_vec

