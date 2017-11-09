import tensorflow as tf 

def get_chain_weights(name, inputs_dim, units, reuse=False):
    weights = {}
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
	weights['L_weight'] = tf.get_variable('L_weight', [inputs_dim, units])
	weights['L_bias'] = tf.get_variable('L_bias', [units])
    return weights

