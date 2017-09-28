import tensorflow as tf

def crf(crf_inputs, crf_weights):
    ## crf_inputs: [batch_size, lm]
    ## crf_weights: [batch_size, lm, lm+1]
    crf_weight = tf.slice(crf_weights, [0, 0, 1], [-1, -1, -1]) #[batch_size, lm, lm]
    crf_bias = tf.slice(crf_weights, [0, 0, 0], [-1, -1, 1]) #[batch_size, lm, 1]
    crf_bias = tf.squeeze(crf_bias, axis=2)
    outputs = tf.map_fn(lambda x: batch_projection(x), [crf_inputs, crf_weight, crf_bias], dtype=tf.float32) ## [batch_size, lm]
    return outputs

def batch_projection(crf_inputs_list):
    crf_inputs = tf.expand_dims(crf_inputs_list[0], 0) ##[1, lm]
    crf_weight = crf_inputs_list[1] ##[lm, lm]
    crf_bias = crf_inputs_list[2] ##[lm]
    output = tf.matmul(crf_inputs, crf_weight) + crf_bias ## [1, lm]
    output = tf.squeeze(output, axis=0) ## [lm]
    return output

