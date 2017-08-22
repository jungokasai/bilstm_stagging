import tensorflow as tf

def get_gold_score(scores, prev_correct_so_far, gold, nb_tags):
    ### scores [B, #tags], prev_correct_so_far [B], gold [1]
    gold_score = tf.reduce_sum(scores*tf.expand_dims(prev_correct_so_far, 1)*tf.one_hot(gold, nb_tags)) ## [B, #tags] => [1]
    gold_score = tf.reshape(gold_score, [1])
    return gold_score
    
