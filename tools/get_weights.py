import tensorflow as tf
import sys
import pickle

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph('{}.meta'.format(sys.argv[1]))

    with tf.Session() as sess:
        saver.restore(sess, sys.argv[1])

        #print [var.name for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        word_embeddings = tf.get_collection('word_embeddings')[0].eval()
        with open('word_dict.pkl') as fhand:
            word_dict = pickle.load(fhand)
        idx_to_word = {u: v for v, u in word_dict.items()}
        idx_to_word[0] = "<padding>"
        with open('learned_word_embeddings.txt', 'wt') as fhand:
            for i in xrange(word_embeddings.shape[0]):
                fhand.write(str(idx_to_word[i]))
                fhand.write(' ')
                fhand.write(' '.join(map(str, word_embeddings[i])))
                fhand.write('\n')
        #stag_embeddings = tf.get_collection('stag_embeddings')[0].eval()
        #suffix_embeddings = tf.get_collection('suffix_embeddings')[0].eval()
        #jk_embeddings = tf.get_collection('jk_embeddings')[0].eval()

