from __future__ import print_function
import numpy as np
import time
import pickle
import tensorflow as tf
import os
import sys
from utils.stagging_model import Stagging_Model

        
def run_model(opts, loader = None, epoch=0):
    g = tf.Graph()
    with g.as_default():
        model = Stagging_Model(opts)
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as session: 
            session.run(tf.global_variables_initializer())
            best_accuracy = 0.0
            bad_times = 0
            for i in xrange(opts.max_epochs):
                print('Epoch {}'.format(i+1))
                loss, accuracy = model.run_epoch(session)
                test_accuracy = model.run_epoch(session, True)
                print('test accuracy {}'.format(test_accuracy))
                if best_accuracy < test_accuracy:
                    best_accuracy = test_accuracy 
                    saving_file = os.path.join(opts.model_dir, 'epoch{0}_accuracy{1:.5f}'.format(i+1, test_accuracy))
                    print('saving it to {}'.format(saving_file))
                    saver.save(session, saving_file)
                    bad_times = 0
                    print('test accuracy improving')
                else:
                    bad_times += 1
                    print('test accuracy deteriorating')
                if bad_times >= opts.early_stopping:
                    print('did not improve {} times in a row. stopping early'.format(bad_times))
                    #if saving_dir:
                    #    print('outputting test pred')
                    #    with open(os.path.join(saving_dir, 'predictions_test.pkl'), 'wb') as fhand:
                    #        pickle.dump(predictions, fhand)
                    break 
                
def run_model_test(opts, test_opts):
    g = tf.Graph()
    with g.as_default():
        model = Stagging_Model(opts, test_opts)
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as session: 
            session.run(tf.global_variables_initializer())
            saver.restore(session, test_opts.modelname)
            test_accuracy = model.run_epoch(session, True)
            if test_opts.get_accuracy:
                print('\nTest accuracy {}'.format(test_accuracy))
