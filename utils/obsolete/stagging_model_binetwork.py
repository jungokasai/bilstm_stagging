from __future__ import print_function
from stagging_model import Stagging_Model
#import matplotlib
from data_process_secsplit import Dataset
from lstm import get_lstm_weights, lstm
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
import os
import sys


class Stagging_Model_Binetwork(Stagging_Model):

    def add_stag_embedding_mat(self):
        with tf.variable_scope('stag_embedding') as scope:
            self.stag_embedding_mat = tf.get_variable('stag_embedding_mat', [self.loader.nb_tags+1, self.opts.lm]) # +1 for padding

    def add_stag_dropout_mat(self, batch_size):
        self.stag_dropout_mat = tf.ones([batch_size, self.opts.lm])
        self.stag_dropout_mat = tf.nn.dropout(self.stag_dropout_mat, self.input_keep_prob)
    def add_lstm(self, inputs, i, name): ## need to access c
        prev_init = tf.zeros([2, tf.shape(inputs)[1], self.opts.units])  # [2, batch_size, num_units]
        #prev_init = tf.zeros([2, 100, self.opts.units])  # [2, batch_size, num_units] if i == 0:
            inputs_dim = self.inputs_dim
        else:
            inputs_dim = self.opts.units
        weights = get_lstm_weights('{}_LSTM_layer{}'.format(name, i), inputs_dim, self.opts.units, tf.shape(inputs)[1], self.hidden_prob)
        cell_hidden = tf.scan(lambda prev, x: lstm(prev, x, weights), inputs, prev_init)
         #cell_hidden [seq_len, 2, batch_size, units]
        c = tf.unstack(cell_hidden, 2, axis=1)[0] #[seq_len, batch_size, units]
        h = tf.unstack(cell_hidden, 2, axis=1)[1] #[seq_len, batch_size, units]
        return c, h


    def add_binetwork(self, lstm_outputs):
        ## lstm_outputs: [n, b, d]
        shape = lstm_outputs
        weights = get_uni_weights('attention', self.outputs_dim)
        self.add_stag_embedding_mat()
        return outputs

    def add_uni_path(self, lstm_outputs):
        batch_size = tf.shape(lstm_outputs)[1]
        prev_init = [tf.zeros([batch_size], tf.int32), tf.zeros([batch_size, self.loader.nb_tags])]
        ## We need the following memory states (list of four elements): 
        ## 1. Previous predictions (stag_idx): [batch_size]
        ## 2. In addition, though it's not a memory state, we also add projected_outputs for calculation of loss: [batch_size, outputs_dim]
        ## Define all the necessary weights for recursion
        self.add_stag_dropout_mat(batch_size)
        ###

        all_states = tf.scan(lambda prev, x: self.add_uni_one(prev, x, weights_list), lstm_outputs, prev_init)
        all_predictions = all_states[1] # [seq_len, batch_size]
        all_predictions = tf.transpose(all_predictions, perm=[1, 0]) # [batch_size, seq_len]
        all_projected_outputs = all_states[3] # [seq_len, batch_size, outputs_dim]
        all_projected_outputs = tf.transpose(all_projected_outputs, perm=[1, 0, 2]) # [batch_size, seq_len, outputs_dim]
        return all_predictions, all_projected_outputs
    def add_uni_one_forward(self, prev_list, bi_h, uni_weights_list, backward_embeddings):
        ## compute one word in the forward direction
        projected_outputs = self.add_projection(tf.nn.relu(bi_h + tf.matmul(weights_list['W-prev']))) ## [batch_size, nb_tags]
        predictions = self.add_predictions(projected_outputs) ## [batch_sizes]
        new_state = [predictions, projected_outputs]
        return new_state
    
    def __init__(self, opts, test_opts=None):
       
        self.opts = opts
        self.test_opts = test_opts
        self.loader = Dataset(opts, test_opts)
        self.batch_size = 100
        self.add_placeholders()
        self.inputs_dim = self.opts.embedding_dim + self.opts.suffix_dim + self.opts.cap + self.opts.num + self.opts.jk_dim
        self.outputs_dim = (1+self.opts.bi)*self.opts.units
        inputs_list = [self.add_word_embedding()]
        if self.opts.suffix_dim > 0:
            inputs_list.append(self.add_suffix_embedding())
        if self.opts.cap:
            inputs_list.append(self.add_cap())
        if self.opts.num:
            inputs_list.append(self.add_num())
        if self.opts.jk_dim > 0:
            inputs_list.append(self.add_jackknife_embedding())
        inputs_tensor = tf.concat(inputs_list, 2) ## [seq_len, batch_size, inputs_dim]
        forward_inputs_tensor = self.add_dropout(inputs_tensor, self.input_keep_prob)
        for i in xrange(self.opts.num_layers):
            c, h = self.add_lstm(forward_inputs_tensor, i, 'Forward') ## [seq_len, batch_size, units]
            forward_inputs_tensor = self.add_dropout(h, self.keep_prob) ## [seq_len, batch_size, units]
        lstm_outputs = forward_inputs_tensor
        cells = self.add_dropout(c, self.keep_prob)
        if self.opts.bi:
            backward_inputs_tensor = self.add_dropout(tf.reverse(inputs_tensor, [0]), self.input_keep_prob)
            for i in xrange(self.opts.num_layers):
                c, h = self.add_lstm(backward_inputs_tensor, i, 'Backward')## [seq_len, batch_size, units]
                backward_inputs_tensor = self.add_dropout(h, self.keep_prob) ## [seq_len, batch_size, units]
            backward_inputs_tensor = tf.reverse(backward_inputs_tensor, [0])
            lstm_outputs = tf.concat([lstm_outputs, backward_inputs_tensor], 2) ## [seq_len, batch_size, outputs_dim]
        #lstm_outputs = self.add_attention(lstm_outputs, cells)
        #lstm_outputs = self.add_dropout(lstm_outputs, self.keep_prob)
        projected_outputs = self.add_binetwork(lstm_outputs)
        
        projected_outputs = tf.transpose(projected_outputs, perm=[1, 0, 2]) # [batch_size, seq_len, nb_tags]
        self.weight = tf.cast(tf.not_equal(self.inputs_placeholder_list[0], tf.zeros(tf.shape(self.inputs_placeholder_list[0]), tf.int32)), tf.float32) ## [batch_size, seq_len]
        self.loss = self.add_loss_op(projected_outputs)
        self.train_op = self.add_train_op(self.loss)
        self.add_accuracy(projected_outputs)

