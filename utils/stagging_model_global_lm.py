from __future__ import print_function
from data_process_secsplit import Dataset
from stagging_model_lm import Stagging_Model_LM
from lstm import get_lstm_weights, lstm
from back_tracking import back_track
import numpy as np
import tensorflow as tf
import os
import sys
import time


class Stagging_Model_Global_LM(Stagging_Model_LM):

    def get_global_loss(self, global_loss):
        ## global_loss [n-1, b]
        global_loss = tf.reduce_mean(global_loss[-1])
        return global_loss

## Supertagging
## Beware of the following notation: batch_size = self.batch_size*beam_size 
    def add_forward_beam_path(self, forward_inputs_tensor, backward_embeddings, beam_size):
        batch_size = tf.shape(forward_inputs_tensor)[1] ## batch_size = self.batch_size = b
        prev_init = [tf.zeros([2, batch_size, self.opts.num_layers*self.opts.units]), tf.zeros([batch_size], tf.int32), 0, tf.zeros([batch_size, 1]), tf.zeros([batch_size], tf.int32), tf.zeros([batch_size]), tf.ones([batch_size, 1])]
        gold = tf.transpose(self.inputs_placeholder_list[5], [1, 0])
        ## [b, n] => [n, b]
        non_paddings = tf.transpose(self.weight, [1, 0])
        ## [b, n] => [n, b]
	first_gold = tf.squeeze(tf.slice(gold, [0, 0], [1, -1]), axis=0) # [b]
	first_non_paddings = tf.squeeze(tf.slice(non_paddings, [0, 0], [1, -1]), axis=0) # [b]
	gold = tf.slice(gold, [1, 0], [-1, -1]) # [n-1, b]
	non_paddings = tf.slice(non_paddings, [1, 0], [-1, -1]) #[n-1, b]
        ## We need the following memory states (list of four elements): 
        ## 1. LSTM cell and h memories for each layer: [2, batch_size, units*num_layers] 
        ## 2. Previous predictions (stag_idx): [batch_size] ## notice the difference between beam and greedy here
        ## 3. Time step for referencing backward path: int
        ## 4. For beam search, we also need to memorize scores: [batch_size]
        ## 5. Backpointer (Parent indices) for predictions
        ## 6. Global Loss
        ## 7. Correct So Far
        name = 'Forward'
        ## Define all the necessary weights for recursion
        lstm_weights_list = []
        for i in xrange(self.opts.num_layers):
            if i == 0:
                inputs_dim = self.inputs_dim + self.opts.lm
            else:
                inputs_dim = self.opts.units
            lstm_weights_list.append(get_lstm_weights('{}_LSTM_layer{}'.format(name, i), inputs_dim, self.opts.units, batch_size, self.hidden_prob, beam_size))
        self.add_stag_embedding_mat()
        #self.add_stag_dropout_mat(batch_size) ## unnecessary since we are only testing
        ## First Iteration has only self.batch_size configurations. For the sake of tf.scan function, calculate the first. 
        first_inputs = tf.squeeze(tf.slice(forward_inputs_tensor, [0, 0, 0], [1, -1, -1]), axis=0) ## [batch_size, inputs_dim+lm]
        forward_inputs_tensor = tf.slice(forward_inputs_tensor, [1, 0, 0], [-1, -1, -1])
        prev_init = self.add_one_beam_forward(prev_init, [first_inputs, first_non_paddings, first_gold], lstm_weights_list, backward_embeddings, beam_size, batch_size) 
        first_predictions = tf.expand_dims(prev_init[1], 0) ## [1, batch_size]
        first_scores = tf.expand_dims(prev_init[3], 0) ## [1, batch_size, 1]

        ## Now, move on to the second iteration and beyond
        initial_shape = tf.shape(forward_inputs_tensor)
        forward_inputs_tensor = tf.reshape(tf.tile(forward_inputs_tensor, [1, 1, beam_size]), [initial_shape[0], initial_shape[1]*beam_size, initial_shape[2]])
        ## [seq_len-1, self.batch_size, inputs_dim] -> [seq_len-1, self.batch_size*beam_size (B*b), inputs_dim]
        batch_size = initial_shape[1]*beam_size ## Bb
        all_states = tf.scan(lambda prev, x: self.add_one_beam_forward(prev, x, lstm_weights_list, backward_embeddings, beam_size, batch_size, True), [forward_inputs_tensor, non_paddings, gold], prev_init, back_prop=True) ## no backprop for testing reuse projection weights from the first iteration
        back_pointers = all_states[4] # [seq_len-1, batch_size]
        back_pointers = tf.transpose(back_pointers, perm=[1, 0])
        all_predictions = all_states[1] # [seq_len-1, batch_size]
        all_predictions = tf.concat([first_predictions, all_predictions], 0)
        all_predictions = tf.transpose(all_predictions, perm=[1, 0]) # [batch_size, seq_len]
        all_scores = all_states[3] # [seq_len-1, batch_size, 1]
        all_scores = tf.concat([first_scores, all_scores], 0)
        all_scores = tf.squeeze(all_scores, axis=2)
        all_scores = tf.transpose(all_scores, perm=[1, 0])
        self.loss = self.get_global_loss(all_states[5])
        return all_predictions, all_scores, back_pointers
    def add_one_beam_forward(self, prev_list, inputs, lstm_weights_list, backward_embeddings, beam_size, batch_size, post_first=False):
	x = inputs[0] ## [b, d]
	non_paddings = inputs[1] ## [b]
	gold = tf.tile(tf.reshape(inputs[2], [-1, 1]), [1, beam_size]) ## [b] => [b, B]
        ## compute one word in the forward direction
        prev_cell_hiddens = prev_list[0] ## [2, batch_size, units*num_layers]
        prev_cell_hidden_list = tf.split(prev_cell_hiddens, self.opts.num_layers, axis=2) ## [[2, batch_size, units] x num_layers]
        prev_predictions = prev_list[1] ## [batch_size]
        time_step = prev_list[2]  ## 0D
        prev_scores = prev_list[3] ## [batch_size (self.batch_size*beam_size), 1]
        prev_prediction = prev_list[4] ## [b]
        prev_global_loss = prev_list[5] ## [b, 1]
        prev_correct_so_far = prev_list[6] ## [Bb, 1]
        prev_embedding = self.add_stag_embedding(prev_predictions) ## [batch_size, inputs_dim]
        #prev_embedding = prev_embedding*self.stag_dropout_mat
        h = tf.concat([x, prev_embedding], 1) ## [batch_size, inputs_dim + lm]
        cell_hiddens = []
        for i in xrange(self.opts.num_layers):
            weights = lstm_weights_list[i]
            cell_hidden = lstm(prev_cell_hidden_list[i], h, weights, post_first) ## [2, batch_size, units]
            cell_hiddens.append(cell_hidden)
            h = tf.unstack(cell_hidden, 2, axis=0)[1] ## [batch_size, units]
        cell_hiddens = tf.concat(cell_hiddens, 2) ## [2, batch_size, units*num_layers]
        with tf.device('/cpu:0'):
            backward_h = tf.nn.embedding_lookup(backward_embeddings, time_step) ## [self.batch_size, units]
        if post_first: ## batch_size = self.batch_size*beam_size
            backward_h = tf.reshape(tf.tile(backward_h, [1, beam_size]), [batch_size, -1]) ## [batch_size, units]
        bi_h = tf.concat([h, backward_h], 1) ## [batch_size, outputs_dim]
        projected_outputs = self.add_projection(bi_h, post_first) ## [batch_size, nb_tags]
        scores, indices = self.add_top_k(projected_outputs, prev_scores, beam_size, post_first) ## [self.batch_size, beam_size], [self.batch_size, beam_size]
        #scores = tf.stop_gradient(scores)
        #indices  = tf.stop_gradient(indices)
        predictions = indices % self.loader.nb_tags ##[b, B]

        if post_first:
            parent_indices = tf.reshape(tf.range(0, batch_size, beam_size), [-1, 1]) + indices//self.loader.nb_tags ## [self.batch_size, 1] + [self.batch_size, beam_size]
            parent_indices = tf.reshape(parent_indices, [-1]) ## [self.batch_size*beam_size (batch_size)]
            cell_hiddens = tf.transpose(cell_hiddens, [1, 0, 2]) ## [batch_size, 2, units*num_layers]
            with tf.device('/cpu:0'):
                cell_hiddens = tf.nn.embedding_lookup(cell_hiddens, parent_indices) ## [batch_size, 2, units*num_layers] 
                prev_correct_so_far = tf.nn.embedding_lookup(prev_correct_so_far, parent_indices) ## [batch_size, 1]
                
            cell_hiddens = tf.transpose(cell_hiddens, [1, 0, 2]) ## [2, batch_size, units*num_layers]
            correct_so_far = tf.cast(tf.equal(predictions, gold), tf.float32)*tf.expand_dims(non_paddings, 1)*tf.reshape(prev_correct_so_far, [-1, beam_size]) ## [b, B]
        else:
            parent_indices = tf.zeros([batch_size*beam_size], tf.int32) ## Dummy parent indices for the first iteration. We know parents for the first iteration
            cell_hiddens = tf.reshape(tf.tile(cell_hiddens, [1, 1, beam_size]), [2, batch_size*beam_size, -1])
            correct_so_far = tf.cast(tf.equal(predictions, gold), tf.float32) ## [b, B]
	current_global_loss = tf.reduce_sum(correct_so_far*scores, axis=1) - tf.reduce_sum(scores, axis=1) # [b]
        nbs_correct = tf.reduce_sum(correct_so_far, axis=1) #[b]
	global_loss = current_global_loss*tf.cast(tf.equal(nbs_correct, tf.ones(tf.shape(nbs_correct))), tf.float32) + prev_global_loss*tf.cast(tf.equal(nbs_correct, tf.zeros(tf.shape(nbs_correct))), tf.float32) ## [b]
        time_step += 1
        predictions = tf.reshape(predictions, [-1]) ## [batch_size]
        scores = tf.reshape(scores, [-1, 1]) ## [batch_size, 1]
        correct_so_far = tf.reshape(correct_so_far, [-1, 1]) ## [batch_size, 1]
        new_state = [cell_hiddens, predictions, time_step, scores, parent_indices, global_loss, correct_so_far]
        return new_state

       
    def __init__(self, opts, test_opts=None, beam_size=16):
        ## Notation:
        ## b: batch_size
        ## d: # units
        ## n: # tokens in the sentence
        ## B: beam_size
        self.opts = opts
        self.test_opts = test_opts
        self.loader = Dataset(opts, test_opts)
        self.batch_size = 100
        self.beam_size = beam_size
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
        forward_inputs_tensor = inputs_tensor

        ## Backward path is deterministic, just run it first and make it embeddings
        if self.opts.bi:
            backward_inputs_tensor = self.add_dropout(tf.reverse(inputs_tensor, [0]), self.input_keep_prob)
            for i in xrange(self.opts.num_layers):
                backward_inputs_tensor = self.add_dropout(self.add_lstm(backward_inputs_tensor, i, 'Backward'), self.keep_prob) ## [seq_len, batch_size, units]
            backward_inputs_tensor = tf.reverse(backward_inputs_tensor, [0])  ## [seq_len, batch_size, units]
        ## backward path is done
        forward_inputs_tensor = self.add_dropout(forward_inputs_tensor, self.input_keep_prob)

        if beam_size > 0:
            self.weight = tf.cast(tf.not_equal(self.inputs_placeholder_list[0], tf.zeros(tf.shape(self.inputs_placeholder_list[0]), tf.int32)), tf.float32) # [self.batch_size, seq_len]
            self.predictions, self.scores, self.back_pointers = self.add_forward_beam_path(forward_inputs_tensor, backward_inputs_tensor, beam_size) ## [seq_len, batch_size, nb_tags]
            self.weight_beam = tf.reshape(tf.tile(self.weight, [1, beam_size]), [-1, tf.shape(self.weight)[1]]) # [batch_size, seq_len]
            self.accuracy = self.loss ## for dummy
            self.train_op = self.add_train_op(self.loss)
        else:
            self.predictions, projected_outputs = self.add_forward_path(forward_inputs_tensor, backward_inputs_tensor) ## [seq_len, batch_size, nb_tags]
            self.weight = tf.cast(tf.not_equal(self.inputs_placeholder_list[0], tf.zeros(tf.shape(self.inputs_placeholder_list[0]), tf.int32)), tf.float32) ## [batch_size, seq_len]
            self.add_lm_accuracy()
            self.loss = self.add_loss_op(projected_outputs)
            self.train_op = self.add_train_op(self.loss)
#        if self.opts.bi:
