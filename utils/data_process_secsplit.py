import numpy as np
from preprocessing import Tokenizer
from preprocessing import pad_sequences 
from mica.nbest import output_mica_nbest
import os
import sys
import pickle
import random
import os
import io

np.random.seed(1234)


class Dataset(object):

    def __init__(self, opts, test_opts=None):
        path_to_text = opts.text_train
        path_to_tag = opts.tag_train
        path_to_jk = opts.jk_train
        if test_opts is None:
            path_to_text_test = opts.text_test
            path_to_tag_test = opts.tag_test
            path_to_jk_test = opts.jk_test
        else:
            path_to_text_test = test_opts.text_test
            path_to_tag_test = test_opts.tag_test
            path_to_jk_test = test_opts.jk_test

        self.inputs_train = {}
        self.inputs_test = {}

        ## indexing sents files
        if opts.pretrained:
            with open() as fin:
                tokenizer = pickle.load(fin)
        else:
            f_train = io.open(path_to_text, encoding='utf-8')
            texts = f_train.readlines()
            self.nb_train_samples = len(texts)
            f_train.close()
            tokenizer = Tokenizer(lower=True)
            tokenizer.fit_on_texts(texts)
        #print(tokenizer.word_index['-unseen-'])
        self.word_index = tokenizer.word_index
        sorted_freqs = tokenizer.sorted_freqs
        self.nb_words = len(self.word_index)
        print('Found {} unique lowercased words including -unseen-.'.format(self.nb_words))

        # lookup the glove word embeddings
        # need to reserve indices for testing file. 
        glove_size = opts.embedding_dim
        self.embeddings_index = {}
        print('Indexing word vectors.')
        #f = open('glovevector/glove.6B.{}d.txt'.format(glove_size))
        f = io.open(opts.word_embeddings_file, encoding='utf-8')
        for line in f:
            values = line.strip().split(' ')
            if len(values) == opts.embedding_dim+1:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        f.close()

        print('Found {} word vectors.'.format(len(self.embeddings_index)))

        unseens = list(set(self.embeddings_index.keys()) - set(self.word_index.keys())) ## list of words that appear in glove but not in the training set
        nb_unseens = len(unseens)
        print('Found {} words not in the training set'.format(nb_unseens))

        self.word_embeddings = np.zeros((self.nb_words+1+nb_unseens, glove_size)) ## +1 for padding (idx 0)

        ## Get Frequencies for Adversarial Training (Yasunaga et al. 2017)
        self.word_freqs = np.zeros([self.nb_words+1+nb_unseens])
        self.word_freqs[1:self.nb_words] = sorted_freqs ## Skip Zero Padding (Index 0)
        self.word_freqs = self.word_freqs.astype(np.float32)
        self.word_freqs = self.word_freqs/np.sum(self.word_freqs)
        for word, i in self.word_index.items(): ## first index the words in the training set
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None: ## otherwise zero vector
                self.word_embeddings[i] = embedding_vector
        for unseen in unseens:
            self.word_index[unseen] = len(self.word_index) + 1 ## add unseen words to the word_index dictionary
            self.word_embeddings[self.word_index[unseen]] = self.embeddings_index[unseen]
        self.idx_to_word = invert_dict(self.word_index)
        print('end glove indexing')
        f_test = io.open(path_to_text_test, encoding='utf-8')
        texts = texts +  f_test.readlines()
        self.nb_validation_samples = len(texts) - self.nb_train_samples
        f_test.close()
        text_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_word[x], text_sequences[self.nb_train_samples]))
        self.inputs_train['words'] = text_sequences[:self.nb_train_samples]
        self.inputs_test['words'] = text_sequences[self.nb_train_samples:]
        ## indexing sents files ends
        ## indexing suffixes 
        if opts.suffix_dim > 0:
            suffix = tokenizer.suffix_extract(texts)
            suffix_tokenizer = Tokenizer()
            suffix_tokenizer.fit_on_texts(suffix[:self.nb_train_samples], non_split=True)
            self.suffix_index = suffix_tokenizer.word_index
            self.nb_suffixes = len(self.suffix_index)
            sorted_freqs = suffix_tokenizer.sorted_freqs
            self.suffix_freqs = np.zeros([self.nb_suffixes+1]).astype(np.float32) ## +1 for zero padding
            self.suffix_freqs[1:self.nb_suffixes] = sorted_freqs ## Skip Zero Padding (Index 0)
            self.suffix_freqs = self.suffix_freqs/np.sum(self.suffix_freqs)
            self.idx_to_suffix = invert_dict(self.suffix_index)
            print('Found {} unique suffixes including -unseen-.'.format(self.nb_suffixes))
            suffix_sequences = suffix_tokenizer.texts_to_sequences(suffix, non_split=True)
            #print(map(lambda x: self.idx_to_suffix[x], suffix_sequences[self.nb_train_samples]))
            self.inputs_train['suffix'] = suffix_sequences[:self.nb_train_samples]
            self.inputs_test['suffix'] = suffix_sequences[self.nb_train_samples:]
            ## indexing suffixes ends
        ## indexing capitalization 
        if opts.cap:
            cap_sequences = tokenizer.cap_indicator(texts)
            #print(cap_sequences[self.nb_train_samples])
            self.inputs_train['cap'] = cap_sequences[:self.nb_train_samples]
            self.inputs_test['cap'] = cap_sequences[self.nb_train_samples:]
            ## indexing capitalization ends
            ## indexing numbers
        if opts.num:
            num_sequences = tokenizer.num_indicator(texts)
            #print(num_sequences[self.nb_train_samples])
            self.inputs_train['num'] = num_sequences[:self.nb_train_samples]
            self.inputs_test['num'] = num_sequences[self.nb_train_samples:]
            ## indexing numbers ends
        ## indexing jackknife files
        if opts.jk_dim > 0:
            f_train = io.open(path_to_jk, encoding='utf-8')
            texts = f_train.readlines()
            f_train.close()
            tokenizer = Tokenizer(lower=False) 
            tokenizer.fit_on_texts(texts)
            self.jk_index = tokenizer.word_index
            self.nb_jk = len(self.jk_index)
            sorted_freqs = tokenizer.sorted_freqs
            self.jk_freqs = np.zeros([self.nb_jk+1]).astype(np.float32) ## +1 for zero padding
            self.jk_freqs[1:self.nb_jk] = sorted_freqs ## Skip Zero Padding (Index 0)
            self.jk_freqs = self.jk_freqs/np.sum(self.jk_freqs)
            self.idx_to_jk = invert_dict(self.jk_index)
            print('Found {} unique tags including -unseen-.'.format(self.nb_jk))
            f_test = io.open(path_to_jk_test, encoding='utf-8')
            texts = texts + f_test.readlines() ## do not lowercase tCO
            f_test.close()
            jk_sequences = tokenizer.texts_to_sequences(texts)
            #print(map(lambda x: self.idx_to_jk[x], jk_sequences[self.nb_train_samples]))
            self.inputs_train['jk'] = jk_sequences[:self.nb_train_samples]
            self.inputs_test['jk'] = jk_sequences[self.nb_train_samples:]
            ## indexing jackknife files ends
        ## indexing char files
        if opts.chars_dim > 0:
            f_train = io.open(path_to_text, encoding='utf-8')
            texts = f_train.readlines()
            f_train.close()
            tokenizer = Tokenizer(lower=False,char_encoding=True) 
            tokenizer.fit_on_texts(texts)
            self.char_index = tokenizer.word_index
            self.nb_chars = len(self.char_index)
            sorted_freqs = tokenizer.sorted_freqs
            self.char_freqs = np.zeros([self.nb_chars+1]).astype(np.float32) ## +1 for zero padding
            self.char_freqs[1:self.nb_chars] = sorted_freqs ## Skip Zero Padding (Index 0)
            self.char_freqs = self.char_freqs/np.sum(self.char_freqs)
            self.idx_to_char = invert_dict(self.char_index)
            print('Found {} unique characters including -unseen-.'.format(self.nb_chars))
            f_test = io.open(path_to_text_test, encoding='utf-8')
            texts = texts + f_test.readlines() ## do not lowercase tCO
            f_test.close()
            char_sequences = tokenizer.texts_to_sequences(texts)
            #print(map(lambda x: self.idx_to_jk[x], jk_sequences[self.nb_train_samples]))
            self.inputs_train['chars'] = char_sequences[:self.nb_train_samples]
            self.inputs_test['chars'] = char_sequences[self.nb_train_samples:]
            ## indexing char files ends
        ## indexing stag files
        f_train = open(path_to_tag)
        texts = f_train.readlines()
        f_train.close()
        tokenizer = Tokenizer(lower=False) ## for tCO
        tokenizer.fit_on_texts(texts, zero_padding=False)
        #print(tokenizer.word_index['-unseen-'])
        self.tag_index = tokenizer.word_index
        self.nb_tags = len(self.tag_index)
        self.idx_to_tag = invert_dict(self.tag_index)
        print('Found {} unique tags including -unseen-.'.format(self.nb_tags))
        f_test = open(path_to_tag_test)
        texts = texts + f_test.readlines() ## do not lowercase tCO
        f_test.close()
        tag_sequences = tokenizer.texts_to_sequences(texts)
        #print(map(lambda x: self.idx_to_tag[x], tag_sequences[self.nb_train_samples+8]))
        self.inputs_train['tags'] = tag_sequences[:self.nb_train_samples]
        self.inputs_test['tags'] = tag_sequences[self.nb_train_samples:]

        ## indexing stag files ends
        self.test_gold = np.hstack(tag_sequences[self.nb_train_samples:]) ## for calculation of accuracy
        ## padding the train inputs and test inputs
        #self.inputs_train = [pad_sequences(x) for x in self.inputs_train]
        self.inputs_train = {key: pad_sequences(x, key) for key, x in self.inputs_train.items()}
        random.seed(0)
        perm = np.arange(self.nb_train_samples)
        random.shuffle(perm)
        self.inputs_train = {key: x[perm] for key, x in self.inputs_train.items()}
        #self.inputs_train = [x[perm] for x in self.inputs_train]

        #self.inputs_test = [pad_sequences(x) for x in self.inputs_test]
        self.inputs_test = {key: pad_sequences(x, key) for key, x in self.inputs_test.items()}

        ## setting the current indices
        self._index_in_epoch = 0
        self._epoch_completed = 0
        self._index_in_test = 0

#        indices = np.arange(self.nb_train_added)
#        np.random.shuffle(indices)
#
#        self.X_train = self.X_train[indices]
#        if self.opts.jackknife:
#            self.jk_labels = self.jk_labels[indices]
#
#        self.train_cap_indicator = self.train_cap_indicator[indices]
#        self.train_num_indicator = self.train_num_indicator[indices]
#        self.suffix_train = self.suffix_train[indices]
#        self.y_train = self.y_train[indices]
#        if self.opts.joint:
#            self.pos_train = self.pos_train[indices]
#
    def next_batch(self, batch_size):

#        start = self._index_in_epoch
#        if self._index_in_epoch >= self.nb_train_samples:
#                # iterate until the very end do not throw away
#            self._index_in_epoch = 0
#            self._epoch_completed+=1
#            perm = np.arange(self.nb_train_samples)
#            random.shuffle(perm)
#            self.inputs_train = [x[perm] for x in self.inputs_train]
#            return False
#        self._index_in_epoch += batch_size
#        end = self._index_in_epoch
#        self.inputs_train_batch = []
#        for i, x in enumerate(self.inputs_train):
#            x_batch = x[start:end]
#            if i == 0:
#                max_len = np.max(np.sum(x_batch!=0, axis=-1))
#            self.inputs_train_batch.append(x_batch[:, :max_len])
#        return True
        start = self._index_in_epoch
        if self._index_in_epoch >= self.nb_train_samples:
                # iterate until the very end do not throw away
            self._index_in_epoch = 0
            self._epoch_completed+=1
            perm = np.arange(self.nb_train_samples)
            random.shuffle(perm)
            self.inputs_train = {key: x[perm] for key, x in self.inputs_train.items()}
            return False
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        self.inputs_train_batch = {}
        x = self.inputs_train['words']
        x_batch = x[start:end]
        max_len = np.max(np.sum(x_batch!=0, axis=1))
        for key, x in self.inputs_train.items():
            x_batch = x[start:end]
            if key == 'chars':
                max_word_len = np.max(np.sum(x_batch!=0, axis=2))
                self.inputs_train_batch[key] = x_batch[:, :max_len, :max_word_len]
            else:
                self.inputs_train_batch[key] = x_batch[:, :max_len]
        return True

    def next_test_batch(self, batch_size):

#        start = self._index_in_test
#        if self._index_in_test >= self.nb_validation_samples:
#                # iterate until the very end do not throw away
#            self._index_in_test = 0
#            return False
#        self._index_in_test += batch_size
#        end = self._index_in_test
#        self.inputs_test_batch = []
#        for i, x in enumerate(self.inputs_test):
#            x_batch = x[start:end]
#            if i == 0:
#                max_len = np.max(np.sum(x_batch!=0, axis=-1))
#            self.inputs_test_batch.append(x_batch[:, :max_len])
#        return True
        start = self._index_in_test
        if self._index_in_test >= self.nb_validation_samples:
                # iterate until the very end do not throw away
            self._index_in_test = 0
            return False
        self._index_in_test += batch_size
        end = self._index_in_test
        self.inputs_test_batch = {}
        x = self.inputs_test['words']
        x_batch = x[start:end]
        max_len = np.max(np.sum(x_batch!=0, axis=1))
        for key, x in self.inputs_test.items():
            x_batch = x[start:end]
            if key == 'chars':
                max_word_len = np.max(np.sum(x_batch!=0, axis=2))
                self.inputs_test_batch[key] = x_batch[:, :max_len, :max_word_len]
            else:
                self.inputs_test_batch[key] = x_batch[:, :max_len]
        return True


    def output_stags(self, predictions, filename):
        stags = map(lambda x: self.idx_to_tag[x], predictions)
        ## For formatting, let's calculate sentence lengths. np.sum is also faster than a for loop
        ## To Do: allow for the CoNLL format
        sents_lengths = np.sum(self.inputs_test['words']!=0, 1)
        stag_idx = 0
        with open(filename, 'wt') as fwrite:
            for sent_idx in xrange(len(sents_lengths)):
                fwrite.write(' '.join(stags[stag_idx:stag_idx+sents_lengths[sent_idx]]))
                fwrite.write('\n')
                stag_idx += sents_lengths[sent_idx]

    def output_probs(self, probs):
        output_mica_nbest(probs, self.idx_to_tag)
    def output_weight(self, stag_embeddings):
        filename = 'stag_embeddings.txt'
        with open(filename, 'wt') as fout:
            for i in xrange(stag_embeddings.shape[0]):
                output_row = [self.idx_to_tag[i]]+map(str, stag_embeddings[i])
                fout.write(' '.join(output_row))
                fout.write('\n')

def invert_dict(index_dict): 
    return {j:i for i,j in index_dict.items()}


if __name__ == '__main__':
    class Opts(object):
        def __init__(self):
            self.jackknife = 1
            self.embedding_dim = 100
            #data_dir = '../project/tag_wsj'
            data_dir = 'sample_data'
            self.text_train = data_dir + '/sents/train.txt'
            self.tag_train = data_dir + '/predicted_stag/train.txt'
            self.jk_train = data_dir + '/predicted_stag/train.txt'
            self.text_test = data_dir + '/sents/dev.txt'
            self.tag_test = data_dir + '/predicted_stag/dev.txt'
            self.jk_test = data_dir + '/predicted_stag/dev.txt'
            self.word_embeddings_file = 'glovevector/glove.6B.100d.txt'
    opts = Opts()
    data_loader = Dataset(opts)
    data_loader.next_batch(2)
    data_loader.next_test_batch(2)
    print(data_loader.inputs_test_batch['chars'][0])
    print(data_loader.idx_to_char)
#    data_loader.next_test_batch(3)
#    print(data_loader.inputs_test_batch[0])
#
