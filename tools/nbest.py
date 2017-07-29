import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle


class N_Best(object):
    def __init__(self):
        self.data_path = '../Super_models/Super_tagging_cap1_num1_bi1_numlayers2_embeddim100_embedtypeglovevector_seed0_units512_dropout0.5_inputdp0.8_embeddingtrain1_suffix1_windowsize0_jackknife1_jkdim5_hiddendp0.5_sync_zero'
        self.output_dir = 'images'

        #with open(os.path.join(self.data_path, 'true_labs_test.pkl'), 'rb') as fhand:
        with open(os.path.join(self.data_path, 'true_labs.pkl'), 'rb') as fhand:
            self.true_labs = pickle.load(fhand)
        #with open(os.path.join(self.data_path, 'dists_nbest_test.pkl'), 'rb') as fhand:
        with open(os.path.join(self.data_path, 'dists_nbest.pkl'), 'rb') as fhand:
            self.dists_nbest = pickle.load(fhand)
        self.true_dists = np.zeros(self.dists_nbest.shape)
        for i, j in enumerate(list(self.true_labs)):
            self.true_dists[i, int(j)] = 1
        self.num_labs = self.dists_nbest.shape[1]
        self.orders = self.dists_nbest.argsort(axis=-1).argsort(axis=-1) #ranking
    def beam_beta(self, beta):
	best = np.reshape(np.max(self.dists_nbest, -1), [-1, 1]) #(num_word, )
	picked_dists = (self.dists_nbest>= best*beta).astype(int)
	num_picked = np.minimum(np.sum(picked_dists, -1), 2)
	return num_picked

    def beta_beam_picked(self, beta):
	best = np.reshape(np.max(self.dists_nbest, -1), [-1, 1]) #(num_word, )
	picked_dists = (self.dists_nbest>= best*beta).astype(int)
	return picked_dists

    def nbest_dist(self, n):
        return self.dists_nbest[self.dists_nbest.argsort(axis=-1)].reshape([-1, n])

    def n_best(self, n):
        return (np.mean(np.max((self.orders >= self.num_labs-n).astype(int)*self.true_dists, -1)))

    def n_best_labs(self, n):
        num_sents = self.true_dists.shape[0]
        index_array = np.zeros([num_sents, self.num_labs], dtype='int')
        for i in xrange(num_sents):
            index_array[i] = np.arange(self.num_labs)
        return index_array[self.orders >= self.num_labs-n].reshape(-1, n)

    def tag_to_lab(self):
        with open('tag_dict.pkl', 'rb') as fhand:
            tag_dict = pickle.load(fhand)
        return tag_dict
        
    def plot(self):
        ns = []
        scores = []

        for i in xrange(10):
            scores.append(self.n_best(i+1))
            ns.append(i+1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ns, scores, 'o-')
        ax.set_title('Accuracy on Section 23')
        ax.grid(True)
        ax.set_xlabel('n best')
        ax.set_ylabel('accuracy')
        fig.savefig(os.path.join(self.output_dir, 'scores_nbest.png'), dpi=500)
if __name__ == '__main__':

    nbest = N_Best()
    nbest.plot()
