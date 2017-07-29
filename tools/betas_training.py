import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

#data_path = '../Super_models/Super_tagging_cap1_num1_bi1_numlayers2_embeddim100_embedtypeglovevector_seed0_units512_dropout0.5_inputdp0.8_embeddingtrain1_suffix1_windowsize0_jackknife1_jkdim5_hiddendp0.7_sync_zero'

with open('tag_dict.pkl') as fdict:
    tag_dict = pickle.load(fdict)
    #with open('idx_to_stag.pkl') as fdict:
    #    idx_to_stag_prime = pickle.load(fdict)
    idx_to_stag = {u:v for v, u in tag_dict.items()}

beta = 0.005

max_picked = 0

with open('../predicted_dist_training.txt') as fhand:
    for line in fhand:
        dist = np.array(map(float, line.split()))
        best = np.max(dist)
        picked = (dist>= best*beta).astype(int)
        num_picked = np.sum(picked)
        if max_picked<=num_picked:
            max_picked = num_picked
        new = dist*picked
        picked_idx = new.argsort()[-num_picked:][::-1]
        #print(map(lambda x: idx_to_stag[x], picked_idx))
print(max_picked)

#def beam_best(beta):
#    best = np.reshape(np.max(dists_nbest, -1), [-1, 1]) #(num_word, )
#    picked_dists = (dists_nbest>= best*beta).astype(int)
#    num_picked = np.mean(np.sum(picked_dists, -1))
#    return np.mean(np.max(picked_dists*true_dists, -1)), num_picked
#
