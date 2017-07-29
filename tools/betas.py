import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

#data_path = '../Super_models/Super_tagging_cap1_num1_bi1_numlayers2_embeddim100_embedtypeglovevector_seed0_units512_dropout0.5_inputdp0.8_embeddingtrain1_suffix1_windowsize0_jackknife1_jkdim5_hiddendp0.7_sync_zero'
data_path = '../Super_models/Super_tagging_cap1_num1_bi1_numlayers2_embeddim100_embedtypeglovevector_seed0_units512_dropout0.5_inputdp0.8_embeddingtrain1_suffix1_windowsize0_jackknife1_jkdim5_hiddendp0.5_sync_zero'
output_dir = 'images'

#with open(os.path.join(data_path, 'true_labs_test.pkl'), 'rb') as fhand:
with open(os.path.join(data_path, 'true_labs.pkl'), 'rb') as fhand:
    true_labs = pickle.load(fhand)
#with open(os.path.join(data_path, 'dists_nbest_test.pkl'), 'rb') as fhand:
with open(os.path.join(data_path, 'dists_nbest.pkl'), 'rb') as fhand:
    dists_nbest = np.array(pickle.load(fhand))
true_dists = np.zeros(dists_nbest.shape)
for i, j in enumerate(list(true_labs)):
    true_dists[i, int(j)] = 1
#orders = dists_nbest.argsort()
#print(orders)
#def n_best(n):
#    return (np.mean(np.max((orders >= num_labs-n).astype(int)*true_dists, -1)))
#def n_best(n):
#    n_best_idxs = orders[:,-n:]
#    num_cor = 0
#    for i in xrange(n_best_idxs.shape[0]):
#        correct = int(list(true_labs)[i] in list(n_best_idxs[i]))
#        num_cor +=correct 
#    return float(num_cor)/float(n_best_idxs.shape[0])
#for i in xrange(10):
#    print(n_best(i+1))

def beam_best(beta):
    best = np.reshape(np.max(dists_nbest, -1), [-1, 1]) #(num_word, )
    picked_dists = (dists_nbest>= best*beta).astype(int)
    num_picked = np.mean(np.sum(picked_dists, -1))
    return np.mean(np.max(picked_dists*true_dists, -1)), num_picked
scores = []
amb_levels = []
for beta in [1, 0.075, 0.03, 0.01, 0.005, 0.001, 0.0001]:
    score, amb_level = beam_best(beta)
    print(score)
    scores.append(score)
    amb_levels.append(amb_level)
print(scores)
print(amb_levels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(amb_levels, scores, 'o-')
ax.set_title('Accuracy on Section 23')
ax.grid(True)
ax.set_xlabel('ambiguity level')
ax.set_ylabel('accuracy')
fig.savefig(os.path.join(output_dir, 'scores_betas.png'), dpi=500)


    







