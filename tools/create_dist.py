from nbest import N_Best
from get_features import Features
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

features= Features()
#print(features.similarity('t1', 't2'))


n_best = N_Best()
tag_to_lab = n_best.tag_to_lab()
tag_list = ['t{}'.format(i) for i in xrange(4725)]
tags = tag_to_lab.keys()
nbest_labs=n_best.n_best_labs(10)
lab_to_tag = {j:i for i,j in tag_to_lab.items()}
#picked = n_best.beta_beam_picked(0.005)
picked = n_best.beta_beam_picked(0.03)

#with open('10_best_tags_23.txt','wt') as fhand:
with open('10_best_tags_pete.txt','wt') as fhand:
#    with open('10_best_probs_23.txt','wt') as fhand_probs:
    with open('10_best_probs_pete.txt','wt') as fhand_probs:
        writer=csv.writer(fhand, delimiter=' ')
        writer_probs=csv.writer(fhand_probs, delimiter=' ')
        for i in xrange(nbest_labs.shape[0]):
            ordering = n_best.dists_nbest[i].argsort()
            labs = list(ordering[-10:])
            labs.reverse()
            tags = map(lambda x: lab_to_tag[x], labs)  
            probs = list(n_best.dists_nbest[i][ordering])
            probs.reverse()
            
            writer_probs.writerow(probs[:10])
            writer.writerow(tags)
    
##with open('beta0.005_tags_23.txt','wt') as fhand:
#with open('beta0.03_tags.txt','wt') as fhand:
#    #with open('beta0.005_probs_23.txt','wt') as fhand_probs:
#    with open('beta0.03_probs.txt','wt') as fhand_probs:
#        writer=csv.writer(fhand, delimiter=' ')
#        writer_probs=csv.writer(fhand_probs, delimiter=' ')
#        for i in xrange(nbest_labs.shape[0]):
#            num_picked = np.sum(picked[i])
#
#            ordering = n_best.dists_nbest[i].argsort()
#            labs = list(ordering[-num_picked:])
#            labs.reverse()
#            tags = map(lambda x: lab_to_tag[x], labs)  
#            probs = list(n_best.dists_nbest[i][ordering])
#            probs.reverse()
#            probs = probs[:len(tags)]
#            writer_probs.writerow(probs)
#            writer.writerow(tags)
#for tag in tag_list:
#    if tag in tags:
#        continue
#    else:
#        print(tag)
#    
