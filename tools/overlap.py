from nbest import N_Best
from get_features import Features
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

features= Features()
#print(features.similarity('t1', 't2'))


n_best = N_Best()
tag_to_lab = n_best.tag_to_lab()
tag_list = ['t{}'.format(i) for i in xrange(4725)]
tags = tag_to_lab.keys()
#for tag in tag_list:
#    if tag in tags:
#        continue
#    else:
#        print(tag)
#    
lab_to_tag = {j:i for i,j in tag_to_lab.items()}
n_best_list = n_best.n_best_labs(2)
betas = [1, 0.075, 0.03, 0.01, 0.005, 0.001, 0.0001]
for beta in betas:
    overalls = 0
    num_labs = 0
    num_picked = list(n_best.beam_beta(beta))
    for i, j in zip(xrange(n_best_list.shape[0]), num_picked):
        predicted_labs = list(n_best_list[i])
        tags = [lab_to_tag[lab] for lab in predicted_labs]
        if 'tco' in tags:
            continue
        num_labs+=1
        if j == 1:
            tags[1]=tags[0]
        ## similarity 
        overall, feature_wise = features.similarity(tags[0], tags[1])
        overalls += overall
        if num_labs == 1:
            feature_wise_test = feature_wise
        else:
            for name in feature_wise_test.keys():
                feature_wise_test[name]+= feature_wise[name]
                #print(feature_wise_test[name])
        #feature_wise_test 
    print(overalls/float(num_labs))
    print(num_labs)
    for name in feature_wise_test.keys():
        feature_wise_test[name]=float(feature_wise_test[name])/float(num_labs)

    print(feature_wise_test)
    x= feature_wise_test.keys()
    y= feature_wise_test.values()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    indecies = np.arange(len(x))
    ax.bar(indecies, y, width=1.0, color='r')
    ax.set_xticks(indecies+0.5)
    ax.set_xticklabels(x, rotation=90)
    ax.set_title('Feature wise Ambiguity')
    #ax.grid(True)
    #ax.set_xlabel('n best')
    #ax.set_ylabel('accuracy')
    fig.tight_layout()
    fig.savefig('test{}.png'.format(beta))

    
