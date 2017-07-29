#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import os

def plot_heatmap(contexts, sentence, file_name, opts, exp = False):
    heat_dir = os.path.dirname(file_name)
    if not os.path.isdir(heat_dir):
        os.makedirs(heat_dir)
    data = np.zeros((len(sentence), len(sentence)))
    # now assign the windows 
    k = opts.atwindow_size
    if opts.recurrent_attention == 0:
        for j, C in enumerate(contexts):
            if j<=(k-1):
                C = C.reshape(-1)[-(k+j+1):] # get rid of attention pad if any
                data[j,:k+j+1] = C[:data[j,:k+j+1].shape[0]] # get rid of sentence pad if any
            else:
                data[j,j-k:j+k+1] = C.reshape(-1)[:data[j,j-k:j+k+1].shape[0]]# get rid of sentence pad if any
    if opts.recurrent_attention == 1: #or opts.attention == 30:
        for j, C in enumerate(contexts):
            #print(np.sum(C, axis = -1))
            data[j,:] = C.reshape(-1)[:len(sentence)] # take the non padding
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #data = np.log(data)
    print('data')
    print(data[-5,:])
    data/np.sum(data, axis = -1, keepdims = True)
    if exp:
        print('taking exp')
        data = np.exp(data)
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(sentence, rotation =90, minor=False, fontsize = 5)
    ax.set_yticklabels(sentence, minor=False, fontsize = 5)
    fig.tight_layout()
    fig.savefig(file_name, dpi = 500)
