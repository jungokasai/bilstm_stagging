import numpy as np

def back_track(predictions, scores, back_pointers, weight):
    ## predictions, scores [batch_size, seq_len]
    ## back_points [batch_size, seq_len-1]
    predictions = predictions.T
    scores = scores.T
    back_pointers = back_pointers.T ## [seq_len-1, batch_size]
    weight = weight.T
    weight = np.invert(weight[1:]) ## [seq_len-1, batch_size]
    seq_len, batch_size = predictions.shape
#    print(back_pointers[:,0:2])
    identity = np.tile(np.arange(batch_size), [seq_len-1, 1])
    back_pointers[weight] = identity[weight] 
    back_pointers = np.vstack([back_pointers, np.arange(batch_size)]) ## [seq_len, batch_size]
    global_back_pointers = back_pointers + np.arange(0, batch_size*seq_len, batch_size).reshape([-1,1]) ## [seq_len, batch_size]
    #print(global_back_pointers[:,0:2])
#    print(back_pointers[:,0:2])
    #print(predictions[:,0:2])
    #print(scores[:,0:2])
    predictions = predictions.reshape(-1)[global_back_pointers]
    scores = scores.reshape(-1)[global_back_pointers]
    predictions = predictions.T
    scores = scores.T
    back_pointers = back_pointers.T
    return predictions, scores, back_pointers
