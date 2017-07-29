def get_stags(filename):
    nb_stags = []
    stags = []
    with open(filename) as fhand: 
        for line in fhand: 
            stags_sent = line.split()
            nb_stags.append(len(stags))
            stags.extend(stags_sent)
    return stags, nb_stags
def get_accuracy(predicted_stags, gold_stags):
    count = 0
    correct = 0
    nb_stags = len(predicted_stags)
    print(nb_stags)
    for stag_idx in xrange(nb_stags):
        count += 1
        predicted_stag = predicted_stags[stag_idx]
        gold_stag = gold_stags[stag_idx]
        if predicted_stag == gold_stag:
            correct += 1
    return float(correct)/count
    
def one_best_accuracy(predicted_filename, gold_filename):
    predicted_stags, predicted_nb_stags = get_stags(predicted_filename)
    gold_stags, gold_nb_stags = get_stags(gold_filename)
    accuracy = get_accuracy(predicted_stags, gold_stags)
    nb_sents = len(predicted_nb_stags)
    for i in xrange(nb_sents):
        print(i)
        if not predicted_nb_stags[i] == gold_nb_stags[i]:
            print('Wrong')
            print(predicted_nb_stags[i])
            print(gold_nb_stags[i])
            print(i)

    return accuracy

if __name__ == '__main__':
#    accuracy = one_best_accuracy('data/predicted_pos/dev_parsey.txt', 'data/pos_data/dev.txt')
    print('jungomica')
    accuracy = one_best_accuracy('/home/lily/jk964/Dropbox/jungomica/dev.txt', 'data/super_data/dev.txt')
    print(accuracy)

