from onebest_accuracy import get_stags
def get_stags_mica(mica_file):
    stags = []
    with open(mica_file) as fhand:
        for line in fhand:
            tokens = line.split()
            if len(tokens)>1:
                stags.append(tokens[2])
    return stags
def get_accuracy(gold_stags, predicted_stags):
    nb_stags = len(gold_stags)
    assert nb_stags == len(predicted_stags)
    count = 0
    for i in xrange(nb_stags):
        gold = gold_stags[i]
        predicted = predicted_stags[i]
        if gold == predicted:
            count += 1
    return float(count)/nb_stags

if __name__ == '__main__':
    gold_file = 'data/super_data/dev.txt'
    mica_file = 'data/mica/d6.clean2.00.jungomicastagged'
    #mica_file = 'data/mica/d6.clean2.00.micastagged'
    gold_stags, nb_stags = get_stags(gold_file)
    predicted_stags = get_stags_mica(mica_file)
    accuracy = get_accuracy(gold_stags, predicted_stags)
    print(accuracy)
    
