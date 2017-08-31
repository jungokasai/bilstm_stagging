def read_sents(file_name):
    stags = []
    unique_stags = set()
    with open(file_name) as fhand:
        for line in fhand:
            stags_sent = line.split()
            stags.append(stags_sent)
            unique_stags = unique_stags.union(set(stags_sent))
    print(len(unique_stags))
    return stags
def sentwise_acc(gold_stags, predicted_stags):
    count = 0
    nb_correct = 0
    assert(len(gold_stags) == len(predicted_stags))
    for sent_idx in xrange(len(gold_stags)):
        count += 1
        gold_stags_sent = gold_stags[sent_idx]
        predicted_stags_sent = predicted_stags[sent_idx]
        if ' '.join(gold_stags_sent) == ' '.join(predicted_stags_sent):
            nb_correct += 1
    return float(nb_correct)/count*100


if __name__ == '__main__':
    gold = '../project/tag_wsj/gold_stag/dev.txt'
    gold_stags = read_sents(gold)
    predicted = '../project/tag_wsj/predicted_stag/dev.txt'
    predicted_stags = read_sents(predicted)
    acc = sentwise_acc(gold_stags, predicted_stags)
    print(acc)
    predicted = '../tag_parsing/data/predicted/dev.txt'
    predicted_stags = read_sents(predicted)
    acc = sentwise_acc(gold_stags, predicted_stags)
    print(acc)
    gold = '../project/tag_wsj/gold_stag/test.txt'
    gold_stags = read_sents(gold)
    predicted = '../project/tag_wsj/predicted_stag/test.txt'
    predicted_stags = read_sents(predicted)
    acc = sentwise_acc(gold_stags, predicted_stags)
    print(acc)
    predicted = '../tag_parsing/data/predicted/test.txt'
    predicted_stags = read_sents(predicted)
    acc = sentwise_acc(gold_stags, predicted_stags)
    print(acc)
