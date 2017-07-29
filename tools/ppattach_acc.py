def read_ppattach(ppattach_file):
    answers = []
    nouns = []
    preps = []
    with open(ppattach_file) as fhand:
        for line in fhand:
            tokens = line.split()
            answers.append(tokens[0])
            nouns.append(tokens[1])
            preps.append(tokens[2])
    return answers, nouns, preps
            
def read_conllu(predicted_conllu_file):
    rels = []
    rels_sent = {}
    sents = []
    sent = []
    with open(predicted_conllu_file) as fhand:
        for line in fhand:
            tokens = line.split()
            if len(tokens)<5:
                if len(rels_sent)>0:
                    rels.append(rels_sent)
                    sents.append(sent)
                rels_sent = {}
                sent = []
            else:
                rels_sent[int(tokens[0])] = int(tokens[6]) ## child, parent
                sent.append(tokens[1])
    return rels, sents

def compute_acc(answers, nouns, preps, rels, sents):
    sent_idx = 0
    nb_examples = len(answers)
    nb_correct = 0
    print(nb_examples)
    for sent_idx in xrange(len(answers)):
        found_noun = False
        ans = answers[sent_idx]
        noun = nouns[sent_idx]
        prep = preps[sent_idx]
        rels_sent = rels[sent_idx]
        sent = sents[sent_idx]
        for word_idx in xrange(len(sent)):
            word = sent[word_idx]
            if word == noun: ## found the preceding noun
                found_noun = True
            if found_noun:
                if word == prep:
                    prep_idx = word_idx + 1 ## + 1 for starting from 1, + 1 for the next word (found the preceding noun)
                    break
        parent_idx = rels_sent[prep_idx]
        parent_word = sent[parent_idx-1]
#        prep = sent[prep_idx-1]
        if parent_word == noun:
            pred = 'N'
        else:
            pred = 'V'
        if pred == ans:
            nb_correct += 1
        else:
            print(sent_idx, parent_word, prep)
    return nb_correct/float(nb_examples)
            
        
if __name__ == '__main__':
    gold_ppattach_file = 'data/sents/toBeSupertaggedPlus.txt'
    predicted_conllu_file = 'data/conllu/pp_attach_parsey.txt'
    rels, sents = read_conllu(predicted_conllu_file)
    print(len(rels), len(sents))
#    print(rels[0])
    answers, nouns, preps = read_ppattach(gold_ppattach_file)
    acc = compute_acc(answers, nouns, preps, rels, sents)
    print(acc)
    
