#with open('data/conllu/en-ud-dev.conll16') as fhand:
import sys

#idx = 1 #word
def separate_conllu(conllu_file):
    with open(conllu_file) as fhand: 
        sents = []
        sent = []
        for line in fhand:
            tokens = line.split()
            if len(tokens) == 0: 
                if len(sent)>0: ## avoid multiple empty lines that sometimes happen
                    sents.append(sent)
                    #if not 'root' in words_sent:
                    #    print('no root')
                    #    print(count)
                    sent = []
                continue
            line = '\t'.join(tokens)
            sent.append(line)
    return sents

    #with open('dev.txt', 'wt') as fhand:
#    with open(output_dir, 'wt') as fhand:
#        for words_sent in words:
#            fhand.write(' '.join(words_sent))
#            fhand.write('\n')
#idx = 4 #pos
#idx = 10 #stag
#idx =  7#relation
#idx = 6 #parent
#    conllu2sents(4, 'dev', 'dev.txt') 
#    conllu2sents(4, 'train_long', 'train.txt') 
