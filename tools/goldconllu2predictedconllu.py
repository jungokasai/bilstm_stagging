def read_sents(filename):
    output = []
    with open(filename) as fhand:
        for line in fhand:
            output.extend(line.split())
    return output
def output_conllu(filename, original_conll, pos):
    word_idx = 0
    with open(original_conll) as fhand:
        with open(filename, 'wt') as fwrite:
            for line in fhand:
                tokens = line.split()
                if len(tokens)>1: # not blank
                    tokens[3] = pos[word_idx]
                    tokens[4] = pos[word_idx]
                    fwrite.write('\t'.join(tokens))
                    fwrite.write('\n')
                    word_idx += 1
                else:
                    fwrite.write('\n')


if __name__ == '__main__':
#    sents_file = 'data/sents/dev.txt'
    print('dev')
    predicted_pos_file = 'data/predicted_pos/dev.txt' ## jackknifing
    output_file = 'dev.conllu'
    gold_conll_file = 'data/conllu/dev.txt'
    pos = read_sents(predicted_pos_file)
    output_conllu(output_file, gold_conll_file, pos)
    print('train')
    predicted_pos_file = 'data/predicted_pos/train.txt' ## jackknifing
    output_file = 'train.conllu'
    gold_conll_file = 'data/conllu/train.txt'
    pos = read_sents(predicted_pos_file)
    output_conllu(output_file, gold_conll_file, pos)
    print('test')
    predicted_pos_file = 'data/predicted_pos/test.txt' ## jackknifing
    output_file = 'test.conllu'
    gold_conll_file = 'data/conllu/test.txt'
    pos = read_sents(predicted_pos_file)
    output_conllu(output_file, gold_conll_file, pos)
