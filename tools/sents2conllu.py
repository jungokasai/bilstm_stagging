def read_sents(filename):
    output = []
    with open(filename) as fhand:
        for line in fhand:
            output.append(line.split())
    return output

def output_conllu(filename, words, pos):
    assert(len(words) == len(pos)) 
    with open(filename, 'wt') as fhand:
        for sent_idx in xrange(len(words)):
            words_sent = words[sent_idx]
            pos_sent = pos[sent_idx]
            assert(len(words_sent) == len(pos_sent))
            for word_idx in xrange(len(words_sent)):
                word = words_sent[word_idx]
                pos_tag = pos_sent[word_idx]
                output_list = [str(word_idx+1), word, '_', pos_tag, pos_tag, '_', '_', '_', '_', '_']
                fhand.write('\t'.join(output_list))
                fhand.write('\n')
            fhand.write('\n') ## blank line

if __name__ == '__main__':
    #sents_file = 'data/sents/dev.txt'
    sents_file = 'data/sents/ppattach_sents.txt'
    #predicted_pos_file = 'data/predicted_pos/dev.txt'
#    predicted_pos_file = 'data/predicted_pos/dev.txt'
    output_file = 'dev.conllu'
    words = read_sents(sents_file)
    pos = words
#    pos = read_sents(predicted_pos_file)
#    pos = read_sents(predicted_pos_file)
    output_conllu(output_file, words, pos)
