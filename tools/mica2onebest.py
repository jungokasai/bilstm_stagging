
def get_stags(file_name):
    stags = []
    stags_sent = []
    with open(file_name) as fhand:
        for line in fhand:
            tokens = line.split()
            if len(tokens) > 5:
                stag = tokens[6]
                stags_sent.append(stag)
            else:
                if len(stags_sent) > 0:
                    stags.append(stags_sent)
                    stags_sent = []
    return stags

def output_sents(stags, out_file='test'):
    with open(out_file, 'wt') as fhand:
        for stags_sent in stags:
            fhand.write(' '.join(stags_sent))
            fhand.write('\n')

if __name__ == '__main__':
    file_name = 'data/mica/maxent_mica.txt'
    stags = get_stags(file_name)
    output_sents(stags)
    print(len(stags))
    
