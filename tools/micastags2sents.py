import numpy as np
def get_stagseqs_with_scores(filename):
    stagseqs = []
    with open(filename) as fhand:
        for line in fhand:
            stagseqs_dict = {}
            line = line.rstrip()
            seqs = line.split('|')[:-1] ## last is empty
            for seq in seqs:
                tokens = seq.split()
                stagseq = tokens[:-1]
                score = float(tokens[-1])
                stagseq_string = ' '.join(stagseq)
                test_string = 't2 t3 t21 t2 t186 t187 t21 t45 t27 t1 t3 t13 t1 t36 t3 t38 t48 t26'
                #test = np.power(stagseqs_dict[stagseq_string], 10) + np.power(score, 10)
                #print(test)
                #print(test<1)
                if stagseq_string in stagseqs_dict.keys():
                    pass
                    #stagseqs_dict[stagseq_string] = np.log10(np.power(10, stagseqs_dict[stagseq_string]) + np.power(10, score))
                    #print(np.power(stagseqs_dict[stagseq_string], 10))
                    #stagseqs_dict[stagseq_string] = score
                else: 
                    stagseqs_dict[stagseq_string] = score
                #stagseqs_dict[stagseq_string] += score 
            stagseqs.append(stagseqs_dict)
    return stagseqs
def check_duplicates(stagseqs, lens):
    seq_id = 0
    for i, length in enumerate(lens):
        stagseqs_dict = {}
        stagseqs_sent = stagseqs[seq_id:seq_id+length]
        seq_id += length
        for stagseq in stagseqs_sent:
            stagseq_string = ''.join(stagseq)
            stagseqs_dict[stagseq_string] = stagseqs_dict.get(stagseq_string, 0) + 1
        if length != len(stagseqs_dict) + 1:
            print(i)
            print(stagseqs_dict)

def output_lens(output_name, lens):
    lens = map(str, lens)
    with open(output_name, 'wt') as fhand:
        fhand.write(' '.join(lens))
def output_stagseqs(stagseqs, stag_file, score_file):
    lens = []
    with open(stag_file, 'wt') as fhand:
        with open(score_file, 'wt') as fhand_score:
            for stagseqs_sent in stagseqs:
                stag_score = sorted(stagseqs_sent.items(), key=lambda x: -x[1])
                lens.append(len(stag_score))
                stag_seqs = [x[0] for x in stag_score]
                scores = [str(x[1]) for x in stag_score] 
                for stag_seq in stag_seqs:
                    fhand.write(stag_seq)
                    fhand.write('\n')
                for score in scores:
                    fhand_score.write(score)
                    fhand_score.write('\n')
    return lens

if __name__ == '__main__':
    filename = 'data/mica/sec00.jungostagged.micastagged'
    lens_filename = 'data/mica/lens.txt'
    stag_file = 'data/mica/mica_stagseqs.txt'
    score_file = 'data/mica/mica_stagseqs_scores.txt'
    stagseqs = get_stagseqs_with_scores(filename)
    lens = output_stagseqs(stagseqs, stag_file, score_file)
    test_string = 't2 t3 t21 t2 t186 t187 t21 t45 t27 t1 t3 t13 t1 t36 t3 t38 t48 t26'
#    print(stagseqs[0][test_string])
    output_lens(lens_filename, lens)
    print(sum(lens))
    
#def check_duplicate():
