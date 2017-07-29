filename = 'data/sents/toBeSupertaggedPlus.txt'
output_file = 'data/sents/ppattach_sents.txt'
with open(filename) as fhand:
    with open(output_file, 'wt') as fwrite:
        for line in fhand:
            tokens = line.split()
            fwrite.write(' '.join(tokens[3:]))
            fwrite.write('\n')
            
        
