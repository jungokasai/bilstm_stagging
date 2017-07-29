#with open('data/conllu/en-ud-dev.conll16') as fhand: 
with open('data/conllu/en-ud-train.conll16') as fhand: 
    with open('en-ud-train.conllu', 'wt') as fwrite:
        for line in fhand:
            line = line.split()[:-1]
            if len(line)>0:
                line[7] = line[7].upper()
            fwrite.write('\t'.join(line))
            fwrite.write('\n')

