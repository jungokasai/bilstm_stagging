filename = 'data/sents/toBeSupertagged.txt'
left = '-LRB-'
right = '-RRB-'
output_file = 'data/sents/Supertagged_ready.txt'
with open(filename) as fhand:
    with open(output_file, 'wt') as fwrite:
        for line in fhand:
            tokens = line.split()
            new_tokens = []
            for token in tokens:
                if token == '(':
                    new_tokens.append(left)
                elif token == ')':
                    new_tokens.append(right)
                else:
                    new_tokens.append(token)
            fwrite.write(' '.join(new_tokens))
            fwrite.write('\n')


