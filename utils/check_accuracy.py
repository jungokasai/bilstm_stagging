stags = []
with open('beta0.005_stagoutput_23') as fhand:
    for line in fhand:
        if not 'EOS' in line:
            stags.append(line.split()[0].split('_')[0][1:])
#gold_stags = []
idx = 0
with open('data/super_data/test_y.txt') as fhand:
    with open('predicted_stags_test.txt', 'wt') as fwrite:
        for line in fhand:
            gold = line.split()
            stags_sent = stags[idx:idx+len(gold)]
            fwrite.write(' '.join(stags_sent))
            fwrite.write('\n')
            idx += len(gold)
            #gold_stags.extend(gold)
#count = 0
#total = 0
#for stag, gold_stag in zip(stags, gold_stags):
#    if stag == gold_stag:
#        count+=1
#    total+=1.0
#print(count/total)
