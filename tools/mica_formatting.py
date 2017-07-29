#f_prob = open('beta0.005_probs.txt')
#f_stag = open('beta0.005_tags.txt')
#f_txt = open('data/super_data/dev_x_longer.txt')
#f_out = open('beta0.005_stagoutput', 'wt')
#f_prob = open('beta0.005_probs_23.txt')
#f_stag = open('beta0.005_tags_23.txt')
#f_txt = open('data/super_data/test_x.txt')
#f_out = open('beta0.005_stagoutput_23', 'wt')
#f_prob = open('10_best_probs_23.txt')
f_prob = open('10_best_probs_pete.txt')
#f_stag = open('10_best_tags_23.txt')
f_stag = open('10_best_tags_pete.txt')
#f_txt = open('data/super_data/test_x.txt')
f_txt = open('data/super_data/all_te.txt')
#f_out = open('10_best_stagoutput_23', 'wt')
f_out = open('10_best_stagoutput_pete', 'wt')

lens = []
for words in f_txt:
    lens.append(len(words.split()))
print(lens)

i = 0
nb_words = 0
for prob_line, stag_line in zip(f_prob, f_stag):
    probs = prob_line.split()
    stags = stag_line.split()
    outputline = []
    for prob, stag in zip(probs, stags):
        outputline.append(stag+'_1:'+prob)
    f_out.write('/')
    f_out.write(' '.join(outputline))
    f_out.write('\n')
        
    nb_words += 1
    if nb_words == lens[i]:
        f_out.write('/...EOS...\n')
        i+=1
        nb_words = 0

    #for word in sent:
    #    print(word)


f_prob.close()
f_stag.close()
#f_txt.close()

