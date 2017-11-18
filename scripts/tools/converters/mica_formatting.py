import sys

def micafy(stags_file, probs_file, sents_file):
    f_stag = open(stags_file)
    f_prob = open(probs_file)
    f_txt = open(sents_file)
    out_file = '10_best_stagoutput'
    f_out = open(out_file, 'wt')
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
    f_prob.close()
    f_stag.close()
if __name__ == '__main__':
    stags_file = sys.argv[1]
    probs_file = sys.argv[2]
    sents_file = sys.argv[3]
    micafy(stags_file, probs_file, sents_file)

