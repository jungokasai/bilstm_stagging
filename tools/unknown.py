import pickle

with open('word_dict.pkl') as fhand:
    word_to_idx = pickle.load(fhand)
data_names = ['te_dev_hyps.txt', 'te_test_hyps.txt', 'te_dev_texts.txt', 'te_test_texts.txt']

tokens = []
for data_name in data_names:
    with open('data/super_data/{}'.format(data_name)) as fhand:
        for line in fhand:
            tokens.extend(line.lower().split())
    
for token in tokens:
    if not token in word_to_idx.keys():
        print(token)
