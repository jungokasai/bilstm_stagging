stag_counts = {}
#with open('1best_stags.txt') as fhand:
#with open('../NLG/pos.txt') as fhand:
with open('data/predicted_pos/dev.txt') as fhand:
    for line in fhand:
        stags = line.split()
        for stag in stags:
            stag_counts[stag] = stag_counts.get(stag, 0) + 1
print(len(stag_counts))
