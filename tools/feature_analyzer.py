from collections import defaultdict

def get_properties():
    feature_dir = 'data/tree_prop/d6.treeproperties'
    props = defaultdict(dict)
    with open(feature_dir, 'rt') as fhand: 
        for line in fhand:
            feature_list = line.split()
            tag = feature_list[0]
            feature_list = feature_list[1:]
            for feature in feature_list:
                parts = feature.split(':')
                name = parts[0]
                ans = parts[1]
                props[tag][name] = ans 
    return props 
def get_nary_list():
    nary_list = ['root', 'dir', 'modif', 'substnodes', 'predaux', 'coanc', 'particle', 'particleShift', 'comp', 'pred', 'datshift', 'esubj', 'rel', 'wh', 'voice']
    return nary_list

def read_stag(stag_file):
    stags = [] 
    with open(stag_file) as fhand:
        for line in fhand:
           stags.extend(line.split())
    return stags

def featwise_sim(feat, feat_prime):
    sim = {}
    for feat_name in feat.keys():
        if feat[feat_name] == feat_prime[feat_name]:
            sim[feat_name] = 1
        else:
            sim[feat_name] = 0
    return sim
            

def stag_similarity(stag, stag_prime, props):
    stag_feat = props[stag]
    stag_prime_feat = props[stag_prime]
    sim = featwise_sim(stag_feat, stag_prime_feat)
    if len(sim) == 0:
        print(stag, stag_prime)
    return sim

def stag_feat(stag, stag_prime, props):
    stag_feats = props[stag]
    stag_prime_feats = props[stag_prime]
    return stag_feats, stag_prime_feats
    
def raw_similarity(gold_stags, predicted_stags, nary_list, props):
    assert(len(gold_stags) == len(predicted_stags))
    nb_tokens = len(gold_stags)
    overall_similarity = {feat: 0 for feat in nary_list}
    count = 0
    for stag_idx in xrange(len(gold_stags)):
        gold_stag = gold_stags[stag_idx] 
        predicted_stag = predicted_stags[stag_idx] 
        with_feat = props.keys()
        if (gold_stag in with_feat) and (predicted_stag in with_feat):
            sim = stag_similarity(gold_stag, predicted_stag, props)
            count += 1
            for feat_name in overall_similarity.keys():
                    overall_similarity[feat_name] += sim[feat_name]
    for u, v in overall_similarity.items():
        overall_similarity[u] = v/float(count)
    return overall_similarity

def compare(gold_stags, predicted_stags, nary_list, props):
    assert(len(gold_stags) == len(predicted_stags))
    nb_tokens = len(gold_stags)
    overall_similarity = {feat: defaultdict(int) for feat in nary_list} # feat: {(gold, predicted): #}
    count = 0
    for stag_idx in xrange(len(gold_stags)):
        gold_stag = gold_stags[stag_idx] 
        predicted_stag = predicted_stags[stag_idx] 
        with_feat = props.keys()
        if (gold_stag in with_feat) and (predicted_stag in with_feat):
            gold_feats, predicted_feats = stag_feat(gold_stag, predicted_stag, props)
            for feat_name in nary_list:
                overall_similarity[feat_name][(gold_feats[feat_name], predicted_feats[feat_name])] += 1
    return overall_similarity 
def output_confusion(comparison, output_file, title):
    import csv
    with open(output_file, 'at') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',)
#    print(output_file)
#    print(comparison)
#        values = sorted(list(set([x[0] for x in comparison list[values]))))
        print('create a list of possible classes for the particular feature')
        values = sorted(set([x[0] for x in comparison] + [x[1] for x in comparison]))
    #    print(values)
        writer.writerow([title] + values) ## first row
        for predicted_value in values:
            output_row = [predicted_value]
            for gold_value in values:
                output_row.append(comparison[gold_value, predicted_value])
            writer.writerow(output_row)

if __name__ == '__main__':
    props = get_properties()
    nary_list = get_nary_list()
    for feat in nary_list:
        print(props['t27'][feat])
#    predicted_stagfile = 'data/predicted_stag/dev.txt'
#    predicted_stagfile = 'data/mica/maxent_dev.txt'
#    predicted_stagfile = 'data/mica/bilstm_dev.txt'
    predicted_stagfile = 'data/mica/jungomica_dev.txt'
#    gold_stagfile = 'data/super_data/dev.txt'
    gold_stagfile = 'data/mica/gold_dev.txt'
    predicted_stags = read_stag(predicted_stagfile)
    gold_stags = read_stag(gold_stagfile)
#    overall_similarity = raw_similarity(gold_stags, predicted_stags, nary_list, props)
    overall_similarity = compare(gold_stags, predicted_stags, nary_list, props)
    for feat in nary_list:
        classes = overall_similarity[feat].keys()
        gold_classes = list(set([x[0] for x in classes]))
        predicted_classes = list(set([x[1] for x in classes]))
#        all_classes = list(set(gold_classes + predicted_classes))
#        print(len(gold_classes), len(predicted_classes), len(all_classes), feat)
#        if feat == 'datshift':
#            print(gold_classes)
        print(len(overall_similarity[feat]), feat)
        output_file = 'confusion_{}.csv'.format(feat)
        output_confusion(overall_similarity[feat], output_file, 'bilstm')

#    predicted_stagfile = '../tag_parsing/data/jungomica/dev.txt'
#    predicted_stags = read_stag(predicted_stagfile)
#    overall_similarity = compare(gold_stags, predicted_stags, nary_list, props)
#    for feat in nary_list:
#        output_file = 'confusion_{}.csv'.format(feat)
#        output_confusion(overall_similarity[feat], output_file, 'bilstm+mica')
#
