import numpy as np
def compute_ratio(numerators, denominators):
    ## avoid zero division
    original_denominators = np.copy(denominators)
    denominators[denominators==0] = 100 ## dummy to avoid zero division
    ratios = numerators/denominators
    micro = np.sum(ratios*original_denominators)/np.sum(original_denominators) ## ignore zero denominator
    macro = np.sum(ratios*(original_denominators!=0))/np.sum(original_denominators!=0)
    return micro, macro

def read_csv(file_name):
    import pandas
    confusion = pandas.read_csv(file_name)
    classes = list(confusion)[1:]
    confusion = confusion.as_matrix()[:,1:]
    confusion = confusion.astype(float)
    correct_counts = np.diagonal(confusion)
    predicted_counts = np.sum(confusion, 1)
    micro_prec, macro_prec = compute_ratio(correct_counts, predicted_counts)
    gold_counts = np.sum(confusion, 0)
    micro_rec, macro_rec = compute_ratio(correct_counts, gold_counts)
    return (micro_prec, macro_prec, micro_rec, macro_rec)

def compute_all():
    feat_list = ['root', 'coanc', 'modif', 'dir', 'predaux', 'pred', 'comp', 'particle', 'particleShift', 'voice', 'wh', 'rel', 'esubj', 'datshift']
    for feat in feat_list:
#        file_name = 'stag_results/maxent/confusion_{}.csv'.format(feat)
        file_name = 'stag_results/jungomica/confusion_{}.csv'.format(feat)
        micro_prec, macro_prec, micro_rec, macro_rec = read_csv(file_name)
        print(feat)
        print('micro')
        print(micro_prec, micro_rec)
        print('macro')
        print(macro_prec, macro_rec)
    
if __name__ == '__main__':
    compute_all()
    
