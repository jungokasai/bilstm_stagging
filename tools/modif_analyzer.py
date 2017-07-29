import pandas as pd
import numpy as np


def collapse_classes(confusion, collapse_dict, classes):
    collapse_names = []
    clusters = []
    idx = 0
    for names in collapse_dict.values():
        collapse_names.extend(names)
    print(collapse_names)
    indices = [classes.index(sub_class) for sub_class in collapse_names]
    collapsed = confusion[indices]
    collapsed = collapsed[:,indices] 
    return collapsed 

def read_csv(file_name):
    confusion = pd.read_csv(file_name)
    classes = list(confusion)[1:]
    confusion = confusion.as_matrix()[:,1:]
    confusion = confusion.astype(int)
#    confusion.rename(columns={'V': 'xx', 'VP': 'xx'}, inplace=True)
#    #print(confusion.rename(index={20: 'N'}))
#    print(confusion)
#    confusion.groupby('xx').sum()
#    print(confusion)
    return confusion, classes

def output_confusion(confusion):
    return 0


if __name__ == '__main__':
    collapse_dict = {}
    #collapse_dict['NOUN'] = ['N', 'NA', 'NP', 'NPP']
    collapse_dict['NOUN'] = ['N', 'NP', 'NPP']
    collapse_dict['VERB'] = ['V', 'VP']
    #collapse_dict['ADJ'] = ['A', 'AP']
    #collapse_dict['ADV'] = ['Ad', 'AdvP']
    file_name = 'stag_results/bilstm/confusion_modif.csv'
    confusion, classes = read_csv(file_name)
    compact_confusion = collapse_classes(confusion, collapse_dict, classes)
    print(compact_confusion)
    output_confusion(compact_confusion)
    #file_name = 'stag_results/bilstm/confusion_modif.csv'
    file_name = 'stag_results/jungomica/confusion_modif.csv'
    confusion, classes = read_csv(file_name)
    compact_confusion = collapse_classes(confusion, collapse_dict, classes)
    print(compact_confusion)
    output_confusion(compact_confusion)
    

