import numpy as np
import os
import csv
import itertools

def invert_dict(index_dict): 
    return {j:i for i,j in index_dict.items()}
def seq_to_word(index_dict, seq_mat, comparison_wrong_mat):
    inv_dict = invert_dict(index_dict) 
    inv_dict[0] = 'Null' # padding
    sents = []
    for i in xrange(seq_mat.shape[0]):
        sent = []
        for num, compare in zip(seq_mat[i], comparison_wrong_mat[i]):
            word = inv_dict[num]
            if not compare: 
                word+='*'
            sent.append(word)
                
        sents.append(sent)
    return sents
    
    

def confusion_mat(X_test, y_test, prediction, padding, word_index, tag_index, model_basename):
    comparison_mat = np.equal(y_test, prediction)
    comparison_mat[padding] = True
    nb_correct_words = np.sum(comparison_mat, axis =-1)
    correct_sents = nb_correct_words==X_test.shape[-1]
    wrong_sents = np.invert(correct_sents)
    comparison_wrong_mat=comparison_mat[wrong_sents]
    wrong_x = X_test[wrong_sents]
    wrong_y = y_test[wrong_sents] # gold tags
    wrong_pred = prediction[wrong_sents]
    sents = seq_to_word(word_index, wrong_x, comparison_wrong_mat)
    gold_tags=seq_to_word(tag_index, wrong_y, comparison_wrong_mat)
    pred_tags=seq_to_word(tag_index, wrong_pred, comparison_wrong_mat)
    mat_name = os.path.join(model_basename, 'confusion_mat.csv')
    with open(mat_name, 'wb') as csvfile:
        mat_writer=csv.writer(csvfile, delimiter=',')
        for sent, gold_tag, pred_tag in zip(sents, gold_tags, pred_tags):

            mat_writer.writerow(['sentence']+sent)
            mat_writer.writerow(['gold']+gold_tag)
            mat_writer.writerow(['predicted']+pred_tag)
    
def confusion_table(y_test, prediction, nonpadding, tag_index, comparison, model_basename):
    preds=prediction[nonpadding]
    targets=y_test[nonpadding]
    comparison = np.invert(comparison)
    wrong_preds=preds[comparison]
    wrong_targets=targets[comparison]
    tags = tag_index.keys()
    combos = list(itertools.product(tags, tags))
    counter={}
    inv_dict = invert_dict(tag_index)
    for comb in combos:
        counter[comb]=0
    for wrong_pred, wrong_target in zip(wrong_preds, wrong_targets):
        counter[(inv_dict[wrong_pred], inv_dict[wrong_target])] += 1
    
    entire_counter={}
    for comb in combos:
        entire_counter[comb]=0
    for pred, target in zip(preds, targets):
        entire_counter[(inv_dict[pred], inv_dict[target])] += 1
    
    table_name = os.path.join(model_basename, 'confusion_table.csv')
    entire_table_name = os.path.join(model_basename, 'entire_confusion_table.csv')
    
    with open(table_name, 'wb') as csvfile:
        mat_writer=csv.writer(csvfile, delimiter=',')
        
        mat_writer.writerow(['pred vs target'] + tags)

        for pred in tags:
            mat_writer.writerow([pred]+ [counter[(pred, target)] for target in tags])
                
    with open(entire_table_name, 'wb') as csvfile:
        mat_writer=csv.writer(csvfile, delimiter=',')
        
        mat_writer.writerow(['pred vs target'] + tags)

        for pred in tags:
            mat_writer.writerow([pred]+ [entire_counter[(pred, target)] for target in tags])
    

    
