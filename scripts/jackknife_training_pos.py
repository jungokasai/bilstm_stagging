import subprocess
import os
import sys
import json 
import tools
from copy import deepcopy
from tools.converters.conllu2sents import conllu2sents
from tools.converters.sents2conllustag import output_conllu
from tools.converters.separate_conllu import separate_conllu


def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict

def clean_dir(jk_base_dir): ## cleaning unnecessary files
    dirs = os.listdir(jk_base_dir)
    print(dirs)

def collect_results(config, jk_base_dir, k_fold):
    collect_command = 'cat'
    for ith in xrange(k_fold):
        collect_command += ' '
        collect_command += os.path.join(jk_base_dir, str(ith),'conllu', 'k_fold-dev.conllu_stag')
    output_file = os.path.join(config['data']['base_dir'], config['data']['split']['train'])
    output_file += '_jkstag'
    collect_command += ' > {}'.format(output_file)
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    print(collect_command)
    subprocess.check_call(collect_command, shell=True)
    

def train_pos_tagger(config):
    base_dir = config['data']['base_dir']
    base_command = 'python bilstm_stagger_main.py train --task POS_models --base_dir {}'.format(base_dir) 
    train_data_info = ' --text_train {} --jk_train {} --tag_train {}'.format(os.path.join(base_dir, 'sents', 'train.txt'), os.path.join(base_dir, 'gold_cpos', 'train.txt'), os.path.join(base_dir, 'gold_cpos', 'train.txt'))
    dev_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', 'dev.txt'), os.path.join(base_dir, 'gold_cpos', 'dev.txt'), os.path.join(base_dir, 'gold_cpos', 'dev.txt'))
    model_config_dict = config['pos_parameters']
    model_config_info = ''
    for option, value in model_config_dict.items():
        model_config_info += ' --{} {}'.format(option, value)
    complete_command = base_command + train_data_info + dev_data_info + model_config_info
    subprocess.check_call(complete_command, shell=True)

def output_sep(file_name, groups):
    with open(file_name, 'wt') as fwrite:
        for group in groups:
            for sent in group:
                for line in sent:
                    fwrite.write(line)
                    fwrite.write('\n')
                fwrite.write('\n')

def create_config(config, config_file, base_dir):
    config = deepcopy(config)
    config['data']['base_dir'] = base_dir 
    new_dict = {}
    new_dict['train'] = 'conllu/k_fold-train.conllu'
    new_dict['dev'] =  'conllu/k_fold-dev.conllu'
    config['data']['split'] = new_dict
    with open(config_file, 'w') as fhand:
        json.dump(config, fhand)


def separate_training_set(config, k_fold):
    base_dir = config['data']['base_dir']
    train_conllu = os.path.join(base_dir, config['data']['split']['train'])
    sents = separate_conllu(train_conllu)
    jackknife_dir = os.path.join(base_dir, '{}_fold'.format(k_fold))
    size = len(sents)//k_fold
    if not os.path.isdir(jackknife_dir):
        os.makedirs(jackknife_dir)
    sent_idx = 0
    groups = []
    for ith in xrange(k_fold):
        if ith == k_fold-1:
            groups.append(sents[sent_idx:])
        else:
            groups.append(sents[sent_idx:sent_idx+size])
            sent_idx += size
    for ith in xrange(k_fold):
        ith_dir = os.path.join(jackknife_dir, str(ith))
        conllu_dir = os.path.join(ith_dir, 'conllu')
        if not os.path.isdir(conllu_dir):
            os.makedirs(conllu_dir)
        train_sents = groups[:ith] + groups[ith+1:]
        train_conllu = os.path.join(conllu_dir, 'k_fold-train.conllu')
        output_sep(train_conllu, train_sents)
        dev_sents = groups[ith:ith+1]
        dev_conllu = os.path.join(conllu_dir, 'k_fold-dev.conllu')
        output_sep(dev_conllu, dev_sents)
        config_file = os.path.join(ith_dir, 'k_fold-config.json')
        create_config(config, config_file, ith_dir)
    return jackknife_dir
         

def jackknife_train(jk_base_dir, k_fold):
    for ith in xrange(k_fold):
        new_config = os.path.join(jk_base_dir, str(ith), 'k_fold-config.json')
        preprocess_command = 'python scripts/preprocess.py {}'.format(new_config)
        subprocess.check_call(preprocess_command, shell=True)
        train_command = 'python scripts/train_bilstm_postagger.py {}'.format(new_config)
        print('{}/{} Jackknifing'.format(ith+1, k_fold))
        subprocess.check_call(train_command, shell=True)
######### main ##########

if __name__ == '__main__':
    config_file = sys.argv[1]
    k_fold = int(sys.argv[2])
    config_file = read_config(config_file)
    print('Jackknife Train Supertagger')
    jk_base_dir = separate_training_set(config_file, k_fold)
    jackknife_train(jk_base_dir, k_fold)
    print('Jacknife Training is done.')
    collect_results(config_file, jk_base_dir, k_fold)
    clean_dir(jk_base_dir)
#    best_model = get_best_model(config_file)
#    data_types = config_file['data']['split'].keys()
#    test_stagger(config_file, best_model, data_types)
#    test_stagger(config_file, best_model, ['train'])
