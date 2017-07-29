import subprocess
import os
import sys
import json 
from tools.conllu2sents import conllu2sents 
def converter(config):
    data_types = ['train', 'dev', 'test']
    features = ['sents', 'gold_pos', 'gold_stag']
    for feature in features:
        for data_type in data_types:
            input_file = os.path.join(config['data']['base_dir'], config['data'][data_type])
            output_file = os.path.join(config['data']['base_dir'], feature, data_type+'.txt')
            if not os.path.isdir(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            if feature == 'sents':
                index = 1
            elif feature == 'gold_pos':
                index = 4
            elif feature == 'gold_stag':
                index = 10
            conllu2sents(index, input_file, output_file)

def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict

def train_pos_tagger(config):
    base_dir = config['data']['base_dir']
    base_command = 'python bilstm_stagger_main.py train --task POS_models --base_dir {}'.format(base_dir)
    train_data_info = ' --text_train {} --jk_train {} --tag_train {}'.format(os.path.join(base_dir, 'sents', 'train.txt'), os.path.join(base_dir, 'gold_pos', 'train.txt'), os.path.join(base_dir, 'gold_pos', 'train.txt'))
    dev_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', 'dev.txt'), os.path.join(base_dir, 'gold_pos', 'dev.txt'), os.path.join(base_dir, 'gold_pos', 'dev.txt'))
    model_config_dict = config['pos_parameters']
    model_config_info = ''
    for option, value in model_config_dict.items():
        model_config_info += ' --{} {}'.format(option, value)
    complete_command = base_command + train_data_info + dev_data_info + model_config_info
    subprocess.check_call(complete_command, shell=True)

def train_stagger(config):
    base_dir = config['data']['base_dir']
    base_command = 'python bilstm_stagger_main.py train --task Super_models --base_dir {}'.format(base_dir)
    train_data_info = ' --text_train {} --jk_train {} --tag_train {}'.format(os.path.join(base_dir, 'sents', 'train.txt'), os.path.join(base_dir, 'gold_stag', 'train.txt'), os.path.join(base_dir, 'gold_pos', 'train.txt'))
    dev_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', 'dev.txt'), os.path.join(base_dir, 'gold_stag', 'dev.txt'), os.path.join(base_dir, 'gold_pos', 'dev.txt'))
    model_config_dict = config['stag_parameters']
    model_config_info = ''
    for option, value in model_config_dict.items():
        model_config_info += ' --{} {}'.format(option, value)
    complete_command = base_command + train_data_info + dev_data_info + model_config_info
    subprocess.check_call(complete_command, shell=True)

######### main ##########

if __name__ == '__main__':
    config_file = sys.argv[1]
    config_file = read_config(config_file)
    print('Convert conllu+stag file to sentences, gold pos, and gold stag')
    converter(config_file)
#    print('Train POS-tagger')
#    train_pos_tagger(config_file)
#    print('Run Jackknife Training of POS tagging for Supertagging')
    print('Train Supertagger')
    train_stagger(config_file)
