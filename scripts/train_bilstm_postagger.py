import subprocess
import os
import sys
import json 
import tools
from tools.converters.conllu2sents import conllu2sents
from tools.converters.sents2conllustag import output_conllu

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

#def train_stagger(config):
#    base_dir = config['data']['base_dir']
#    base_command = 'python bilstm_stagger_main.py train --task Super_models --base_dir {}'.format(base_dir)
#    train_data_info = ' --text_train {} --jk_train {} --tag_train {}'.format(os.path.join(base_dir, 'sents', 'train.txt'), os.path.join(base_dir, 'gold_pos', 'train.txt'), os.path.join(base_dir, 'gold_pos', 'train.txt'))
#    dev_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', 'dev.txt'), os.path.join(base_dir, 'gold_pos', 'dev.txt'), os.path.join(base_dir, 'gold_pos', 'dev.txt'))
#    model_config_dict = config['stag_parameters']
#    model_config_info = ''
#    for option, value in model_config_dict.items():
#        model_config_info += ' --{} {}'.format(option, value)
#    complete_command = base_command + train_data_info + dev_data_info + model_config_info
##    complete_command += ' --max_epochs 1' ## for debugging
#    subprocess.check_call(complete_command, shell=True)
#
def test_pos_tagger(config, best_model, data_types):
    base_dir = config['data']['base_dir']
    base_command = 'python bilstm_stagger_main.py test'
    model_info = ' --model {}'.format(best_model)
    for data_type in data_types:
        output_file = os.path.join(base_dir, 'predicted_pos', '{}.txt'.format(data_type))
        inputs = {}
        inputs[10] = output_file
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info = ' --save_tags {} --get_accuracy'.format(output_file)
        test_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', '{}.txt'.format(data_type)), os.path.join(base_dir, 'gold_pos', '{}.txt'.format(data_type)), os.path.join(base_dir, 'gold_pos', '{}.txt'.format(data_type)))
        complete_command = base_command + model_info + output_info + test_data_info
        subprocess.check_call(complete_command, shell=True)
########## main ##########

if __name__ == '__main__':
    config_file = sys.argv[1]
    config_file = read_config(config_file)
#    print('Convert conllu+stag file to sentences, gold pos, and gold stag')
#    converter(config_file)
#    print('Train POS-tagger')
#    train_pos_tagger(config_file)
#    print('Run Jackknife Training of POS tagging for Supertagging')
    print('Train a POS tagger')
#    train_stagger(config_file)
    train_pos_tagger(config_file)
#    print('Training is done. Run the POS-tagger.')
#    best_model = get_best_model(config_file)
#    data_types = config_file['data']['split'].keys()
#    test_pos_tagger(config_file, best_model, data_types)
#    test_stagger(config_file, best_model, ['train'])
