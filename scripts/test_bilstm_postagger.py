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


def test_postagger(config, best_model, data_types):
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
        output_conllu(os.path.join(base_dir, config['data']['split'][data_type]), os.path.join(base_dir, config['data']['split'][data_type]+'_stag'), inputs)
######### main ##########

if __name__ == '__main__':
    config_file = sys.argv[1]
    config_file = read_config(config_file)
    best_model = sys.argv[2]
    data_types = config_file['data']['split'].keys()
    test_postagger(config_file, best_model, data_types)
#    test_stagger(config_file, best_model, ['train'])
