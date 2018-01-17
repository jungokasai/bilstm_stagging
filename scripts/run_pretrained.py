import subprocess
import os
import sys
import json 
from argparse import ArgumentParser 

parser = ArgumentParser()
parser.add_argument('config_file', metavar='N', help='an integer for the accumulator')
parser.add_argument('model_name', metavar='N', help='an integer for the accumulator')
parser.add_argument("--no_gold",  help="compute tag accuracy", action="store_true", default=False)
parser.add_argument("--save_probs",  help="save probabilities", action="store_true", default=False)
parser.add_argument("--get_weight",  help="get stag weight", action="store_true", default=False)
opts = parser.parse_args()

def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict


def test_stagger(config, best_model, data_types, opts):
    base_dir = config['data']['base_dir']
    base_command = 'python bilstm_stagger_main.py test --pretrained --base_dir {}'.format(base_dir)
    model_info = ' --model {}'.format(best_model)
    for data_type in data_types:
        output_file = os.path.join(base_dir, 'predicted_stag', '{}.txt'.format(data_type))
        inputs = {}
        inputs[10] = output_file
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        if opts.no_gold:
            output_info = ' --save_tags {}'.format(output_file)
            test_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', '{}.txt'.format(data_type)), os.path.join(base_dir, 'predicted_pos', '{}.txt'.format(data_type)), os.path.join(base_dir, 'sents', '{}.txt'.format(data_type)))
        ## notice that sents for jk_test and tag_test. If no_gold is True, we don't have the gold data like PETE.

        else:
            output_info = ' --save_tags {} --get_accuracy'.format(output_file)
            test_data_info = ' --text_test {} --jk_test {} --tag_test {}'.format(os.path.join(base_dir, 'sents', '{}.txt'.format(data_type)), os.path.join(base_dir, 'predicted_pos', '{}.txt'.format(data_type)), os.path.join(base_dir, 'gold_stag', '{}.txt'.format(data_type)))

        if opts.save_probs: 
            output_info += ' --save_probs'
        if opts.get_weight: 
            output_info += ' --get_weight'
        complete_command = base_command + model_info + output_info + test_data_info
        subprocess.check_call(complete_command, shell=True)
######### main ##########

if __name__ == '__main__':
    config_file = opts.config_file
    config_file = read_config(config_file)
    best_model = opts.model_name
    data_types = config_file['data']['split'].keys()
    test_stagger(config_file, best_model, data_types, opts)
