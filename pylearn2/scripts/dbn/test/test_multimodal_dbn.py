'''
Created on Jul 15, 2015

@author: ningzhang
'''
"""
This module tests multimodal_dbn 
"""

import os

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse


@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train_layer1(yaml_file_path, save_path):

    yaml = open("{0}/image_rbm1.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'batch_size': 100,
                    'monitoring_batches': 100,
                    'nhid': 2048,
                    'num_chains': 100,
                    'num_gibbs_steps': 1,
                    'max_epochs': 10000,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_layer2(yaml_file_path, save_path):

    yaml = open("{0}/image_rbm2.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'nvis': 2048,
                    'batch_size': 100,
                    'monitoring_batches': 100,
                    'nhid': 2048,
                    'num_chains': 100,
                    'num_gibbs_steps': 1,
                    'max_epochs': 10000,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_mlp(yaml_file_path, save_path):

    yaml = open("{0}/dae_mlp.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'batch_size': 50,
                    'max_epochs': 1,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def test_multimodal_dbn():

    skip.skip_if_no_data()

    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..'))
    save_path = os.path.dirname(os.path.realpath(__file__))

    train_layer1(yaml_file_path, save_path)
    train_layer2(yaml_file_path, save_path)
 #   train_mlp(yaml_file_path, save_path)

    try:
        os.remove("{}/image_rbm1.pkl".format(save_path))
        os.remove("{}/image_rbm2.pkl".format(save_path))
    except OSError:
        pass

if __name__ == '__main__':
    test_multimodal_dbn()
