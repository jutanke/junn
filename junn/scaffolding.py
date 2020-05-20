from abc import abstractmethod
import time
from os.path import isdir, join, isfile
from os import makedirs
import shutil
import numpy as np
import torch
import torch.nn as nn
from junn.settings import get_data_loc
import pandas as pd


class Scaffolding(nn.Module):

    def __init__(self, force_new_training, model_seed=0):
        """
        :param force_new_training:
        :param model_seed: identifiy different training runs for the same model
        """
        super(Scaffolding, self).__init__()
        self.project_folder = ''
        self.model_seed = model_seed
        self.is_weights_loaded = False
        self.force_new_training = force_new_training
        self.scaffolding_is_init = False
    
    def init(self):
        if not self.scaffolding_is_init:
            force_new_training = self.force_new_training
            train_dir = self.get_train_dir()
            if isdir(train_dir) and force_new_training:
                print('[hma-with-symbolic-label][model] - delete dir:', train_dir)
                shutil.rmtree(train_dir)
                time.sleep(.5)
            if not isdir(train_dir):
                makedirs(train_dir)
            fweights = self.get_weights_file()
            if isfile(fweights):
                self.is_weights_loaded = True
            self.scaffolding_is_init = True

    @abstractmethod
    def get_unique_directory(self):
        raise NotImplementedError

    def number_of_parameters(self):
        total_sum = []
        for param in self.parameters():
            total_sum.append(np.product(param.size()))
        return np.sum(total_sum)

    def prettyprint_number_of_parameters(self):
        n_params = self.number_of_parameters()
        return '{:,}'.format(n_params)

    def load_weights_if_possible(self):
        if isfile(self.get_weights_file()):
            checkpoint = torch.load(self.get_weights_file())
            self.load_state_dict(checkpoint['model_state_dict'])
            self.is_weights_loaded = True
            return True
        self.is_weights_loaded = False
        return False

    def save_weights(self, epoch=-1, optim=None, name=''):
        if optim is None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict()
            }, self.get_weights_file(name=name))
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optim_state_dict': optim.state_dict()
            }, self.get_weights_file(name=name))

    def get_train_dir(self):
        
        if len(self.project_folder) > 0:
            train_dir = self.project_folder
        else:
            train_dir = join(get_data_loc(), 'training')
        return join(join(train_dir, self.get_unique_directory()),
                    'seed' + str(self.model_seed))

    def get_weights_file(self, name=''):
        return join(self.get_train_dir(), 'weights' + name + '.h5')

    def get_log_file(self):
        return join(self.get_train_dir(), 'training.csv')
