import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

import os
import pickle

@register('max_n')
class Max_n(Component):
    
    def __init__(self, filename, *args, **kwargs):
        fl = open(filename, 'r')
        lines = fl.readlines()
        self.classes = [line.split('\t')[0] for line in lines]
        
    
    def __call__(self, probas_batch, *args, **kwargs):
        
        max_n_batch = []

        for probas in probas_batch:
            max_n = np.asarray(probas).argsort()[-5:][::-1]
            max_n_classes = [self.classes[num] for num in max_n]
            max_n_batch.append(max_n_classes)
        
        return max_n_batch
