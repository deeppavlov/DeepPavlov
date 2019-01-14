import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

import os
import pickle

from collections import defaultdict
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords


@register('answer_generation')
class AnswerGeneration(Component):
    
    def __init__(self, names_load_path, *args, **kwargs):
        g = open(names_load_path, 'rb')
        self.names_dict = pickle.load(g, encoding = "latin1")
    
    def __call__(self, classes, entity_triplets, *args, **kwargs):

        f = open("/home/dmitry/files_new/log.txt", 'w')
        for triplet in entity_triplets[0]:
            f.write("entity_triplets\n"+str(triplet)+'\n')
        f.close()
        
        objects_batch = []
        for n, rel_list in enumerate(classes):
            found = False
            objects = []
            for relation in rel_list:
                for triplet in entity_triplets[n]:
                    if triplet[0][0].split('com/')[1] == relation.split(':')[1].replace('.', '/'):
                        found_objects = triplet[0][1].split(' ')
                        for obj in found_objects:
                            objects.append(obj.split('/')[-1])
                        found = True
                        break
                if found == True:
                    break
            if found == False:
                for relation in rel_list:
                    for triplet in entity_triplets[n]:
                        base_rel = triplet[0][0].split('com/')[1]
                        found_rel = relation.split(':')[1]
                        if base_rel.split('/')[-1] == found_rel.split('.')[-1]:
                            found_objects = triplet[0][1].split(' ')
                            for obj in found_objects:
                                objects.append(obj.split('/')[-1])
                            found = True
                            break
                    if found == True:
                        break
            objects_batch.append(objects)

        word_batch = []
        id_to_word = True
        if id_to_word == True:
            for objects in objects_batch:
                words = []
                for obj in objects:
                    if ("fb:m."+obj) in self.names_dict:
                        words.append(self.names_dict[("fb:m."+obj)])
                word_batch.append(words)

        #return objects_batch
        return word_batch

