import os
import pickle
from collections import defaultdict
from nltk.corpus import stopwords
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common_file import load_pickle, save_pickle
from deeppavlov.core.commands.utils import expand_path


class EntitiesParser(Serializable):
    def __init__(self, load_path: str = "~/.deeppavlov/downloads/wikidata_rus",
                       save_path: str = "~/.deeppavlov/downloads/wikidata_rus",
                       name_to_idslist_filename: str = "name_to_idlist_ru.pickle",
                       ent_list_filename: str = "ent_list_rus.pickle",
                       word_to_idlist_filename: str = "word_to_idlist_rus.pickle",
                       entities_ranking_dict_filename: str = "entities_ranking_dict_rus.pickle",
                       entities_descr_filename: str = "q_to_descr_ru.pickle"):
                       
        self.wiki_dict = {}
        
        self.load_path = load_path
        self.save_path = save_path
        self.name_to_idlist_filename = name_to_idlist_filename
        self.ent_list_rus_filename = ent_list_rus_filename
        self.word_to_idlist_filename = word_to_idlist_filename
        self.entities_ranking_dict_filename = entities_ranking_dict_filename
        self.entities_descr_filename = entities_descr_filename
    
        self.name_to_idlist = defaultdict(list)
        self.word_to_idlist = {}
        self.flat_list = []
        self.entities_ranking_dict = {}
        self.entities_descr = {}
        
        self.stopwords = set(stopwords.words("russian"))
        self.alphabet_full = set(" `abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789.,?!@#$%^&*()-+=\/№;:<>_–|")
        self.alphabet_full.add('"')
        self.alphabet_full.add("'")
        self.alphabet = set("abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789-")
        
    def load(self):
        self.load_path = str(expand_path(self.load_path))
        files = os.listdir(self.load_path)
        for fl in files:
            wiki_chunk = load_pickle(self.load_path / fl)
            for elem in wiki_chunk:
                self.wiki_dict[elem] = wiki_chunk[elem]
        
    def save(self):
        save_pickle(self.name_to_idlist, self.save_path / self.name_to_idlist_filename)
        save_pickle(self.word_to_idlist, self.save_path / self.word_to_idlist_filename)
        save_pickle(self.flat_list, self.save_path / self.ent_list_filename)
        save_pickle(self.entities_ranking_dict, self.save_path / self.entities_ranking_dict_filename)
        save_pickle(self.entities_descr, self.save_path / self.entities_descr_filename)
        
    def parse(self):
        for entity_id in self.wiki_dict:
            entity_info = self.wiki_dict[entity_id]
            name = entity_info.get("name", "")
            aliases = entity_info.get("aliases", [])
            if name:
                self.name_to_idlist[name].append(entity_id)
            if aliases:
                for alias in aliases:
                    self.name_to_idlist[alias].append(entity_id)
            number_of_relations = entity_info.get("number_of_relations", 0)
            self.entities_ranking_dict[entity_id] = number_of_relations
        
        for entity_id in self.wiki_dict:
            entity_info = self.wiki_dict[entity_id]
            descr = entity_info.get("descr", "")
            triplets = entity_info.get("triplets", [])
            if not descr:
                descr = self.find_descr(triplets)
            if descr:
                self.entities_descr_filename[entity_id] = descr
                    
        word_to_idlist = defaultdict(list)

        for label in self.name_to_idlist:
            bad_symb = False
            for symb in label:
                if symb not in self.alphabet_full:
                    bad_symb = True
                    break
            if not bad_symb:
                label_split = label.split(' ')
                num_words = 0
                for label_elem in label_split:
                    label_sanitized = ''.join([ch for ch in label_elem if ch in self.alphabet])
                    if len(label_sanitized) > 1 and label_sanitized not in self.stopwords:
                        num_words += 1

                if num_words > 0:
                    for label_elem in label_split:
                        label_sanitized = ''.join([ch for ch in label_elem if ch in self.alphabet])
                        if len(label_sanitized) > 1 and label_sanitized not in self.stopwords:
                            word_to_idlist[label_sanitized] += \
                                [(entity, num_words) for entity in self.name_to_idlist[label]]


        word_to_idlist_init = {label: list(set(idlist)) for label, idlist in word_to_idlist.items()}

        count = 0
        for elem in word_to_idlist_init:
            start = count
            end = start + len(word_to_idlist_init[elem])
            count = end
            self.flat_list += word_to_idlist_init[elem]
            self.word_to_idlist[elem] = [start, end]
            
def find_descr(self, triplets):
    descr = ""
    for rel, *objects in triplets:
        if rel == "P31":
            for obj in objects:
                if obj != "Q5":
                    obj_label = self.wiki_dict.get(obj, {}).get("name", "")
                    if obj_label:
                        return obj_label
        if rel == "P106":
            obj_label = self.wiki_dict.get(objects[0], {}).get("name", "")
            if obj_label:
                return obj_label
    return descr            

