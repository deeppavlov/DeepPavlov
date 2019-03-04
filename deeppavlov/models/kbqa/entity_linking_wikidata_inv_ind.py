# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher
import pickle
from pathlib import Path
from collections import defaultdict
import itertools

from fuzzywuzzy import fuzz
import pymorphy2
import nltk

log = getLogger(__name__)


@register('entity_linking_wikidata_inv_ind')
class EntityLinkingWikidata(Component):
    """
        Class for linking the words in the question and the corresponding entity
        in Freebase, then extracting triplets from Freebase with the entity
    """

    def __init__(self, entities_load_path: str,
                 wiki_load_path: str,
                 debug: bool = True,
                 *args, **kwargs) -> None:

        entities_load_path = Path(entities_load_path).expanduser()
        with open(entities_load_path, "rb") as e:
            self.name_to_q = pickle.load(e)
        alphabet = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890-_()=+!?.,/;:&@<>|#$%^*"
        
        dictionary = []
        for key in self.name_to_q:
            dictionary.append(key)
        self.searcher = LevenshteinSearcher(alphabet, dictionary)

        wiki_load_path = Path(wiki_load_path).expanduser()
        with open(wiki_load_path, "rb") as f:
            self.wikidata = pickle.load(f)

        self.morph = pymorphy2.MorphAnalyzer()
        self._debug = debug

    def __call__(self, texts: List[List[str]],
                 tags: List[List[int]],
                 *args, **kwargs) -> List[List[List[str]]]:

        text_entities = []
        for i, text in enumerate(texts):
            entity = []
            for j, tok in enumerate(text):
                if tags[i][j] != 0:
                    entity.append(tok)
            entity = ' '.join(entity)
            text_entities.append(entity)

        if self._debug:
            questions = [' '.join(t) for t in texts]
            log.debug(f'Questions: {questions}')
            log.debug(f'Entities extracted by NER: {text_entities}')

        wiki_entities_batch = []
        confidences = []

        for entity in text_entities:
            if not entity:
                wiki_entities_batch.append(["None"])
                confidences.append([0.0])
            else:
                word_tokens = nltk.word_tokenize(entity)

                candidate_entities = []
                
                frequencies = []
                found_words = []

                for tok in word_tokens:
                    frequency = 0
                    candidate_entity_tok = []
                    if tok in self.name_to_q:
                        candidate_entity_tok += self.name_to_q[tok]
                        frequency += len(self.name_to_q[tok])
                    
                    morph_parse_tok = self.morph.parse(tok)[0]
                    lemmatized_tok = morph_parse_tok.normal_form
                    if lemmatized_tok in self.name_to_q:
                        candidate_entity_tok += self.name_to_q[lemmatized_tok]
                        frequency += len(self.name_to_q[lemmatized_tok])
                    
                    words_with_levens_1 = self.searcher.search(tok, d=1)
                    candidates = []
                    for word in words_with_levens_1:
                        candidates += self.name_to_q[word[0]]
                    if len(candidates) > 0:
                        candidate_entity_tok += candidates
                    
                    if len(candidate_entity_tok) > 0:
                        candidate_entities.append(candidate_entity_tok)
                        frequencies.append(frequency)
                        found_words.append(lemmatized_tok)
                        
                min_freq = max(frequencies)
                min_num = 0
                
                for i in range(len(frequencies)):
                    if frequencies[i] < min_freq:
                        min_freq = frequencies[i]
                        min_num = i

                sets_intersect = defaultdict(list)

                set_intersect = set(candidate_entities[0])
                for i in range((len(candidate_entities)-1)):
                    set_intersect = set_intersect&set(candidate_entities[(i+1)])
                    
                if len(set_intersect) > 0:
                    sets_intersect[len(candidate_entities)].append(set_intersect)
                    
                numbers = [i for i in range(len(candidate_entities))]

                for i in range((len(candidate_entities)-1), 0, -1):
                    combs = itertools.combinations(numbers, i)
                    for comb_0 in combs:
                        comb = list(comb_0)
                        set_intersect = set(candidate_entities[comb[0]])
                        for j in range((len(comb)-1)):
                            set_intersect = set_intersect & set(candidate_entities[comb[(j+1)]])
                        if len(set_intersect) > 0:
                            sets_intersect[i].append(set_intersect)
                
                entities = []
                        
                for i in range(len(candidate_entities), 0, -1):
                    if len(sets_intersect[i]) > 0:
                        if i == 1:
                            for ent in sets_intersect[i][min_num]:
                                title = ent[0]
                                #ratio = fuzz.ratio(title.lower(), found_words[min_num].lower())
                                #entities.append((ent, ratio))
                                if title.lower().find(found_words[min_num].lower()) > -1:
                                    entities.append(ent)
                        else:
                            for set_intersect in sets_intersect[i]:
                                for ent in set_intersect:
                                    title = ent[0]
                                    #ratio = fuzz.ratio(title.lower(), found_words[min_num].lower())
                                    #entities.append((ent, ratio))
                                    entities.append(ent)
                        break
                            
                srtd = sorted(entities, key=lambda x: x[2], reverse=True)
                if len(srtd) > 0:
                    #wiki_entities_batch.append([srtd[i][0][1] for i in range(len(srtd))])
                    wiki_entities_batch.append([srtd[i][1] for i in range(len(srtd))])
                    confidences.append([1.0 for i in range(len(srtd))])
                else:
                    wiki_entities_batch.append(["None"])
                    confidences.append([0.0])
        
        if self._debug:
            log.debug(f'results of entity linking: {wiki_entities_batch[0][:5]}')                

        entity_triplets_batch = []
        for entity_ids in wiki_entities_batch:
            entity_triplets = []
            for entity_id in entity_ids:
                if entity_id in self.wikidata and entity_id.startswith('Q'):
                    entity_triplets.append(self.wikidata[entity_id])
                else:
                    entity_triplets.append([])
            entity_triplets_batch.append(entity_triplets)

        return entity_triplets_batch, confidences


