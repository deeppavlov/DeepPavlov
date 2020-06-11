# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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

import copy
import json
from collections import Counter
from logging import getLogger
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.stats import entropy

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import save_pickle, load_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.deprecated.skill import Skill
from deeppavlov.metrics.bleu import bleu_advanced

log = getLogger(__name__)


@register("ecommerce_skill_bleu")
class EcommerceSkillBleu(Skill):
    """Class to retrieve product items from `load_path` catalogs
    in sorted order according to the similarity measure
    Retrieve the specification attributes with corresponding values
    in sorted order according to entropy.

    Parameters:
        preprocess: text preprocessing component
        save_path: path to save a model
        load_path: path to load a model
        entropy_fields: the specification attributes of the catalog items
        min_similarity: similarity threshold for ranking
        min_entropy: min entropy threshold for specifying
    """

    def __init__(self,
                 preprocess: Component,
                 save_path: str,
                 load_path: str,
                 entropy_fields: list,
                 min_similarity: float = 0.5,
                 min_entropy: float = 0.5,
                 **kwargs) -> None:

        self.preprocess = preprocess
        self.save_path = expand_path(save_path)

        if isinstance(load_path, list):
            self.load_path: List = [expand_path(path) for path in load_path]
        else:
            self.load_path: List = [expand_path(load_path)]

        self.min_similarity = min_similarity
        self.min_entropy = min_entropy
        self.entropy_fields = entropy_fields
        self.ec_data: List = []
        if kwargs.get('mode') != 'train':
            self.load()

    def fit(self, data: List[Dict[Any, Any]]) -> None:
        """Preprocess items `title` and `description` from the `data`

        Parameters:
            data: list of catalog items

        Returns:
            None
        """

        log.info(f"Items to nlp: {len(data)}")
        self.ec_data = [dict(item, **{
            'title_nlped': self.preprocess.spacy2dict(self.preprocess.analyze(item['Title'])),
            'feat_nlped': self.preprocess.spacy2dict(self.preprocess.analyze(item['Title'] + '. ' + item['Feature']))
        }) for item in data]
        log.info('Data are nlped')

    def save(self, **kwargs) -> None:
        """Save classifier parameters"""
        log.info(f"Saving model to {self.save_path}")
        save_pickle(self.ec_data, self.save_path)

    def load(self, **kwargs) -> None:
        """Load classifier parameters"""
        log.info(f"Loading model from {self.load_path}")
        for path in self.load_path:
            if Path.is_file(path):
                self.ec_data += load_pickle(path)
            else:
                raise FileNotFoundError

        log.info(f"Loaded items {len(self.ec_data)}")

    def __call__(self, queries: List[str], history: List[Any], states: List[Dict[Any, Any]]) -> \
            Tuple[Tuple[List[Any], List[Any]], List[float], List[Any]]:
        """Retrieve catalog items according to the BLEU measure

        Parameters:
            queries: list of queries
            history: list of previous queries
            states: list of dialog state

        Returns:
            response:   items:      list of retrieved items
                        entropies:  list of entropy attributes with corresponding values

            confidence: list of similarity scores
            state: dialog state
        """

        response: List = []
        confidence: List = []
        results_args: List = []
        entropies: List = []
        back_states: List = []
        results_args_sim: List = []

        log.debug(f"queries: {queries} states: {states}")

        for item_idx, query in enumerate(queries):

            state = states[item_idx]

            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except:
                    state = self.preprocess.parse_input(state)

            if not state:
                state = {}

            start = state['start'] if 'start' in state else 0
            stop = state['stop'] if 'stop' in state else 5

            state['start'] = start
            state['stop'] = stop

            query = self.preprocess.analyze(query)

            query, money_range = self.preprocess.extract_money(query)
            log.debug(f"money detected: {query} {money_range}")

            if len(money_range) == 2:
                state['Price'] = money_range

            score_title = [bleu_advanced(self.preprocess.lemmas(item['title_nlped']),
                                         self.preprocess.lemmas(self.preprocess.filter_nlp_title(query)),
                                         weights=(1,), penalty=False) for item in self.ec_data]

            score_feat = [bleu_advanced(self.preprocess.lemmas(item['feat_nlped']),
                                        self.preprocess.lemmas(self.preprocess.filter_nlp(query)),
                                        weights=(0.3, 0.7), penalty=False) for idx, item in enumerate(self.ec_data)]

            scores = np.mean([score_feat, score_title], axis=0).tolist()

            scores_title = [(score, -len(self.ec_data[idx]['Title'])) for idx, score in enumerate(scores)]

            raw_scores_ar = np.array(scores_title, dtype=[('x', 'float_'), ('y', 'int_')])

            results_args = np.argsort(raw_scores_ar, order=('x', 'y'))[::-1].tolist()

            results_args_sim = [idx for idx in results_args if scores[idx] >= self.min_similarity]

            log.debug(
                f"Items before similarity filtering {len(results_args)} and after {len(results_args_sim)} with th={self.min_similarity} " +
                f"the best one has score {scores[results_args[0]]} with title {self.ec_data[results_args[0]]['Title']}")

            results_args_sim = self._filter_state(state, results_args_sim)

            results_args_sim_fil = [idx for idx in results_args_sim[start:stop]]

            local_response = self._clean_items(results_args_sim_fil)

            response.append(local_response)

            confidence.append([(score_title[idx], score_feat[idx])
                               for idx in results_args_sim[start:stop]])

            entropies.append(self._entropy_subquery(results_args_sim))
            log.debug(f"Total number of relevant answers {len(results_args_sim)}")
            back_states.append(state)

        return (response, entropies), confidence, back_states

    def _clean_items(self, results: List[int]) -> List[Any]:
        local_response: List = []
        for idx in results:
            temp = copy.copy(self.ec_data[idx])
            del temp['title_nlped']
            del temp['feat_nlped']
            local_response.append(temp)
        return local_response

    def _filter_state(self, state: Dict[Any, Any], results_args_sim: List[int]) -> List[Any]:
        for key, value in state.items():
            log.debug(f"Filtering for {key}:{value}")

            if key == 'Price':
                price = value
                log.debug(f"Items before price filtering {len(results_args_sim)} with price {price}")
                results_args_sim = [idx for idx in results_args_sim
                                    if price[0] <= self.preprocess.price(self.ec_data[idx]) <= price[1] and
                                    self.preprocess.price(self.ec_data[idx]) != 0]
                log.debug(f"Items after price filtering {len(results_args_sim)}")

            elif key in ['query', 'start', 'stop', 'history']:
                continue

            else:
                results_args_sim = [idx for idx in results_args_sim
                                    if key in self.ec_data[idx]
                                    if self.ec_data[idx][key].lower() == value.lower()]

        return results_args_sim

    def _entropy_subquery(self, results_args: List[int]) -> List[Tuple[float, str, List[Tuple[str, int]]]]:
        """Calculate entropy of selected attributes for items from the catalog.

        Parameters:
            results_args: items id to consider

        Returns:
            entropies: entropy score with attribute name and corresponding values
        """

        ent_fields: Dict = {}

        for idx in results_args:
            for field in self.entropy_fields:
                if field in self.ec_data[idx]:
                    if field not in ent_fields:
                        ent_fields[field] = []

                    ent_fields[field].append(self.ec_data[idx][field].lower())

        entropies = []
        for key, value in ent_fields.items():
            count = Counter(value)
            entropies.append(
                (entropy(list(count.values()), base=2), key, count.most_common()))

        entropies = sorted(entropies, key=itemgetter(0), reverse=True)
        entropies = [
            ent_item for ent_item in entropies if ent_item[0] >= self.min_entropy]

        return entropies
