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

from collections import Counter
from logging import getLogger
from operator import itemgetter
from typing import List, Tuple, Dict, Union, Any

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import norm as sparse_norm
from scipy.stats import entropy

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import save_pickle, load_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

log = getLogger(__name__)


@register("ecommerce_skill_tfidf")
class EcommerceSkillTfidf(Component):
    """Class to retrieve product items from `load_path` catalogs
    in sorted order according to the similarity measure
    Retrieve the specification attributes with corresponding values
    in sorted order according to entropy.

    Parameters:
        save_path: path to save a model
        load_path: path to load a model
        entropy_fields: the specification attributes of the catalog items
        min_similarity: similarity threshold for ranking
        min_entropy: min entropy threshold for specifying
    """

    def __init__(self,
                 save_path: str,
                 load_path: str,
                 entropy_fields: list,
                 min_similarity: float = 0.5,
                 min_entropy: float = 0.5,
                 **kwargs) -> None:

        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.min_similarity = min_similarity
        self.min_entropy = min_entropy
        self.entropy_fields = entropy_fields
        self.ec_data: List = []
        self.x_train_features = None
        if kwargs.get('mode') != 'train':
            self.load()

    def fit(self, data, query) -> None:
        """Preprocess items `title` and `description` from the `data`

        Parameters:
            data: list of catalog items

        Returns:
            None
        """

        self.x_train_features = vstack(list(query))
        self.ec_data = data

    def save(self) -> None:
        """Save classifier parameters"""
        log.info("Saving to {}".format(self.save_path))
        path = expand_path(self.save_path)
        save_pickle((self.ec_data, self.x_train_features), path)

    def load(self) -> None:
        """Load classifier parameters"""
        log.info("Loading from {}".format(self.load_path))
        self.ec_data, self.x_train_features = load_pickle(
            expand_path(self.load_path))

    def __call__(self, q_vects: List[csr_matrix], histories: List[Any], states: List[Dict[Any, Any]]) -> Tuple[
        Tuple[List[Dict[Any, Any]], List[Any]], List[float], Dict[Any, Any]]:
        """Retrieve catalog items according to the TFIDF measure

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

        log.info(f"Total catalog {len(self.ec_data)}")

        if not isinstance(q_vects, list):
            q_vects = [q_vects]

        if not isinstance(states, list):
            states = [states]

        if not isinstance(histories, list):
            histories = [histories]

        items: List = []
        confidences: List = []
        back_states: List = []
        entropies: List = []

        for idx, q_vect in enumerate(q_vects):

            log.info(f"Search query {q_vect}")

            if len(states) >= idx + 1:
                state = states[idx]
            else:
                state = {'start': 0, 'stop': 5}

            if not isinstance(state, dict):
                state = {'start': 0, 'stop': 5}

            if 'start' not in state:
                state['start'] = 0
            if 'stop' not in state:
                state['stop'] = 5

            if 'history' not in state:
                state['history'] = []

            log.info(f"Current state {state}")

            if state['history']:
                his_vect = self._list_to_csr(state['history'][-1])
                if not np.array_equal(his_vect.todense(), q_vect.todense()):
                    q_comp = q_vect.maximum(his_vect)
                    complex_bool = self._take_complex_query(q_comp, q_vect)
                    log.info(f"Complex query:{complex_bool}")

                    if complex_bool is True:
                        q_vect = q_comp
                        state['start'] = 0
                        state['stop'] = 5
                    else:
                        # current short query wins that means that the state should be zeroed
                        state['history'] = []
                else:
                    log.info("the save query came")
            else:
                log.info("history is empty")

            state['history'].append(self._csr_to_list(q_vect))
            log.info(f"Final query {q_vect}")

            scores = self._similarity(q_vect)
            answer_ids = np.argsort(scores)[::-1]
            answer_ids = [idx for idx in answer_ids if scores[idx] >= self.min_similarity]

            answer_ids = self._state_based_filter(answer_ids, state)

            items.append([self.ec_data[idx]
                          for idx in answer_ids[state['start']:state['stop']]])
            confidences.append(
                [scores[idx] for idx in answer_ids[state['start']:state['stop']]])
            back_states.append(state)

            entropies.append(self._entropy_subquery(answer_ids))
        return (items, entropies), confidences, back_states

    def _csr_to_list(self, csr: csr_matrix) -> List[Any]:
        return [csr.data.tolist(), csr.indices.tolist()]

    def _list_to_csr(self, _list: List) -> csr_matrix:
        row_ind = [0] * len(_list[0])
        col_ind = _list[1]
        return csr_matrix((_list[0], (row_ind, col_ind)))

    def _take_complex_query(self, q_prev: csr_matrix, q_cur: csr_matrix) -> bool:
        """Decides whether to use the long compound query or the current short query

        Parameters:
            q_prev: previous query
            q_cur: current query

        Returns:
            Bool: whether to use the compound query
        """

        prev_sim = self._similarity(q_prev)
        cur_sim = self._similarity(q_cur)

        log.debug(f"prev_sim.max(): {prev_sim.max()}")
        log.debug(f"cur_sim.max(): {cur_sim.max()}")

        if prev_sim.max() > cur_sim.max():
            return True

        return False

    def _similarity(self, q_vect: Union[csr_matrix, List]) -> List[float]:
        """Calculates cosine similarity between the user's query and product items.

        Parameters:
            q_cur: user's query

        Returns:
            cos_similarities: lits of similarity scores
        """

        norm = sparse_norm(q_vect) * sparse_norm(self.x_train_features, axis=1)
        cos_similarities = np.array(q_vect.dot(self.x_train_features.T).todense()) / norm

        cos_similarities = cos_similarities[0]
        cos_similarities = np.nan_to_num(cos_similarities)
        return cos_similarities

    def _state_based_filter(self, ids: List[int], state: Dict[Any, Any]):
        """Filters the candidates based on the key-values from the state

        Parameters:
            ids: list of candidates
            state: dialog state

        Returns:
            ids: filtered list of candidates
        """

        for key, value in state.items():
            log.debug(f"Filtering for {key}:{value}")

            if key in ['query', 'start', 'stop', 'history']:
                continue

            else:
                ids = [idx for idx in ids
                       if key in self.ec_data[idx]
                       if self.ec_data[idx][key].lower() == value.lower()]
        return ids

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
            entropies.append((entropy(list(count.values()), base=2), key, count.most_common()))

        entropies = sorted(entropies, key=itemgetter(0), reverse=True)
        entropies = [ent_item for ent_item in entropies if ent_item[0] >= self.min_entropy]

        return entropies
