"""
Copyright 2018 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import save_pickle, load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs
from deeppavlov.core.models.estimator import Estimator, Component
from deeppavlov.metrics.bleu import bleu_advanced

import copy
import json
import numpy as np
from scipy.stats import entropy
from collections import Counter
from typing import List, Tuple, Dict, Any
from operator import itemgetter

log = get_logger(__name__)


@register("ecommerce_bot")
class EcommerceBot(Component):
    """Class to perform ranking catalogue items accroding to the query
    Ask questions (according to entropy) to specify the query

    Attributes:
        min_similarity: simililarity threshlold for ranking
        min_entropy: min entropy threshold for specifying
        tokenizer: component for SpaCy analyze
        preprocess: text preprocessing component
        state: global state
    """

    def __init__(self, preprocess: Component, tokenizer: Component, save_path: str, 
        load_path: str, min_similarity: float = 0.5, min_entropy: float = 0.5, **kwargs) -> None:
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.min_similarity = min_similarity
        self.min_entropy = min_entropy
        if kwargs.get('mode') != 'train':
            self.load()

    def fit(self, x) -> None:
        log.info('Items to nlp: '+str(len(x)))
        self.ec_data = [dict(item, **{'title_nlped': self.tokenizer.analyze(item['Title']),
                                      'feat_nlped':self.tokenizer.analyze(item['Title']+'. '+item['Feature'])}) for item in x]
        log.info('Data are nlped')

    def save(self, **kwargs) -> None:
        log.info(f"Saving model to {self.save_path}")
        make_all_dirs(self.save_path)
        save_pickle(self.ec_data, self.save_path)

    def load(self, **kwargs) -> None:
        log.info(f"Loading model from {self.load_path}")
        self.ec_data = load_pickle(self.load_path)

    def __call__(self, X, State, **kwargs) -> Tuple[Dict[str, Dict[Any, Any]], List[float], Dict[Any, Any]]:
        response = []
        confidence = []
        results_args = []
        results_args_sim = []

        log.debug(f"query: {X} state:{State}")


        for item_idx, query in enumerate(X):

            state = State[item_idx]
            start = state['start'] if 'start' in state else 0
            stop = state['stop'] if 'stop' in state else 5

            state['start'] = start
            state['stop'] = stop

            query = self.tokenizer.analyze(query)

            query, money_range = self.tokenizer.extract_money(query)
            log.debug(f"money detected: {query} {money_range}")

            if len(money_range) == 2:
                state['Price'] = money_range
            
            score_title = []
            for item in self.ec_data:
                hyp = self.preprocess.lemmas(
                    self.preprocess.filter_nlp_title(query))
                ref = self.preprocess.lemmas(item['title_nlped'])
                score_title.append(bleu_advanced(
                    hyp, ref, weights=(1,), penalty=False))

            score_feat = []
            for idx, item in enumerate(self.ec_data):
                hyp = self.preprocess.lemmas(
                    self.preprocess.filter_nlp(query))
                ref = self.preprocess.lemmas(item['feat_nlped'])
                score_feat.append(bleu_advanced(
                    hyp, ref, weights=(0.3, 0.7), penalty=False))

            scores = np.mean(
                [score_feat, score_title], axis=0).tolist()

            scores_title = [(score, len(self.ec_data[idx]['Title']))
                            for idx, score in enumerate(scores)]

            raw_scores_ar = np.array(scores_title, dtype=[
                ('x', 'float_'), ('y', 'int_')])

            results_args = np.argsort(raw_scores_ar, order=('x', 'y'))[
                ::-1].tolist()

            results_args_sim = [
                idx for idx in results_args if scores[idx] >= self.min_similarity]

            log.debug(f"Items before similarity filtering {len(results_args)} and after {len(results_args_sim)} with th={self.min_similarity} "+
                f"the best one has score {scores[results_args[0]]} with title {self.ec_data[results_args[0]]['Title']}")

            for key, value in state.items():
                log.debug(f"Filtering for {key}:{value}")

                if key == 'Price':
                    price = value
                    log.debug(f"Items before price filtering {len(results_args_sim)} with price {price}")
                    results_args_sim = [idx for idx in results_args_sim
                                    if self.preprocess.price(self.ec_data[idx]) >= price[0] and
                                    self.preprocess.price(self.ec_data[idx]) <= price[1] and
                                    self.preprocess.price(self.ec_data[idx]) != 0]
                    log.debug(f"Items after price filtering {len(results_args_sim)}")
                    
                elif key in ['query', 'start', 'stop']:
                    continue

                else:
                    results_args_sim = [idx for idx in results_args_sim if key in self.ec_data[idx] if self.ec_data[idx][key].lower() == value.lower()]

            response = []
            for idx in results_args_sim[start:stop]:
                temp = copy.deepcopy(self.ec_data[idx])
                del temp['title_nlped']
                del temp['feat_nlped']
                response.append(temp)

            confidence.append(
                [scores[idx] for idx in results_args_sim[start:stop]])
            log.debug(f"Response confidence {[scores[idx] for idx in results_args_sim[start:stop+2]]}")

            entropies = self._entropy_subquery(
                scores, results_args_sim)
            log.debug(f"Response entropy {entropies}")

        return json.dumps(({'items': response, 'entropy': entropies}, confidence, state))
        # return {'items': response, 'entropy': entropies}, confidence

    def _entropy_subquery(self, scores, results_args) -> List[Tuple[float, str, List[Tuple[str, int]]]]:
        fields = ['Size', 'Brand', 'Author', 'Color', 'Genre']
        ent_fields: Dict = {}

        i = 0
        for idx in results_args:
            for field in fields:
                if field in self.ec_data[idx]:
                    if field not in ent_fields:
                        ent_fields[field] = []

                    ent_fields[field].append(self.ec_data[idx][field].lower())
            i += 1

        entropies = []

        for key, value in ent_fields.items():
            count = Counter(value)
            entropies.append(
                (entropy(list(count.values()), base=2), key, count.most_common()))

        entropies = sorted(entropies, key=itemgetter(0), reverse=True)
        entropies = [ent_item for ent_item in entropies if ent_item[0]
                     >= self.min_entropy]

        return entropies