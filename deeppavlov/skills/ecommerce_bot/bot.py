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
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.estimator import Estimator, Component
from deeppavlov.metrics.bleu import bleu_advanced

import re
import json
import numpy as np
from scipy.stats import entropy
from collections import Counter
from typing import List, Tuple, Dict, Any
from operator import itemgetter

log = get_logger(__name__)

@register("ecommerce_bot")
class EcommerceBot(Component):
    """Rank catalogue items and specify user query

    Attributes:
        min_similarity: simililarity threshlold for ranking
        min_entropy: min entropy threshold for specifying
        tokenizer: component for SpaCy analyze
        preprocess: text preprocessing component
    """

    def __init__(self, preprocess: Component, tokenizer: Component, save_path: str, load_path: str, min_similarity: float = 0.5, min_entropy: float = 0.5, **kwargs) -> None:
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.load_path = load_path
        self.min_similarity = min_similarity
        self.min_entropy = min_entropy
        self.state = None
        if kwargs.get('mode') != 'train':
            self.load()

    def fit(self, x) -> None:
        log.info('Items to nlp: '+str(len(x)))
        self.ec_data = [dict(item, **{'title_nlped': self.tokenizer.analyze(item['Title']),
                                      'feat_nlped':self.tokenizer.analyze(item['Title']+'. '+item['Feature'])}) for item in x]

        print(self.ec_data[0])
        log.info('Data are nlped')

    def save(self, **kwargs) -> None:
        log.info(f"Saving model to {self.save_path}")
        save_pickle(self.ec_data, expand_path(self.save_path))

    def load(self, **kwargs) -> None:
        log.info(f"Loading model from {self.load_path}")
        self.ec_data = load_pickle(expand_path(self.load_path))

    def __call__(self, x, start=0, stop=5, **kwargs) -> Tuple[Dict[str, Dict[Any, Any]], List[float]]:
        response = []
        confidence = []

        for item_idx, query in enumerate(x):

            query = self.tokenizer.analyze(query)

            doc, ran = self.tokenizer.extract_money(query)
            print(doc)
            print(ran)
            return True

            score_title = []
            for item in self.ec_data:
                hyp = self.preprocess.lemmas(self.preprocess.filter_nlp_title(query))
                ref = self.preprocess.lemmas(item['title_nlped'])
                score_title.append(bleu_advanced(
                    hyp, ref, weights=(1,), penalty=False))

            score_feat = []
            for idx, item in enumerate(self.ec_data):
                hyp = self.preprocess.lemmas(self.preprocess.filter_nlp(query))
                ref = self.preprocess.lemmas(item['feat_nlped'])
                score_feat.append(bleu_advanced(
                    hyp, ref, weights=(0.3, 0.7), penalty=False))

            # score_feat = [bleu_advanced(lemmas(filter_nlp(query)), lemmas(item['feat_nlped']), weights=(0.3, 0.7), penalty=False) if score_title[idx]<1 else 1 for idx, item in enumerate(self.ec_data)]
            scores = np.mean([score_feat, score_title], axis=0).tolist()

            scores_title = [(score, len(self.ec_data[idx]['Title']))
                            for idx, score in enumerate(scores)]

            raw_scores_ar = np.array(scores_title, dtype=[
                ('x', 'float_'), ('y', 'int_')])

            results_args = np.argsort(raw_scores_ar, order=('x', 'y'))[
                ::-1].tolist()

            response.append([self.ec_data[idx] for idx in results_args[int(
                start[item_idx]):int(stop[item_idx])]])

            confidence.append(
                [scores[idx] for idx in results_args[int(start[item_idx]):int(stop[item_idx])]])

            entropies = self._entropy_subquery(scores, results_args)

        return {'items': response, 'entropy': entropies}, confidence

    def _entropy_subquery(self, scores, results_args) -> List[Tuple[float, str, List[Tuple[str, int]]]]:
        fields = ['Size', 'Brand', 'Author', 'Color', 'Genre']
        ent_fields: Dict = {}

        i = 0
        while scores[results_args[i]] > self.min_similarity:
            for field in fields:
                if field in self.ec_data[i]:
                    if field not in ent_fields:
                        ent_fields[field] = []

                    ent_fields[field].append(self.ec_data[i][field].lower())
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



# text = "I need bluetooth speaker cheaper 20 dollars"
# doc = nlp(text)
# mon = EcommerceBot._extract_money(doc)
# print(mon)