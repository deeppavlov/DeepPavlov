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
from typing import Tuple, List, Any, Optional
from scipy.special import softmax

from nltk import sent_tokenize

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import load_pickle
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor

log = getLogger(__name__)


@register('fact_ranking_infer')
class FactRankerInfer(Component):
    """Class for ranking of paths in subgraph"""

    def __init__(self, ranker = None,
                 batch_size: int = 32,
                 facts_to_leave: int = 3, **kwargs):
        """

        Args:
            load_path: path to folder with wikidata files
            rel_q2name_filename: name of file which maps relation id to name
            ranker: component deeppavlov.models.ranking.rel_ranker
            bert_perprocessor: component deeppavlov.models.preprocessors.bert_preprocessor
            batch_size: infering batch size
            **kwargs:
        """
        self.ranker = ranker
        self.batch_size = batch_size
        self.facts_to_leave = facts_to_leave
        self.load()

    def __call__(self, dialog_history_list: List[str], paragraphs_list: List[List[str]],
                       topical_chat_facts_list: List[List[str]], first_paragraphs_list: List[List[str]]) -> List[str]:
        facts_with_scores_batch = []
        for dialog_history, paragraphs, topical_chat_facts, first_paragraphs in \
            zip(dialog_history_list, paragraphs_list, topical_chat_facts_list, first_paragraphs_list):
            cand_facts = []
            for paragraph in paragraphs + topical_chat_facts + first_paragraphs:
                sentences = sent_tokenize(paragraph)
                cand_facts.extend([sentence for sentence in sentences if len(sentence.split())<150])
            
            facts_with_scores = []
            n_batches = len(cand_facts) // self.batch_size + int(len(cand_facts) % self.batch_size > 0)
            for i in range(n_batches):
                dh_batch = []
                facts_batch = []
                for candidate_fact in candidate_facts[i * self.batch_size: (i + 1) * self.batch_size]:
                    dh_batch.append(dialog_history)
                    facts_batch.append(candidate_fact)
                        
                if dh_batch:
                    probas = self.ranker(dh_batch, facts_batch)
                    probas = [proba[1] for proba in probas]
                    for j, fact in enumerate(facts_batch):
                        facts_with_scores.append((fact, probas[j]))
                        
            facts_with_scores = sorted(facts_with_scores, key=lambda x: x[1], reverse=True)
            facts_with_scores_batch.append(facts_with_scores[:self.facts_to_leave])

        return facts_with_scores_batch
