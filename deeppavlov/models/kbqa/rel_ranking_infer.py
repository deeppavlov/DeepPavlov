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

from typing import Tuple, List, Any

from scipy.special import softmax

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import load_pickle
from deeppavlov.models.ranking.rel_ranker import RelRanker


@register('rel_ranking_infer')
class RelRankerInfer(Component, Serializable):
    """This class performs ranking of candidate relations"""

    def __init__(self, load_path: str,
                 rel_q2name_filename: str,
                 ranker: RelRanker,
                 rels_to_leave: int = 15,
                 batch_size: int = 100, **kwargs):

        """

        Args:
            load_path: path to folder with wikidata files
            rel_q2name_filename: name of file which maps relation id to name
            ranker: deeppavlov.models.ranking.rel_ranker
            rels_to_leave: how many top scored relations leave
            batch_size: infering batch size
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.rel_q2name_filename = rel_q2name_filename
        self.ranker = ranker
        self.rels_to_leave = rels_to_leave
        self.batch_size = batch_size
        self.load()

    def load(self) -> None:
        self.rel_q2name = load_pickle(self.load_path / self.rel_q2name_filename)

    def save(self) -> None:
        pass

    def __call__(self, question_batch: List[str], candidate_rels_batch: List[List[str]]) -> \
                                                                        List[List[Tuple[str, Any]]]:
        rels_with_scores_batch = []
        for question, candidate_rels in zip(question_batch, candidate_rels_batch):
            rels_with_scores_batch.append(self.rank_rels(question, candidate_rels))
        return rels_with_scores_batch

    def rank_rels(self, question: str, candidate_rels: List[str]) -> List[Tuple[str, Any]]:
        rels_with_scores = []
        n_batches = len(candidate_rels) // self.batch_size + int(len(candidate_rels) % self.batch_size > 0)
        for i in range(n_batches):
            questions_batch = []
            rels_labels_batch = []
            rels_batch = []
            for candidate_rel in candidate_rels[i * self.batch_size: (i + 1) * self.batch_size]:
                if candidate_rel in self.rel_q2name:
                    questions_batch.append(question)
                    rels_batch.append(candidate_rel)
                    rels_labels_batch.append(self.rel_q2name[candidate_rel])
            if questions_batch:
                probas = self.ranker(questions_batch, rels_labels_batch)
                probas = [proba[1] for proba in probas]
                for j, rel in enumerate(rels_batch):
                    rels_with_scores.append((rel, probas[j]))
        scores = [score for rel, score in rels_with_scores]
        if scores:
            softmax_scores = softmax(scores)
            rels_with_scores = [(rel, softmax_score) for (rel, score), softmax_score in 
                                                              zip(rels_with_scores, softmax_scores)]
            rels_with_scores = sorted(rels_with_scores, key=lambda x: x[1], reverse=True)

        return rels_with_scores[:self.rels_to_leave]
