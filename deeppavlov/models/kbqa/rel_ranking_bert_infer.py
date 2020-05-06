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
from typing import Tuple, List, Any

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import load_pickle
from deeppavlov.models.ranking.rel_ranker import RelRanker
from deeppavlov.models.kbqa.wiki_parser import WikiParser

log = getLogger(__name__)


@register('rel_ranking_bert_infer')
class RelRankerBertInfer(Component, Serializable):
    """Class for ranking of paths in subgraph"""

    def __init__(self, load_path: str,
                 rel_q2name_filename: str,
                 wiki_parser: WikiParser,
                 ranker: RelRanker,
                 batch_size: int = 32,
                 rels_to_leave: int = 40, **kwargs):
        """

        Args:
            load_path: path to folder with wikidata files
            rel_q2name_filename: name of file which maps relation id to name
            wiki_parser: component deeppavlov.models.wiki_parser
            ranker: component deeppavlov.models.ranking.rel_ranker
            batch_size: infering batch size
            rels_to_leave: how many relations to leave after relation ranking
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.rel_q2name_filename = rel_q2name_filename
        self.ranker = ranker
        self.wiki_parser = wiki_parser
        self.batch_size = batch_size
        self.rels_to_leave = rels_to_leave
        self.load()

    def load(self) -> None:
        self.rel_q2name = load_pickle(self.load_path / self.rel_q2name_filename)

    def save(self) -> None:
        pass

    def __call__(self, questions_list: List[str], candidate_answers_list: List[List[Tuple[str]]]) -> List[str]:
        log.debug(f"candidate_answers_list: {candidate_answers_list}") 
        answers = []
        for question, candidate_answers in zip(questions_list, candidate_answers_list):
            answers_with_scores = []
            if len(candidate_answers) == 0:
                answer = "Not Found"

            for i in range(len(candidate_answers) // self.batch_size):
                questions_batch = []
                rels_labels_batch = []
                answers_batch = []
                for j in range(self.batch_size):
                    candidate_rels = candidate_answers[(i * self.batch_size + j)][:-1]
                    candidate_rels = [candidate_rel.split('/')[-1] for candidate_rel in candidate_rels]
                    candidate_answer = candidate_answers[(i * self.batch_size + j)][-1]
                    candidate_rels = " [SEP] ".join([self.rel_q2name[candidate_rel] \
                                                     for candidate_rel in candidate_rels if
                                                     candidate_rel in self.rel_q2name])

                    if candidate_rels:
                        questions_batch.append(question)
                        rels_labels_batch.append(candidate_rels)
                        answers_batch.append(candidate_answer)

                probas = self.ranker(questions_batch, rels_labels_batch)
                probas = [proba[1] for proba in probas]
                for j, answer in enumerate(answers_batch):
                    answers_with_scores.append((answer, probas[j]))

            questions_batch = []
            rels_labels_batch = []
            answers_batch = []
            for j in range(len(candidate_answers) % self.batch_size):
                candidate_rels = candidate_answers[(len(candidate_answers) // self.batch_size \
                              * self.batch_size + j)][:-1]
                candidate_rels = [candidate_rel.split('/')[-1] for candidate_rel in candidate_rels]
                candidate_answer = candidate_answers[(len(candidate_answers) // self.batch_size \
                              * self.batch_size + j)][-1]
                candidate_rels = " [SEP] ".join([self.rel_q2name[candidate_rel] \
                              for candidate_rel in candidate_rels if candidate_rel in self.rel_q2name])

                if candidate_rels:
                    questions_batch.append(question)
                    rels_labels_batch.append(candidate_rels)
                    answers_batch.append(candidate_answer)

            if questions_batch:
                probas = self.ranker(questions_batch, rels_labels_batch)
                probas = [proba[1] for proba in probas]
                for j, answer in enumerate(answers_batch):
                    answers_with_scores.append((answer, probas[j]))

            answers_with_scores = sorted(answers_with_scores, key=lambda x: x[1], reverse=True)

            if answers_with_scores:
                log.debug(f"answers: {answers_with_scores[0][0]}")
                answer = self.wiki_parser("objects", "forw", answers_with_scores[0][0], find_label=True)
            
            answers.append(answer)

        return answers

    def rank_rels(self, question: str, candidate_rels: List[str]) -> List[Tuple[str, Any]]:
        rels_with_scores = []
        for i in range(len(candidate_rels) // self.batch_size):
            questions_batch = []
            rels_labels_batch = []
            rels_batch = []
            for j in range(self.batch_size):
                candidate_rel = candidate_rels[(i * self.batch_size + j)]
                if candidate_rel in self.rel_q2name:
                    questions_batch.append(question)
                    rels_batch.append(candidate_rel)
                    rels_labels_batch.append(self.rel_q2name[candidate_rel])
            probas = self.ranker(questions_batch, rels_labels_batch)
            probas = [proba[1] for proba in probas]
            for j, rel in enumerate(rels_batch):
                rels_with_scores.append((rel, probas[j]))
        
        questions_batch = []
        rels_batch = []
        rels_labels_batch = []
        for j in range(len(candidate_rels) % self.batch_size):
            candidate_rel = candidate_rels[(len(candidate_rels) // self.batch_size * self.batch_size + j)]
            if candidate_rel in self.rel_q2name:
                questions_batch.append(question)
                rels_batch.append(candidate_rel)
                rels_labels_batch.append(self.rel_q2name[candidate_rel])

        probas = self.ranker(questions_batch, rels_labels_batch)
        probas = [proba[1] for proba in probas]
        for j, rel in enumerate(rels_batch):
            rels_with_scores.append((rel, probas[j]))
        rels_with_scores = sorted(rels_with_scores, key=lambda x: x[1], reverse=True)

        return rels_with_scores[:self.rels_to_leave]
