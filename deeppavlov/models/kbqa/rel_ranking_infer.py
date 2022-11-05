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

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.kbqa.sentence_answer import sentence_answer
from deeppavlov.models.kbqa.wiki_parser import WikiParser

log = getLogger(__name__)


@register('rel_ranking_infer')
class RelRankerInfer(Component, Serializable):
    """Class for ranking of paths in subgraph"""

    def __init__(self, load_path: str,
                 rel_q2name_filename: str,
                 ranker: Chainer = None,
                 wiki_parser: Optional[WikiParser] = None,
                 batch_size: int = 32,
                 rels_to_leave: int = 40,
                 softmax: bool = False,
                 return_all_possible_answers: bool = False,
                 return_answer_ids: bool = False,
                 use_api_requester: bool = False,
                 return_sentence_answer: bool = False,
                 rank: bool = True,
                 return_confidences: bool = False, **kwargs):
        """

        Args:
            load_path: path to folder with wikidata files
            rel_q2name_filename: name of file which maps relation id to name
            ranker: component deeppavlov.models.ranking.rel_ranker
            wiki_parser: component deeppavlov.models.wiki_parser
            batch_size: infering batch size
            rels_to_leave: how many relations to leave after relation ranking
            softmax: whether to process relation scores with softmax function
            return_all_possible_answers: whether to return all found answers
            return_answer_ids: whether to return answer ids from Wikidata
            use_api_requester: whether wiki parser will be used as external api
            return_sentence_answer: whether to return answer as a sentence
            rank: whether to rank relations or simple copy input
            return_confidences: whether to return confidences of candidate answers
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.rel_q2name_filename = rel_q2name_filename
        self.ranker = ranker
        self.wiki_parser = wiki_parser
        self.batch_size = batch_size
        self.rels_to_leave = rels_to_leave
        self.softmax = softmax
        self.return_all_possible_answers = return_all_possible_answers
        self.return_answer_ids = return_answer_ids
        self.use_api_requester = use_api_requester
        self.return_sentence_answer = return_sentence_answer
        self.rank = rank
        self.return_confidences = return_confidences
        self.load()

    def load(self) -> None:
        self.rel_q2name = load_pickle(self.load_path / self.rel_q2name_filename)

    def save(self) -> None:
        pass

    def __call__(self, questions_list: List[str],
                 candidate_answers_list: List[List[Tuple[str]]],
                 entities_list: List[List[str]] = None,
                 template_answers_list: List[str] = None) -> List[str]:
        answers = []
        confidence = 0.0
        if entities_list is None:
            entities_list = [[] for _ in questions_list]
        if template_answers_list is None:
            template_answers_list = ["" for _ in questions_list]
        for question, candidate_answers, entities, template_answer in \
                zip(questions_list, candidate_answers_list, entities_list, template_answers_list):
            answers_with_scores = []
            answer = "Not Found"
            if self.rank:
                n_batches = len(candidate_answers) // self.batch_size + int(
                    len(candidate_answers) % self.batch_size > 0)
                for i in range(n_batches):
                    questions_batch = []
                    rels_batch = []
                    rels_labels_batch = []
                    answers_batch = []
                    entities_batch = []
                    confidences_batch = []
                    for candidate_ans_and_rels in candidate_answers[i * self.batch_size: (i + 1) * self.batch_size]:
                        candidate_rels = []
                        candidate_rels_str, candidate_answer = "", ""
                        candidate_entities, candidate_confidence = [], []
                        if candidate_ans_and_rels:
                            candidate_rels = candidate_ans_and_rels["relations"]
                            candidate_rels = [candidate_rel.split('/')[-1] for candidate_rel in candidate_rels]
                            candidate_answer = candidate_ans_and_rels["answers"]
                            candidate_entities = candidate_ans_and_rels["entities"]
                            candidate_confidence = candidate_ans_and_rels["rel_conf"]
                            candidate_rels_str = " # ".join([self.rel_q2name[candidate_rel] \
                                                             for candidate_rel in candidate_rels if
                                                             candidate_rel in self.rel_q2name])
                        if candidate_rels_str:
                            questions_batch.append(question)
                            rels_batch.append(candidate_rels)
                            rels_labels_batch.append(candidate_rels_str)
                            answers_batch.append(candidate_answer)
                            entities_batch.append(candidate_entities)
                            confidences_batch.append(candidate_confidence)

                    if questions_batch:
                        probas = self.ranker(questions_batch, rels_labels_batch)
                        probas = [proba[1] for proba in probas]
                        for j, (answer, entities, confidence, rels_ids, rels_labels) in \
                                enumerate(zip(answers_batch, entities_batch, confidences_batch, rels_batch,
                                              rels_labels_batch)):
                            answers_with_scores.append(
                                (answer, entities, rels_labels, rels_ids, max(probas[j], confidence)))

                answers_with_scores = sorted(answers_with_scores, key=lambda x: x[-1], reverse=True)
            else:
                answers_with_scores = [(answer, rels, conf) for *rels, answer, conf in candidate_answers]

            answer_ids = tuple()
            if answers_with_scores:
                log.debug(f"answers: {answers_with_scores[0]}")
                answer_ids = answers_with_scores[0][0]
                if self.return_all_possible_answers and isinstance(answer_ids, tuple):
                    answer_ids_input = [(answer_id, question) for answer_id in answer_ids]
                    answer_ids = list(map(lambda x: x.split("/")[-1] if str(x).startswith("http") else x, answer_ids))
                else:
                    answer_ids_input = [(answer_ids, question)]
                    if str(answer_ids).startswith("http:"):
                        answer_ids = answer_ids.split("/")[-1]
                parser_info_list = ["find_label" for _ in answer_ids_input]
                answer_labels = self.wiki_parser(parser_info_list, answer_ids_input)
                log.debug(f"answer_labels {answer_labels}")
                if self.return_all_possible_answers:
                    answer_labels = list(set(answer_labels))
                    answer_labels = [label for label in answer_labels if (label and label != "Not Found")][:5]
                    answer_labels = [str(label) for label in answer_labels]
                    if len(answer_labels) > 2:
                        answer = f"{', '.join(answer_labels[:-1])} and {answer_labels[-1]}"
                    else:
                        answer = ', '.join(answer_labels)
                else:
                    answer = answer_labels[0]
                if self.return_sentence_answer:
                    try:
                        answer = sentence_answer(question, answer, entities, template_answer)
                    except:
                        log.info("Error in sentence answer")
                confidence = answers_with_scores[0][2]
            if self.return_confidences:
                answers.append((answer, confidence))
            else:
                if self.return_answer_ids:
                    if not answer_ids:
                        answer_ids = "Not found"
                    answers.append((answer, answer_ids))
                else:
                    answers.append(answer)
        if not answers:
            if self.return_confidences:
                answers.append(("Not found", 0.0))
            else:
                answers.append("Not found")

        return answers

    def rank_rels(self, question: str, candidate_rels: List[str]) -> List[Tuple[str, Any]]:
        rels_with_scores = []
        if question is not None:
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
            if self.softmax:
                scores = [score for rel, score in rels_with_scores]
                softmax_scores = softmax(scores)
                rels_with_scores = [(rel, softmax_score) for (rel, score), softmax_score in
                                    zip(rels_with_scores, softmax_scores)]
            rels_with_scores = sorted(rels_with_scores, key=lambda x: x[1], reverse=True)

        return rels_with_scores[:self.rels_to_leave]
