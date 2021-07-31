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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import load_pickle
from deeppavlov.models.ranking.rel_ranker import RelRanker
from deeppavlov.models.kbqa.wiki_parser import WikiParser
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor
from deeppavlov.models.kbqa.sentence_answer import sentence_answer

log = getLogger(__name__)


@register('rel_ranking_bert_infer')
class RelRankerBertInfer(Component, Serializable):
    """Class for ranking of paths in subgraph"""

    def __init__(self, load_path: str,
                 rel_q2name_filename: str,
                 ranker: RelRanker,
                 bert_preprocessor: Optional[BertPreprocessor] = None,
                 wiki_parser: Optional[WikiParser] = None,
                 batch_size: int = 32,
                 rels_to_leave: int = 40,
                 return_all_possible_answers: bool = False,
                 return_answer_ids: bool = False,
                 use_api_requester: bool = False,
                 use_mt_bert: bool = False,
                 return_sentence_answer: bool = False,
                 return_confidences: bool = False, **kwargs):
        """

        Args:
            load_path: path to folder with wikidata files
            rel_q2name_filename: name of file which maps relation id to name
            ranker: component deeppavlov.models.ranking.rel_ranker
            bert_perprocessor: component deeppavlov.models.preprocessors.bert_preprocessor
            wiki_parser: component deeppavlov.models.wiki_parser
            batch_size: infering batch size
            rels_to_leave: how many relations to leave after relation ranking
            return_all_possible_answers: whether to return all found answers
            return_answer_ids: whether to return answer ids from Wikidata
            use_api_requester: whether wiki parser will be used as external api
            use_mt_bert: whether nultitask bert is used for ranking
            return_sentence_answer: whether to return answer as a sentence
            return_confidences: whether to return confidences of candidate answers
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.rel_q2name_filename = rel_q2name_filename
        self.ranker = ranker
        self.bert_preprocessor = bert_preprocessor
        self.wiki_parser = wiki_parser
        self.batch_size = batch_size
        self.rels_to_leave = rels_to_leave
        self.return_all_possible_answers = return_all_possible_answers
        self.return_answer_ids = return_answer_ids
        self.use_api_requester = use_api_requester
        self.use_mt_bert = use_mt_bert
        self.return_sentence_answer = return_sentence_answer
        self.return_confidences = return_confidences
        self.load()

    def load(self) -> None:
        self.rel_q2name = load_pickle(self.load_path / self.rel_q2name_filename)

    def save(self) -> None:
        pass

    def __call__(self, questions_list: List[str], candidate_answers_list: List[List[Tuple[str]]],
                 entities_list: List[List[str]] = None, template_answers_list: List[str] = None) -> List[str]:
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

            n_batches = len(candidate_answers) // self.batch_size + int(len(candidate_answers) % self.batch_size > 0)
            for i in range(n_batches):
                questions_batch = []
                rels_labels_batch = []
                answers_batch = []
                confidences_batch = []
                for candidate_ans_and_rels in candidate_answers[i * self.batch_size: (i + 1) * self.batch_size]:
                    candidate_rels = candidate_ans_and_rels[:-2]
                    candidate_rels = [candidate_rel.split('/')[-1] for candidate_rel in candidate_rels]
                    candidate_answer = candidate_ans_and_rels[-2]
                    candidate_confidence = candidate_ans_and_rels[-1]
                    candidate_rels = " # ".join([self.rel_q2name[candidate_rel] \
                                                 for candidate_rel in candidate_rels if
                                                 candidate_rel in self.rel_q2name])

                    if candidate_rels:
                        questions_batch.append(question)
                        rels_labels_batch.append(candidate_rels)
                        answers_batch.append(candidate_answer)
                        confidences_batch.append(candidate_confidence)

                if self.use_mt_bert:
                    features = self.bert_preprocessor(questions_batch, rels_labels_batch)
                    probas = self.ranker(features)
                else:
                    probas = self.ranker(questions_batch, rels_labels_batch)
                probas = [proba[1] for proba in probas]
                for j, (answer, confidence, rels_labels) in \
                        enumerate(zip(answers_batch, confidences_batch, rels_labels_batch)):
                    answers_with_scores.append((answer, rels_labels, max(probas[j], confidence)))

            answers_with_scores = sorted(answers_with_scores, key=lambda x: x[-1], reverse=True)

            if answers_with_scores:
                log.debug(f"answers: {answers_with_scores[0]}")
                answer_ids = answers_with_scores[0][0]
                if self.return_all_possible_answers and isinstance(answer_ids, tuple):
                    answer_ids_input = [(answer_id, question) for answer_id in answer_ids]
                else:
                    answer_ids_input = [(answer_ids, question)]
                parser_info_list = ["find_label" for _ in answer_ids_input]
                answer_labels = self.wiki_parser(parser_info_list, answer_ids_input)
                if self.use_api_requester:
                    answer_labels = [label[0] for label in answer_labels]
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
                    answer = sentence_answer(question, answer, entities, template_answer)
                confidence = answers_with_scores[0][2]

            if self.return_confidences:
                answers.append((answer, confidence))
            else:
                if self.return_answer_ids:
                    answers.append((answer, answer_ids))
                else:
                    answers.append(answer)

        return answers

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
                if self.use_mt_bert:
                    features = self.bert_preprocessor(questions_batch, rels_labels_batch)
                    probas = self.ranker(features)
                else:
                    probas = self.ranker(questions_batch, rels_labels_batch)
                probas = [proba[1] for proba in probas]
                for j, rel in enumerate(rels_batch):
                    rels_with_scores.append((rel, probas[j]))
        rels_with_scores = sorted(rels_with_scores, key=lambda x: x[1], reverse=True)

        return rels_with_scores[:self.rels_to_leave]
