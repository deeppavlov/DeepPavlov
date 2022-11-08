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
from deeppavlov.core.common.file import load_pickle, read_json
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
                 softmax: bool = False,
                 return_answer_ids: bool = False,
                 use_api_requester: bool = False,
                 return_sentence_answer: bool = False,
                 rank: bool = True,
                 delete_rel_prefix: bool = True,
                 return_entities_and_rels: bool = False,
                 nll_ranking: bool = False,
                 top_possible_answers: int = -1,
                 top_n: int = 1,
                 return_confidences: bool = False, **kwargs):
        """

        Args:
            load_path: path to folder with wikidata files
            rel_q2name_filename: name of file which maps relation id to name
            ranker: component deeppavlov.models.ranking.rel_ranker
            wiki_parser: component deeppavlov.models.wiki_parser
            batch_size: infering batch size
            softmax: whether to process relation scores with softmax function
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
        self.softmax = softmax
        self.return_answer_ids = return_answer_ids
        self.use_api_requester = use_api_requester
        self.return_sentence_answer = return_sentence_answer
        self.rank = rank
        self.delete_rel_prefix = delete_rel_prefix
        self.return_entities_and_rels = return_entities_and_rels
        self.nll_ranking = nll_ranking
        self.top_possible_answers = top_possible_answers
        self.top_n = top_n
        self.return_confidences = return_confidences
        self.load()

    def load(self) -> None:
        if self.rel_q2name_filename.endswith("pickle"):
            self.rel_q2name = load_pickle(self.load_path / self.rel_q2name_filename)
        elif self.rel_q2name_filename.endswith("json"):
            self.rel_q2name = read_json(self.load_path / self.rel_q2name_filename)

    def save(self) -> None:
        pass

    def __call__(self, questions_list: List[str],
                 answers_list: List[List[Tuple[str]]],
                 entities_list: List[List[str]],
                 template_answers_list: List[str],
                 equal_flag_list: List[str] = None) -> List[str]:
        answers_batch, confidences_batch, answer_ids_batch, entities_and_rels_batch = [], [], [], []
        if equal_flag_list is None:
            equal_flag_list = ["" for _ in questions_list]
        for question, answers, entities, template_answer, equal_flag in \
                zip(questions_list, answers_list, entities_list, template_answers_list, equal_flag_list):
            answers_with_scores = []
            n_batches = len(answers) // self.batch_size + int(len(answers) % self.batch_size > 0)
            for i in range(n_batches):
                questions_batch, rels_batch, rels_labels_batch, cur_answers_batch = [], [], [], []
                entities_batch, confidences_batch = [], []
                for ans_and_rels in answers[i * self.batch_size: (i + 1) * self.batch_size]:
                    rels, rels_str, answer, entities, confidence, rels_labels = [], [], "", [], [], []
                    if ans_and_rels:
                        rels = ans_and_rels["relations"]
                        if self.delete_rel_prefix:
                            rels = [rel.split('/')[-1] for rel in rels]
                        answer = ans_and_rels["answers"]
                        entities = ans_and_rels["entities"]
                        confidence = ans_and_rels["output_conf"]
                        rels_labels = [self.rel_q2name[rel][0].lower() for rel in rels if rel in self.rel_q2name]
                    check_equal = False
                    if (rels_labels[0] == rels_labels[1] and equal_flag == "equal") or \
                            (rels_labels[0] != rels_labels[1] and equal_flag == "not_equal"):
                        check_equal = True
                    if rels_labels and ((len(rels_labels) == 2 and equal_flag and check_equal)
                                        or not equal_flag or len(rels_labels) == 1):
                        questions_batch.append(question)
                        rels_batch.append(rels)
                        rels_labels_batch.append(rels_labels)
                        cur_answers_batch.append(answer)
                        entities_batch.append(entities)
                        confidences_batch.append(confidence)
                if questions_batch:
                    if self.rank:
                        if self.nll_ranking:
                            probas = self.ranker([questions_batch[0]], [rels_labels_batch])
                            probas = probas[0]
                        else:
                            what_to_rank_batch = [self.what_to_rank for _ in questions_batch]
                            probas = self.ranker(questions_batch, rels_labels_batch, what_to_rank_batch)
                            probas = [proba[0] for proba in probas]
                    else:
                        probas = [rel_conf for rel_conf, entity_conf in confidences_batch]
                    for j, (answer, entities, (rel_conf, entity_conf), rels_ids, rels_labels) in \
                            enumerate(zip(cur_answers_batch, entities_batch, confidences_batch, rels_batch,
                                          rels_labels_batch)):
                        answers_with_scores.append(
                            (answer, entities, rels_labels, rels_ids, probas[j], entity_conf))

            answers_with_scores = sorted(answers_with_scores, key=lambda x: x[-1] * x[-2], reverse=True)

            res_answers_list, res_answer_ids_list, res_confidences_list, res_entities_and_rels_list = [], [], [], []
            for n, answers_with_scores_elem in enumerate(answers_with_scores):
                init_answer_ids, query_entities, _, query_rels, confidence, _ = answers_with_scores_elem
                answer_ids = []
                for answer_id in init_answer_ids:
                    if answer_id not in answer_ids:
                        answer_ids.append(answer_id)

                if self.top_possible_answers > 0:
                    answer_ids = answer_ids[:self.top_possible_answers]
                answer_ids_input = [(answer_id, question) for answer_id in answer_ids]
                answer_ids = [str(answer_id).split("/")[-1] for answer_id in answer_ids]
                parser_info_list = ["find_label" for _ in answer_ids_input]
                answer_labels = self.wiki_parser(parser_info_list, answer_ids_input)
                if n < 20:
                    log.debug(f"answers: {answers_with_scores_elem} --- answer_labels {answer_labels}")
                answer_labels = list(set(answer_labels))
                answer_labels = [label for label in answer_labels if (label and label != "Not Found")][:5]
                answer_labels = [str(label) for label in answer_labels]
                if len(answer_labels) > 2:
                    answer = f"{', '.join(answer_labels[:-1])} and {answer_labels[-1]}"
                else:
                    answer = ', '.join(answer_labels)

                if self.return_sentence_answer:
                    try:
                        answer = sentence_answer(question, answer, entities, template_answer)
                    except:
                        log.info("Error in sentence answer")
                res_answers_list.append(answer)
                res_answer_ids_list.append(answer_ids)
                res_confidences_list.append(confidence)
                res_entities_and_rels_list.append([query_entities[:-1], query_rels])

            if self.top_n == 1:
                if answers_with_scores:
                    answers_batch.append(res_answers_list[0])
                    confidences_batch.append(res_confidences_list[0])
                    answer_ids_batch.append(res_answer_ids_list[0])
                    entities_and_rels_batch.append(res_entities_and_rels_list[0])
                else:
                    answers_batch.append("Not Found")
                    confidences_batch.append(0.0)
                    answer_ids_batch.append([])
                    entities_and_rels_batch.append([])
            else:
                answers_batch.append(res_answers_list[:self.top_n])
                confidences_batch.append(res_confidences_list[:self.top_n])
                answer_ids_batch.append(res_answer_ids_list[:self.top_n])
                entities_and_rels_batch.append(res_entities_and_rels_list[:self.top_n])

        answer_tuple = (answers_batch,)
        if self.return_confidences:
            answer_tuple += (confidences_batch,)
        if self.return_answer_ids:
            answer_tuple += (answer_ids_batch,)
        if self.return_entities_and_rels:
            answer_tuple += (entities_and_rels_batch,)

        return answer_tuple

    def rank_rels(self, question: str, candidate_rels: List[str]) -> List[Tuple[str, Any]]:
        rels_with_scores = []
        if question is not None:
            questions, rels_labels, rels = [], [], []
            for candidate_rel in candidate_rels:
                if candidate_rel in self.rel_q2name:
                    cur_rels_labels = self.rel_q2name[candidate_rel]
                    if isinstance(cur_rels_labels, str):
                        cur_rels_labels = [cur_rels_labels]
                    for cur_rel in cur_rels_labels:
                        questions.append(question)
                        rels.append(candidate_rel)
                        rels_labels.append(cur_rel)
            if questions:
                n_batches = len(rels) // self.batch_size + int(len(rels) % self.batch_size > 0)
                for i in range(n_batches):
                    probas = self.ranker(questions[i * self.batch_size:(i + 1) * self.batch_size],
                                         rels_labels[i * self.batch_size:(i + 1) * self.batch_size])
                    probas = [proba[1] for proba in probas]
                    for j, rel in enumerate(rels[i * self.batch_size:(i + 1) * self.batch_size]):
                        rels_with_scores.append((rel, probas[j]))
            if self.softmax:
                scores = [score for rel, score in rels_with_scores]
                softmax_scores = softmax(scores)
                rels_with_scores = [(rel, softmax_score) for (rel, score), softmax_score in
                                    zip(rels_with_scores, softmax_scores)]
            rels_with_scores_dict = {}
            for rel, score in rels_with_scores:
                if rel not in rels_with_scores_dict:
                    rels_with_scores_dict[rel] = []
                rels_with_scores_dict[rel].append(score)
            rels_with_scores = [(rel, max(scores)) for rel, scores in rels_with_scores_dict.items()]
            rels_with_scores = sorted(rels_with_scores, key=lambda x: x[1], reverse=True)

        return rels_with_scores
