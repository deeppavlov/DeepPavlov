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
                 return_elements: List[str] = None,
                 ranker: Chainer = None,
                 wiki_parser: Optional[WikiParser] = None,
                 batch_size: int = 32,
                 softmax: bool = False,
                 use_api_requester: bool = False,
                 rank: bool = True,
                 nll_rel_ranking: bool = False,
                 nll_path_ranking: bool = False,
                 top_possible_answers: int = -1,
                 top_n: int = 1,
                 pos_class_num: int = 1,
                 rel_thres: float = 0.0,
                 type_rels: List[str] = None, **kwargs):
        """

        Args:
            load_path: path to folder with wikidata files
            rel_q2name_filename: name of file which maps relation id to name
            return_elements: what elements return in output
            ranker: component deeppavlov.models.ranking.rel_ranker
            wiki_parser: component deeppavlov.models.wiki_parser
            batch_size: infering batch size
            softmax: whether to process relation scores with softmax function
            use_api_requester: whether wiki parser will be used as external api
            rank: whether to rank relations or simple copy input
            nll_rel_ranking: whether use components trained with nll loss for relation ranking
            nll_path_ranking: whether use components trained with nll loss for relation path ranking
            top_possible_answers: number of answers returned for a question in each list of candidate answers
            top_n: number of lists of candidate answers returned for a question
            pos_class_num: index of positive class in the output of relation ranking model
            rel_thres: threshold of relation confidence
            type_rels: list of relations in the knowledge base which connect an entity and its type 
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.rel_q2name_filename = rel_q2name_filename
        self.ranker = ranker
        self.wiki_parser = wiki_parser
        self.batch_size = batch_size
        self.softmax = softmax
        self.return_elements = return_elements or list()
        self.use_api_requester = use_api_requester
        self.rank = rank
        self.nll_rel_ranking = nll_rel_ranking
        self.nll_path_ranking = nll_path_ranking
        self.top_possible_answers = top_possible_answers
        self.top_n = top_n
        self.pos_class_num = pos_class_num
        self.rel_thres = rel_thres
        self.type_rels = type_rels or set()
        self.load()

    def load(self) -> None:
        if self.rel_q2name_filename.endswith("pickle"):
            self.rel_q2name = load_pickle(self.load_path / self.rel_q2name_filename)
        elif self.rel_q2name_filename.endswith("json"):
            self.rel_q2name = read_json(self.load_path / self.rel_q2name_filename)

    def save(self) -> None:
        pass

    def __call__(self, questions_batch: List[str],
                 template_type_batch: List[str],
                 raw_answers_batch: List[List[Tuple[str]]],
                 entity_substr_batch: List[List[str]],
                 template_answers_batch: List[str]) -> List[str]:
        answers_batch, outp_confidences_batch, answer_ids_batch = [], [], []
        entities_and_rels_batch, queries_batch, triplets_batch = [], [], []
        for question, template_type, raw_answers, entities, template_answer in \
                zip(questions_batch, template_type_batch, raw_answers_batch, entity_substr_batch,
                    template_answers_batch):
            answers_with_scores = []
            l_questions, l_rels, l_rels_labels, l_cur_answers, l_entities, l_types, l_sparql_queries, l_triplets, \
            l_confs = self.preprocess_ranking_input(question, raw_answers)

            n_batches = len(l_questions) // self.batch_size + int(len(l_questions) % self.batch_size > 0)
            for i in range(n_batches):
                if self.rank:
                    if self.nll_path_ranking:
                        probas = self.ranker([l_questions[0]],
                                             [l_rels_labels[self.batch_size * i:self.batch_size * (i + 1)]])
                        probas = probas[0]
                    else:
                        probas = self.ranker(l_questions[self.batch_size * i:self.batch_size * (i + 1)],
                                             l_rels_labels[self.batch_size * i:self.batch_size * (i + 1)])
                        probas = [proba[0] for proba in probas]
                else:
                    probas = [rel_conf for rel_conf, entity_conf in
                              l_confs[self.batch_size * i:self.batch_size * (i + 1)]]
                for j in range(self.batch_size * i, self.batch_size * (i + 1)):
                    if j < len(l_cur_answers) and (probas[j - self.batch_size * i] > self.rel_thres or
                                                   (len(l_rels[j]) > 1 and not set(l_rels[j]).intersection(
                                                       self.type_rels))):
                        answers_with_scores.append((l_cur_answers[j], l_sparql_queries[j], l_triplets[j],
                                                    l_entities[j], l_types[j], l_rels_labels[j], l_rels[j],
                                                    round(probas[j - self.batch_size * i], 3),
                                                    round(l_confs[j][0], 3), l_confs[j][1]))
            answers_with_scores = sorted(answers_with_scores, key=lambda x: x[-1] * x[-3], reverse=True)
            if template_type == "simple_boolean" and not answers_with_scores:
                answers_with_scores = [(["No"], "", [], [], [], [], [], 1.0, 1.0, 1.0)]
            res_answers_list, res_answer_ids_list, res_confidences_list, res_entities_and_rels_list = [], [], [], []
            res_queries_list, res_triplets_list = [], []
            for n, ans_sc_elem in enumerate(answers_with_scores):
                init_answer_ids, query, triplets, q_entities, q_types, _, q_rels, p_conf, r_conf, e_conf = ans_sc_elem
                answer_ids = []
                for answer_id in init_answer_ids:
                    answer_id = str(answer_id).replace("@en", "").strip('"')
                    if answer_id not in answer_ids:
                        answer_ids.append(answer_id)

                if self.top_possible_answers > 0:
                    answer_ids = answer_ids[:self.top_possible_answers]
                answer_ids_input = [(answer_id, question) for answer_id in answer_ids]
                answer_ids = [str(answer_id).split("/")[-1] for answer_id in answer_ids]
                parser_info_list = ["find_label" for _ in answer_ids_input]
                init_answer_labels = self.wiki_parser(parser_info_list, answer_ids_input)
                if n < 7:
                    log.debug(f"answers: {init_answer_ids[:3]} --- query {query} --- entities {q_entities} --- "
                              f"types {q_types[:3]} --- q_rels {q_rels} --- {ans_sc_elem[5:]} --- "
                              f"answer_labels {init_answer_labels[:3]}")
                answer_labels = []
                for label in init_answer_labels:
                    if label not in answer_labels:
                        answer_labels.append(label)
                answer_labels = [label for label in answer_labels if (label and label != "Not Found")][:5]
                answer_labels = [str(label) for label in answer_labels]
                if len(answer_labels) > 2:
                    answer = f"{', '.join(answer_labels[:-1])} and {answer_labels[-1]}"
                else:
                    answer = ', '.join(answer_labels)

                if "sentence_answer" in self.return_elements:
                    try:
                        answer = sentence_answer(question, answer, entities, template_answer)
                    except ValueError as e:
                        log.warning(f"Error in sentence answer, {e}")

                res_answers_list.append(answer)
                res_answer_ids_list.append(answer_ids)
                if "several_confidences" in self.return_elements:
                    res_confidences_list.append((p_conf, r_conf, e_conf))
                else:
                    res_confidences_list.append(p_conf)
                res_entities_and_rels_list.append([q_entities[:-1], q_rels])
                res_queries_list.append(query)
                res_triplets_list.append(triplets)

            if self.top_n == 1:
                if answers_with_scores:
                    answers_batch.append(res_answers_list[0])
                    outp_confidences_batch.append(res_confidences_list[0])
                    answer_ids_batch.append(res_answer_ids_list[0])
                    entities_and_rels_batch.append(res_entities_and_rels_list[0])
                    queries_batch.append(res_queries_list[0])
                    triplets_batch.append(res_triplets_list[0])
                else:
                    answers_batch.append("Not Found")
                    outp_confidences_batch.append(0.0)
                    answer_ids_batch.append([])
                    entities_and_rels_batch.append([])
                    queries_batch.append([])
                    triplets_batch.append([])
            else:
                answers_batch.append(res_answers_list[:self.top_n])
                outp_confidences_batch.append(res_confidences_list[:self.top_n])
                answer_ids_batch.append(res_answer_ids_list[:self.top_n])
                entities_and_rels_batch.append(res_entities_and_rels_list[:self.top_n])
                queries_batch.append(res_queries_list[:self.top_n])
                triplets_batch.append(res_triplets_list[:self.top_n])

        answer_tuple = (answers_batch,)
        if "confidences" in self.return_elements:
            answer_tuple += (outp_confidences_batch,)
        if "answer_ids" in self.return_elements:
            answer_tuple += (answer_ids_batch,)
        if "entities_and_rels" in self.return_elements:
            answer_tuple += (entities_and_rels_batch,)
        if "queries" in self.return_elements:
            answer_tuple += (queries_batch,)
        if "triplets" in self.return_elements:
            answer_tuple += (triplets_batch,)

        return answer_tuple

    def preprocess_ranking_input(self, question, answers):
        l_questions, l_rels, l_rels_labels, l_cur_answers = [], [], [], []
        l_entities, l_types, l_sparql_queries, l_triplets, l_confs = [], [], [], [], []
        for ans_and_rels in answers:
            answer, sparql_query, confidence = "", "", []
            entities, types, rels, rels_labels, triplets = [], [], [], [], []
            if ans_and_rels:
                rels = [rel.split('/')[-1] for rel in ans_and_rels["relations"]]
                answer = ans_and_rels["answers"]
                entities = ans_and_rels["entities"]
                types = ans_and_rels["types"]
                sparql_query = ans_and_rels["sparql_query"]
                triplets = ans_and_rels["triplets"]
                confidence = ans_and_rels["output_conf"]
                rels_labels = []
                for rel in rels:
                    if rel in self.rel_q2name:
                        label = self.rel_q2name[rel]
                        if isinstance(label, list):
                            label = label[0]
                        rels_labels.append(label.lower())
            if rels_labels:
                l_questions.append(question)
                l_rels.append(rels)
                l_rels_labels.append(rels_labels)
                l_cur_answers.append(answer)
                l_entities.append(entities)
                l_types.append(types)
                l_sparql_queries.append(sparql_query)
                l_triplets.append(triplets)
                l_confs.append(confidence)
        return l_questions, l_rels, l_rels_labels, l_cur_answers, l_entities, l_types, l_sparql_queries, l_triplets, \
               l_confs

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
                    if self.nll_rel_ranking:
                        probas = self.ranker([questions[0]],
                                             [rels_labels[i * self.batch_size:(i + 1) * self.batch_size]])
                        probas = probas[0]
                    else:
                        probas = self.ranker(questions[i * self.batch_size:(i + 1) * self.batch_size],
                                             rels_labels[i * self.batch_size:(i + 1) * self.batch_size])
                        probas = [proba[self.pos_class_num] for proba in probas]
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
