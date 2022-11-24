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
                 bs: int = 32,
                 softmax: bool = False,
                 return_answer_ids: bool = False,
                 use_api_requester: bool = False,
                 return_sentence_answer: bool = False,
                 rank: bool = True,
                 delete_rel_prefix: bool = True,
                 return_entities_and_rels: bool = False,
                 return_queries: bool = False,
                 nll_rel_ranking: bool = False,
                 nll_path_ranking: bool = False,
                 top_possible_answers: int = -1,
                 top_n: int = 1,
                 pos_class_num: int = 1,
                 rel_thres: float = 0.0,
                 filter_high_rel_conf_flag: bool = False,
                 type_rels: List[str] = None,
                 return_confidences: bool = False,
                 return_several_confidences: bool = False, **kwargs):
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
        self.bs = bs
        self.softmax = softmax
        self.return_answer_ids = return_answer_ids
        self.use_api_requester = use_api_requester
        self.return_sentence_answer = return_sentence_answer
        self.rank = rank
        self.delete_rel_prefix = delete_rel_prefix
        self.return_entities_and_rels = return_entities_and_rels
        self.return_queries = return_queries
        self.nll_rel_ranking = nll_rel_ranking
        self.nll_path_ranking = nll_path_ranking
        self.top_possible_answers = top_possible_answers
        self.top_n = top_n
        self.pos_class_num = pos_class_num
        self.rel_thres = rel_thres
        self.filter_high_rel_conf_flag = filter_high_rel_conf_flag
        if type_rels is None:
            self.type_rels = set()
        else:
            self.type_rels = type_rels
        self.return_confidences = return_confidences
        self.return_several_confidences = return_several_confidences
        self.equal_flag_oppos = {"equal": "not_equal", "not_equal": "equal"}
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
                 template_answers_batch: List[str],
                 equal_flag_batch: List[str] = None) -> List[str]:
        answers_batch, outp_confidences_batch, answer_ids_batch = [], [], []
        entities_and_rels_batch, queries_batch = [], []
        if equal_flag_batch is None:
            equal_flag_batch = ["" for _ in questions_batch]
        log.debug(f"equal_flag_batch {equal_flag_batch}")
        for question, template_type, raw_answers, entities, template_answer, equal_flag in \
                zip(questions_batch, template_type_batch, raw_answers_batch, entity_substr_batch,
                    template_answers_batch, equal_flag_batch):
            answers_with_scores = []
            l_questions, l_rels, l_rels_labels, l_cur_answers, l_entities, l_types, l_sparql_queries, l_confs = \
                self.preprocess_ranking_input(question, raw_answers, equal_flag)
            if not l_cur_answers and equal_flag:
                oppos_equal_flag = self.equal_flag_oppos[equal_flag]
                l_questions, l_rels, l_rels_labels, l_cur_answers, l_entities, l_types, l_sparql_queries, l_confs = \
                    self.preprocess_ranking_input(question, raw_answers, oppos_equal_flag)

            n_batches = len(l_questions) // self.bs + int(len(l_questions) % self.bs > 0)
            for i in range(n_batches):
                if self.rank:
                    if self.nll_path_ranking:
                        probas = self.ranker([l_questions[0]], [l_rels_labels[self.bs*i:self.bs*(i+1)]])
                        probas = probas[0]
                    else:
                        probas = self.ranker(l_questions[self.bs*i:self.bs*(i+1)],
                                             l_rels_labels[self.bs*i:self.bs*(i+1)])
                        probas = [proba[0] for proba in probas]
                else:
                    probas = [rel_conf for rel_conf, entity_conf in l_confs[self.bs*i:self.bs*(i+1)]]
                for j in range(self.bs*i, self.bs*(i+1)):
                    if j < len(l_cur_answers) and (probas[j - self.bs*i] > self.rel_thres or \
                            (len(l_rels[j]) > 1 and not set(l_rels[j]).intersection(self.type_rels))):
                        answers_with_scores.append((l_cur_answers[j], l_sparql_queries[j], l_entities[j], l_types[j],
                                                    l_rels_labels[j], l_rels[j], round(probas[j - self.bs*i], 3),
                                                    round(l_confs[j][0], 3), l_confs[j][1]))
            answers_with_scores = sorted(answers_with_scores, key=lambda x: x[-1] * x[-3], reverse=True)
            if template_type == "simple_boolean" and not answers_with_scores:
                answers_with_scores = [(["No"], "", [], [], [], [], 1.0, 1.0)]
            res_answers_list, res_answer_ids_list, res_confidences_list, res_entities_and_rels_list = [], [], [], []
            res_queries_list = []
            if self.filter_high_rel_conf_flag:
                answers_with_scores = self.filter_high_rel_conf(template_type, answers_with_scores)
            for n, ans_sc_elem in enumerate(answers_with_scores):
                init_answer_ids, query, q_entities, q_types, _, q_rels, p_conf, r_conf, e_conf = ans_sc_elem
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
                if n < 7:
                    log.debug(f"answers: {init_answer_ids[:3]} --- query {query} --- entities {q_entities} --- "
                              f"types {q_types[:3]} {ans_sc_elem[5:]} answer_labels {answer_labels[:3]}")
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
                if self.return_several_confidences:
                    res_confidences_list.append((p_conf, r_conf, e_conf))
                else:
                    res_confidences_list.append(p_conf)
                res_entities_and_rels_list.append([q_entities[:-1], q_rels])
                res_queries_list.append(query)

            if self.top_n == 1:
                if answers_with_scores:
                    answers_batch.append(res_answers_list[0])
                    outp_confidences_batch.append(res_confidences_list[0])
                    answer_ids_batch.append(res_answer_ids_list[0])
                    entities_and_rels_batch.append(res_entities_and_rels_list[0])
                    queries_batch.append(res_queries_list[0])
                else:
                    answers_batch.append("Not Found")
                    outp_confidences_batch.append(0.0)
                    answer_ids_batch.append([])
                    entities_and_rels_batch.append([])
                    queries_batch.append([])
            else:
                answers_batch.append(res_answers_list[:self.top_n])
                outp_confidences_batch.append(res_confidences_list[:self.top_n])
                answer_ids_batch.append(res_answer_ids_list[:self.top_n])
                entities_and_rels_batch.append(res_entities_and_rels_list[:self.top_n])
                queries_batch.append(res_queries_list[:self.top_n])

        answer_tuple = (answers_batch,)
        if self.return_confidences:
            answer_tuple += (outp_confidences_batch,)
        if self.return_answer_ids:
            answer_tuple += (answer_ids_batch,)
        if self.return_entities_and_rels:
            answer_tuple += (entities_and_rels_batch,)
        if self.return_queries:
            answer_tuple += (queries_batch,)

        return answer_tuple

    def preprocess_ranking_input(self, question, answers, equal_flag):
        l_questions, l_rels, l_rels_labels, l_cur_answers = [], [], [], []
        l_entities, l_types, l_sparql_queries, l_confs = [], [], [], []    
        for ans_and_rels in answers:
            answer, sparql_query, confidence = "", "", []
            entities, types, rels, rels_labels = [], [], [], []
            if ans_and_rels:
                rels = ans_and_rels["relations"]
                if self.delete_rel_prefix:
                    rels = [rel.split('/')[-1] for rel in rels]
                answer = ans_and_rels["answers"]
                entities = ans_and_rels["entities"]
                types = ans_and_rels["types"]
                sparql_query = ans_and_rels["sparql_query"]
                confidence = ans_and_rels["output_conf"]
                rels_labels = []
                for rel in rels:
                    if rel in self.rel_q2name:
                        label = self.rel_q2name[rel]
                        if isinstance(label, list):
                            label = label[0]
                        rels_labels.append(label.lower())
            check_equal = False
            if len(rels_labels) == 2 and ((rels_labels[0] == rels_labels[1] and equal_flag == "equal") or \
                    (rels_labels[0] != rels_labels[1] and equal_flag == "not_equal")):
                check_equal = True
            if rels_labels and ((len(rels_labels) == 2 and equal_flag and check_equal)
                                or not equal_flag or len(rels_labels) == 1):
                l_questions.append(question)
                l_rels.append(rels)
                l_rels_labels.append(rels_labels)
                l_cur_answers.append(answer)
                l_entities.append(entities)
                l_types.append(types)
                l_sparql_queries.append(sparql_query)
                l_confs.append(confidence)
        return l_questions, l_rels, l_rels_labels, l_cur_answers, l_entities, l_types, l_sparql_queries, l_confs

    def filter_high_rel_conf(self, template_type, answers_with_scores):
        for tp, thres in [("2_hop", 0.98), ("simple", 0.99)]:
            if template_type.startswith(tp):
                f_answers_with_scores = []
                for ans_sc in answers_with_scores:
                    if ans_sc[-2] > thres:
                        f_answers_with_scores.append(ans_sc)
                if f_answers_with_scores:
                    return f_answers_with_scores
        return answers_with_scores

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
                n_batches = len(rels) // self.bs + int(len(rels) % self.bs > 0)
                for i in range(n_batches):
                    if self.nll_rel_ranking:
                        probas = self.ranker([questions[0]],
                                             [rels_labels[i * self.bs:(i + 1) * self.bs]])
                        probas = probas[0]
                    else:
                        probas = self.ranker(questions[i * self.bs:(i + 1) * self.bs],
                                             rels_labels[i * self.bs:(i + 1) * self.bs])
                        probas = [proba[self.pos_class_num] for proba in probas]
                    for j, rel in enumerate(rels[i * self.bs:(i + 1) * self.bs]):
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
