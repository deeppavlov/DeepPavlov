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

import itertools
import re
from logging import getLogger
from typing import Tuple, List, Optional, Union, Dict, Any
from collections import namedtuple, defaultdict

import numpy as np
import nltk

from deeppavlov.core.common.registry import register
from deeppavlov.models.kbqa.wiki_parser import WikiParser
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.rel_ranking_bert_infer import RelRankerBertInfer
from deeppavlov.models.kbqa.utils import \
    extract_year, extract_number, order_of_answers_sorting, make_combs, fill_query
from deeppavlov.models.kbqa.query_generator_base import QueryGeneratorBase

log = getLogger(__name__)


@register('query_generator')
class QueryGenerator(QueryGeneratorBase):
    """
        Class for query generation using Wikidata hdt file
    """

    def __init__(self, wiki_parser: WikiParser,
                 rel_ranker: Union[RelRankerInfer, RelRankerBertInfer],
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 7,
                 max_comb_num: int = 10000,
                 return_all_possible_answers: bool = False,
                 return_answers: bool = False, *args, **kwargs) -> None:
        """

        Args:
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            rel_ranker: component deeppavlov.models.kbqa.rel_ranking_infer
            entities_to_leave: how many entities to leave after entity linking
            rels_to_leave: how many relations to leave after relation ranking
            max_comb_num: the maximum number of combinations of candidate entities and relations
            return_all_possible_answers: whether to return all found answers
            return_answers: whether to return answers or candidate answers
            **kwargs:
        """
        self.wiki_parser = wiki_parser
        self.rel_ranker = rel_ranker
        self.entities_to_leave = entities_to_leave
        self.rels_to_leave = rels_to_leave
        self.max_comb_num = max_comb_num
        self.return_all_possible_answers = return_all_possible_answers
        self.return_answers = return_answers
        self.replace_tokens = [("wdt:p31", "wdt:P31"), ("pq:p580", "pq:P580"),
                               ("pq:p582", "pq:P582"), ("pq:p585", "pq:P585"), ("pq:p1545", "pq:P1545")]
        super().__init__(wiki_parser=self.wiki_parser, rel_ranker=self.rel_ranker,
                         entities_to_leave=self.entities_to_leave, rels_to_leave=self.rels_to_leave,
                         return_answers=self.return_answers, *args, **kwargs)

    def __call__(self, question_batch: List[str],
                 question_san_batch: List[str],
                 template_type_batch: Union[List[List[str]], List[str]],
                 entities_from_ner_batch: List[List[str]],
                 types_from_ner_batch: List[List[str]]) -> List[Union[List[Tuple[str, Any]], List[str]]]:

        candidate_outputs_batch = []
        template_answers_batch = []
        for question, question_sanitized, template_type, entities_from_ner, types_from_ner in \
                zip(question_batch, question_san_batch, template_type_batch,
                    entities_from_ner_batch, types_from_ner_batch):
            candidate_outputs, template_answer = self.find_candidate_answers(question, question_sanitized,
                                                                             template_type, entities_from_ner,
                                                                             types_from_ner)
            candidate_outputs_batch.append(candidate_outputs)
            template_answers_batch.append(template_answer)
        if self.return_answers:
            answers = self.rel_ranker(question_batch, candidate_outputs_batch, entities_from_ner_batch,
                                      template_answers_batch)
            log.debug(f"(__call__)answers: {answers}")
            if not answers:
                answers = ["Not Found"]
            return answers
        else:
            log.debug(f"(__call__)candidate_outputs_batch: {[output[:5] for output in candidate_outputs_batch]}")
            return candidate_outputs_batch

    def query_parser(self, question: str, query_info: Dict[str, str],
                     entities_and_types_select: List[str],
                     entity_ids: List[List[str]],
                     type_ids: List[List[str]],
                     rels_from_template: Optional[List[Tuple[str]]] = None) -> List[List[Union[Tuple[Any, ...], Any]]]:
        question_tokens = nltk.word_tokenize(question)
        query = query_info["query_template"].lower()
        for old_tok, new_tok in self.replace_tokens:
            query = query.replace(old_tok, new_tok)
        log.debug(f"\n_______________________________\nquery: {query}\n_______________________________\n")
        rels_for_search = query_info["rank_rels"]
        rel_types = query_info["rel_types"]
        query_seq_num = query_info["query_sequence"]
        return_if_found = query_info["return_if_found"]
        define_sorting_order = query_info["define_sorting_order"]
        property_types = query_info["property_types"]
        log.debug(f"(query_parser)query: {query}, {rels_for_search}, {query_seq_num}, {return_if_found}")
        query_triplets = re.findall("{[ ]?(.*?)[ ]?}", query)[0].split(' . ')
        log.debug(f"(query_parser)query_triplets: {query_triplets}")
        query_triplets = [triplet.split(' ')[:3] for triplet in query_triplets]
        query_sequence_dict = {num: triplet for num, triplet in zip(query_seq_num, query_triplets)}
        query_sequence = []
        for i in range(1, max(query_seq_num) + 1):
            query_sequence.append(query_sequence_dict[i])
        triplet_info_list = [("forw" if triplet[2].startswith('?') else "backw", search_source, rel_type)
                             for search_source, triplet, rel_type in zip(rels_for_search, query_triplets, rel_types) if
                             search_source != "do_not_rank"]
        log.debug(f"(query_parser)rel_directions: {triplet_info_list}")
        entity_ids = [entity[:self.entities_to_leave] for entity in entity_ids]
        if rels_from_template is not None:
            rels = [[(rel, 1.0) for rel in rel_list] for rel_list in rels_from_template]
        else:
            rels = [self.find_top_rels(question, entity_ids, triplet_info)
                    for triplet_info in triplet_info_list]
        log.debug(f"(query_parser)rels: {rels}")
        rels_from_query = [triplet[1] for triplet in query_triplets if triplet[1].startswith('?')]
        answer_ent = re.findall("select [\(]?([\S]+) ", query)
        order_info_nt = namedtuple("order_info", ["variable", "sorting_order"])
        order_variable = re.findall("order by (asc|desc)\((.*)\)", query)
        if order_variable:
            if define_sorting_order:
                answers_sorting_order = order_of_answers_sorting(question)
            else:
                answers_sorting_order = order_variable[0][0]
            order_info = order_info_nt(order_variable[0][1], answers_sorting_order)
        else:
            order_info = order_info_nt(None, None)
        log.debug(f"question, order_info: {question}, {order_info}")
        filter_from_query = re.findall("contains\((\?\w), (.+?)\)", query)
        log.debug(f"(query_parser)filter_from_query: {filter_from_query}")

        year = extract_year(question_tokens, question)
        number = extract_number(question_tokens, question)
        log.debug(f"year {year}, number {number}")
        if year:
            filter_info = [(elem[0], elem[1].replace("n", year)) for elem in filter_from_query]
        elif number:
            filter_info = [(elem[0], elem[1].replace("n", number)) for elem in filter_from_query]
        else:
            filter_info = [elem for elem in filter_from_query if elem[1] != "n"]
        for unk_prop, prop_type in property_types.items():
            filter_info.append((unk_prop, prop_type))
        log.debug(f"(query_parser)filter_from_query: {filter_from_query}")
        rel_combs = make_combs(rels, permut=False)
        import datetime
        start_time = datetime.datetime.now()
        entity_positions, type_positions = [elem.split('_') for elem in entities_and_types_select.split(' ')]
        log.debug(f"entity_positions {entity_positions}, type_positions {type_positions}")
        selected_entity_ids = [entity_ids[int(pos) - 1] for pos in entity_positions if int(pos) > 0]
        selected_type_ids = [type_ids[int(pos) - 1] for pos in type_positions if int(pos) > 0]
        entity_combs = make_combs(selected_entity_ids, permut=True)
        type_combs = make_combs(selected_type_ids, permut=False)
        log.debug(f"(query_parser)entity_combs: {entity_combs[:3]}, type_combs: {type_combs[:3]},"
                  f" rel_combs: {rel_combs[:3]}")
        queries_list = []
        parser_info_list = []
        confidences_list = []
        all_combs_list = list(itertools.product(entity_combs, type_combs, rel_combs))
        if self.wiki_file_format == "pickle":
            total_entities_list = list(itertools.chain.from_iterable(selected_entity_ids)) + \
                                  list(itertools.chain.from_iterable(selected_type_ids))
            parse_res = self.wiki_parser(["parse_triplets"], [total_entities_list])
        for comb_num, combs in enumerate(all_combs_list):
            confidence = np.prod([score for rel, score in combs[2][:-1]])
            confidences_list.append(confidence)
            query_hdt_seq = [
                fill_query(query_hdt_elem, combs[0], combs[1], combs[2]) for query_hdt_elem in query_sequence]
            if comb_num == 0:
                log.debug(f"\n__________________________\nfilled query: {query_hdt_seq}\n__________________________\n")
            queries_list.append((rels_from_query + answer_ent, query_hdt_seq, filter_info, order_info, return_if_found))
            parser_info_list.append("query_execute")
            if comb_num == self.max_comb_num:
                break

        candidate_outputs = []
        candidate_outputs_list = self.wiki_parser(parser_info_list, queries_list)
        if self.use_api_requester and isinstance(candidate_outputs_list, list) and candidate_outputs_list:
            candidate_outputs_list = candidate_outputs_list[0]

        if isinstance(candidate_outputs_list, list) and candidate_outputs_list:
            outputs_len = len(candidate_outputs_list)
            all_combs_list = all_combs_list[:outputs_len]
            confidences_list = confidences_list[:outputs_len]
            for combs, confidence, candidate_output in zip(all_combs_list, confidences_list, candidate_outputs_list):
                candidate_outputs += [[combs[0]] + [rel for rel, score in combs[2][:-1]] + output + [confidence]
                                      for output in candidate_output]
            if self.return_all_possible_answers:
                candidate_outputs_dict = defaultdict(list)
                for candidate_output in candidate_outputs:
                    candidate_outputs_dict[(tuple(candidate_output[0]),
                                            tuple(candidate_output[1:-2]))].append(candidate_output[-2:])
                candidate_outputs = []
                for (candidate_entity_comb, candidate_rel_comb), candidate_output in candidate_outputs_dict.items():
                    candidate_outputs.append(list(candidate_rel_comb) +
                                             [tuple([ans for ans, conf in candidate_output]), candidate_output[0][1]])
            else:
                candidate_outputs = [output[1:] for output in candidate_outputs]
        log.debug(f"(query_parser)loop time: {datetime.datetime.now() - start_time}")
        log.debug(f"(query_parser)final outputs: {candidate_outputs[:3]}")

        return candidate_outputs
