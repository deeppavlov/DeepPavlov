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

import copy
import itertools
import re
from collections import defaultdict
from logging import getLogger
from typing import Tuple, List, Optional, Union, Dict, Any, Set

import nltk
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.kbqa.query_generator_base import QueryGeneratorBase
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.utils import extract_year, extract_number, make_combs, fill_query, find_query_features, \
    make_sparql_query, merge_sparql_query
from deeppavlov.models.kbqa.wiki_parser import WikiParser

log = getLogger(__name__)


@register('query_generator')
class QueryGenerator(QueryGeneratorBase):
    """
        Class for query generation using Wikidata hdt file
    """

    def __init__(self, wiki_parser: WikiParser,
                 rel_ranker: RelRankerInfer,
                 entities_to_leave: int = 5,
                 types_to_leave: int = 2,
                 rels_to_leave: int = 7,
                 max_comb_num: int = 10000,
                 gold_query_info: Dict[str, str] = None,
                 map_query_str_to_kb: List[Tuple[str, str]] = None,
                 return_answers: bool = True, *args, **kwargs) -> None:
        """

        Args:
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            rel_ranker: component deeppavlov.models.kbqa.rel_ranking_infer
            entities_to_leave: how many entities to leave after entity linking
            types_to_leave: how many types to leave after entity linking
            rels_to_leave: how many relations to leave after relation ranking
            max_comb_num: the maximum number of combinations of candidate entities and relations
            gold_query_info: dict of variable names used for formatting output sparql queries
            map_query_str_to_kb: mapping of knowledge base prefixes to full https
            return_answers: whether to return answers or candidate relations and answers for further ranking
            **kwargs:
        """
        self.wiki_parser = wiki_parser
        self.rel_ranker = rel_ranker
        self.entities_to_leave = entities_to_leave
        self.types_to_leave = types_to_leave
        self.rels_to_leave = rels_to_leave
        self.max_comb_num = max_comb_num
        self.gold_query_info = gold_query_info
        self.map_query_str_to_kb = map_query_str_to_kb
        self.return_answers = return_answers
        self.replace_tokens = [("wdt:p", "wdt:P"), ("pq:p", "pq:P")]
        super().__init__(wiki_parser=self.wiki_parser, rel_ranker=self.rel_ranker,
                         entities_to_leave=self.entities_to_leave, rels_to_leave=self.rels_to_leave,
                         *args, **kwargs)

    def __call__(self, question_batch: List[str],
                 question_san_batch: List[str],
                 template_type_batch: Union[List[List[str]], List[str]],
                 entities_from_ner_batch: List[List[str]],
                 types_from_ner_batch: List[List[str]],
                 entity_tags_batch: List[List[str]],
                 probas_batch: List[List[float]],
                 answer_types_batch: List[Set[str]] = None,
                 entities_to_link_batch: List[List[int]] = None) -> Tuple[List[Any], List[Any]]:

        candidate_outputs_batch, template_answers_batch = [], []
        if not answer_types_batch or answer_types_batch[0] is None:
            answer_types_batch = [[] for _ in question_batch]
        if not entities_to_link_batch or entities_to_link_batch[0] is None:
            entities_to_link_batch = [[1 for _ in substr_list] for substr_list in entities_from_ner_batch]
        log.debug(f"kbqa inputs {question_batch} {question_san_batch} template_type_batch: {template_type_batch} --- "
                  f"entities_from_ner: {entities_from_ner_batch} --- types_from_ner: {types_from_ner_batch} --- "
                  f"entity_tags_batch: {entity_tags_batch} --- answer_types_batch: "
                  f"{[list(elem)[:3] for elem in answer_types_batch]}")
        for question, question_sanitized, template_type, entities_from_ner, types_from_ner, entity_tags_list, \
            probas, entities_to_link, answer_types in zip(question_batch, question_san_batch, template_type_batch,
                                                          entities_from_ner_batch, types_from_ner_batch,
                                                          entity_tags_batch, probas_batch, entities_to_link_batch,
                                                          answer_types_batch):
            if template_type == "-1":
                template_type = "7"
            candidate_outputs, template_answer = \
                self.find_candidate_answers(question, question_sanitized, template_type, entities_from_ner,
                                            types_from_ner, entity_tags_list, probas, entities_to_link, answer_types)
            candidate_outputs_batch.append(candidate_outputs)
            template_answers_batch.append(template_answer)

        if self.return_answers:
            answers = self.rel_ranker(question_batch, template_type_batch, candidate_outputs_batch,
                                      entities_from_ner_batch, template_answers_batch)
            log.debug(f"(__call__)answers: {answers}")
            if not answers:
                answers = ["Not Found" for _ in question_batch]
            return answers
        else:
            return candidate_outputs_batch, template_answers_batch

    def parse_queries_info(self, question, queries_info, entity_ids, type_ids, rels_from_template):
        parsed_queries_info = []
        question_tokens = nltk.word_tokenize(question)
        rels_scores_dict = {}
        for query_info in queries_info:
            query = query_info["query_template"].lower()
            for old_tok, new_tok in self.replace_tokens:
                query = query.replace(old_tok, new_tok)
            log.debug(f"\n_______________________________\nquery: {query}\n_______________________________\n")
            entities_and_types_select = query_info["entities_and_types_select"]
            rels_for_search = query_info["rank_rels"]
            rel_types = query_info["rel_types"]
            n_hops = query_info["n_hops"]
            unk_rels = query_info.get("unk_rels", [])
            query_seq_num = query_info["query_sequence"]
            return_if_found = query_info["return_if_found"]
            log.debug(f"(query_parser)query: {query}, rels_for_search {rels_for_search}, rel_types {rel_types} "
                      f"n_hops {n_hops}, {query_seq_num}, {return_if_found}")
            query_triplets = re.findall("{[ ]?(.*?)[ ]?}", query)[0].split(' . ')
            log.debug(f"(query_parser)query_triplets: {query_triplets}")
            query_triplets_split = [triplet.split(' ')[:3] for triplet in query_triplets]
            property_types = {}
            for rel_type, query_triplet in zip(rel_types, query_triplets_split):
                if query_triplet[1].startswith("?") and rel_type == "qualifier":
                    property_types[query_triplet[1]] = rel_type
            query_sequence_dict = {num + 1: triplet for num, triplet in enumerate(query_triplets_split)}
            query_sequence = []
            for i in query_seq_num:
                query_sequence.append(query_sequence_dict[i])
            triplet_info_list = [("forw" if triplet[2].startswith('?') else "backw", search_source, rel_type, n_hop)
                                 for search_source, triplet, rel_type, n_hop in \
                                 zip(rels_for_search, query_sequence, rel_types, n_hops)
                                 if search_source != "do_not_rank"]
            log.debug(f"(query_parser)query_sequence_dict: {query_sequence_dict} --- rel_directions: "
                      f"{triplet_info_list} --- query_sequence: {query_sequence}")
            entity_ids = [entity[:self.entities_to_leave] for entity in entity_ids]
            rels, entities_rel_conn = [], set()
            if rels_from_template is not None:
                rels = [[(rel, 1.0) for rel in rel_list] for rel_list in rels_from_template]
            elif not rels:
                for triplet_info in triplet_info_list:
                    ex_rels, cur_rels_scores_dict, entity_rel_conn = self.find_top_rels(question, entity_ids,
                                                                                        triplet_info)
                    rels.append(ex_rels)
                    rels_scores_dict = {**rels_scores_dict, **cur_rels_scores_dict}
                    entities_rel_conn = entities_rel_conn.union(entity_rel_conn)
            log.debug(f"(query_parser)rels: {rels}")
            rels_from_query = [triplet[1] for triplet in query_triplets_split if triplet[1].startswith('?')]
            qualifier_rels = [triplet[1] for triplet in query_triplets_split if triplet[1].startswith("pq:P")]

            answer_ent, order_info, filter_from_query = find_query_features(query, qualifier_rels, question)
            log.debug(f"(query_parser) filter_from_query: {filter_from_query} --- order_info: {order_info}")

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

            entity_positions, type_positions = [elem.split('_') for elem in entities_and_types_select.split(' ')]
            log.debug(f"entity_positions {entity_positions}, type_positions {type_positions}")
            selected_entity_ids, selected_type_ids = [], []
            if len(entity_ids) > 1 and len(entity_positions) == 1:
                selected_entity_ids = []
                for j in range(max([len(elem) for elem in entity_ids])):
                    for elem in entity_ids:
                        if j < len(elem):
                            selected_entity_ids.append(elem[j])
                selected_entity_ids = [selected_entity_ids]
            elif entity_ids:
                selected_entity_ids = [entity_ids[int(pos) - 1] for pos in entity_positions if int(pos) > 0]
            if type_ids:
                selected_type_ids = [type_ids[int(pos) - 1][:self.types_to_leave]
                                     for pos in type_positions if int(pos) > 0]
            entity_combs = make_combs(selected_entity_ids, permut=True)
            type_combs = make_combs(selected_type_ids, permut=False)
            log.debug(f"(query_parser)entity_combs: {entity_combs[:3]}, type_combs: {type_combs[:3]},"
                      f" rel_combs: {rel_combs[:3]}")

            all_combs_list = list(itertools.product(entity_combs, type_combs, rel_combs))
            all_combs_list = sorted(all_combs_list, key=lambda x: (sum([elem[-1] for elem in x]), x[0][-1]))
            parsed_queries_info.append({"query_triplets": query_triplets,
                                        "query_sequence": query_sequence,
                                        "rels_from_query": rels_from_query,
                                        "answer_ent": answer_ent,
                                        "filter_info": filter_info,
                                        "order_info": order_info,
                                        "rel_types": rel_types,
                                        "unk_rels": unk_rels,
                                        "return_if_found": return_if_found,
                                        "selected_entity_ids": selected_entity_ids,
                                        "selected_type_ids": selected_type_ids,
                                        "rels": rels,
                                        "entities_rel_conn": entities_rel_conn,
                                        "entity_combs": entity_combs,
                                        "type_combs": type_combs,
                                        "rel_combs": rel_combs,
                                        "all_combs_list": all_combs_list})
        return parsed_queries_info, rels_scores_dict

    def check_valid_query(self, entities_rel_conn, query_hdt_seq):
        entity_rel_valid = True
        if entities_rel_conn:
            for query_hdt_elem in query_hdt_seq:
                entity, rel = "", ""
                if len(query_hdt_elem) == 3 and any([query_hdt_elem[i].startswith("?") for i in [0, 2]]):
                    if "statement" in self.kb_prefixes and query_hdt_elem[1].startswith(self.kb_prefixes["statement"]):
                        continue
                    else:
                        if not query_hdt_elem[0].startswith("?"):
                            entity = query_hdt_elem[0].split("/")[-1]
                        elif not query_hdt_elem[2].startswith("?"):
                            entity = query_hdt_elem[2].split("/")[-1]
                        if not query_hdt_elem[1].startswith("?"):
                            rel = query_hdt_elem[1].split("/")[-1]
                        if entity and rel and rel not in self.kb_prefixes["type_rels"] \
                                and (entity, rel) not in entities_rel_conn:
                            entity_rel_valid = False
        return entity_rel_valid

    def query_parser(self, question: str,
                     queries_info: Dict[str, str],
                     entity_ids: List[List[str]],
                     type_ids: List[List[str]],
                     answer_types: Set[str],
                     rels_from_template: Optional[List[Tuple[str]]] = None) -> Union[List[Dict[str, Any]], list]:
        parsed_queries_info, rels_scores_dict = self.parse_queries_info(question, queries_info, entity_ids, type_ids,
                                                                        rels_from_template)
        queries_list, parser_info_list, entity_conf_list = [], [], []
        new_combs_list, query_info_list = [], []
        combs_num_list = [len(parsed_query_info["all_combs_list"]) for parsed_query_info in parsed_queries_info]
        if combs_num_list:
            max_comb_nums = max(combs_num_list)
        else:
            max_comb_nums = 0
        for comb_num in range(max_comb_nums):
            for parsed_query_info in parsed_queries_info:
                if comb_num < min(len(parsed_query_info["all_combs_list"]), self.max_comb_num):
                    query_triplets = parsed_query_info["query_triplets"]
                    query_sequence = parsed_query_info["query_sequence"]
                    rels_from_query = parsed_query_info["rels_from_query"]
                    answer_ent = parsed_query_info["answer_ent"]
                    filter_info = parsed_query_info["filter_info"]
                    order_info = parsed_query_info["order_info"]
                    rel_types = parsed_query_info["rel_types"]
                    unk_rels = parsed_query_info["unk_rels"]
                    return_if_found = parsed_query_info["return_if_found"]
                    entities_rel_conn = parsed_query_info["entities_rel_conn"]
                    combs = parsed_query_info["all_combs_list"][comb_num]
                    if combs[0][-1] == 0:
                        entity_conf_list.append(1.0)
                    else:
                        entity_conf_list.append(0.9)
                    query_hdt_seq = [fill_query(query_hdt_elem, combs[0], combs[1], combs[2],
                                                self.map_query_str_to_kb)
                                     for query_hdt_elem in query_sequence]
                    if comb_num == 0:
                        log.debug(f"\n______________________\nfilled query: {query_hdt_seq}\n______________________\n")

                    entity_rel_valid = self.check_valid_query(entities_rel_conn, query_hdt_seq)
                    if entity_rel_valid:
                        new_combs_list.append(combs)
                        queries_list.append((answer_ent, rels_from_query, query_hdt_seq, filter_info, order_info,
                                             answer_types, rel_types, return_if_found))
                        query_info_list.append((query_triplets, query_hdt_seq, answer_ent, filter_info, order_info))
                        parser_info_list.append("query_execute")
                    if comb_num < 3 and unk_rels:
                        unk_query_sequence = copy.deepcopy(query_sequence)
                        unk_rels_from_query = copy.deepcopy(rels_from_query)
                        for unk_rel, rel_var in zip(unk_rels, ["?p", "?p2"]):
                            unk_query_sequence[int(unk_rel) - 1][1] = rel_var
                            combs[-1][int(unk_rel) - 1] = (rel_var, 1.0)
                            if rel_var not in rels_from_query:
                                unk_rels_from_query.append(rel_var)
                        query_hdt_seq = [
                            fill_query(query_hdt_elem, combs[0], combs[1], combs[2], self.map_query_str_to_kb)
                            for query_hdt_elem in unk_query_sequence]
                        new_combs_list.append(combs)
                        queries_list.append((answer_ent, unk_rels_from_query, query_hdt_seq, filter_info, order_info,
                                             answer_types, rel_types, return_if_found))
                        query_info_list.append((query_triplets, query_hdt_seq, answer_ent, filter_info, order_info))
                        parser_info_list.append("query_execute")

        outputs_list = self.wiki_parser(parser_info_list, queries_list)
        outputs = self.parse_outputs(outputs_list, new_combs_list, query_info_list, entity_conf_list, rels_scores_dict)
        return outputs

    def parse_outputs(self, outputs_list, combs_list, query_info_list, entity_conf_list, rels_scores_dict):
        outputs = []
        if isinstance(outputs_list, list) and outputs_list:
            outputs_len = len(outputs_list)
            combs_list = combs_list[:outputs_len]
            entity_conf_list = entity_conf_list[:outputs_len]
            for combs, query_info, entity_conf, (answers_list, found_rels_list, found_combs_list) in \
                    zip(combs_list, query_info_list, entity_conf_list, outputs_list):
                for answers, found_rels, found_comb in zip(answers_list, found_rels_list, found_combs_list):
                    found_rels = [found_rel.split("/")[-1] for found_rel in found_rels]
                    new_combs = list(copy.deepcopy(combs))
                    found_unk_rel = False
                    for j, rel_var in enumerate(["?p", "?p2"]):
                        if isinstance(new_combs[2][j], tuple) and new_combs[2][j][0] == rel_var:
                            if found_rels:
                                new_combs[2][j] = (found_rels[j], rels_scores_dict.get(found_rels[j], 1.0))
                            else:
                                new_combs[2][j] = (new_combs[2][j][0], 0.0)
                            found_unk_rel = True
                    if found_rels and not found_unk_rel:
                        new_combs[2] = new_combs[2][:-1] + [(found_rels[0], 1.0), new_combs[2][-1]]
                    confidence = np.prod([score for rel, score in new_combs[2][:-1]])
                    if answers:
                        outputs.append([new_combs[0], new_combs[1]] + [rel for rel, score in new_combs[2][:-1]] +
                                       answers + [(confidence, entity_conf), found_comb, query_info, new_combs[2]])
            outputs_dict = defaultdict(list)
            types_dict = defaultdict(list)
            for output in outputs:
                key = (tuple(output[0]), tuple([rel.split("/")[-1] for rel in output[2:-5]]))
                if key not in outputs_dict or output[-5:] not in outputs_dict[key]:
                    outputs_dict[key].append(output[-5:])
                    types_dict[key].append(tuple(output[1]))
            outputs = []
            for (entity_comb, rel_comb), output in outputs_dict.items():
                type_comb = types_dict[(entity_comb, rel_comb)]
                output_conf = [elem[1] for elem in output]
                output_conf = sorted(output_conf, key=lambda x: x[0] * x[1], reverse=True)
                found_combs = [elem[2] for elem in output]
                queries = [elem[3] for elem in output]
                rel_combs = [elem[4] for elem in output]
                cur_rel_comb = rel_combs[0]
                cur_rel_comb = [rel for rel, score in cur_rel_comb[:-1]]
                sparql_query = make_sparql_query(queries[0], entity_comb, rel_combs[0], type_comb[0],
                                                 self.gold_query_info)
                parser_info_list = ["fill_triplets"]
                parser_query_list = [(queries[0][1], queries[0][2], found_combs[0])]
                filled_triplets = self.wiki_parser(parser_info_list, parser_query_list)
                outputs.append({"entities": entity_comb, "types": type_comb, "relations": list(cur_rel_comb),
                                "answers": tuple([ans for ans, *_ in output]), "output_conf": output_conf[0],
                                "sparql_query": sparql_query, "triplets": filled_triplets[0]})
        return outputs


@register('query_formatter')
class QueryFormatter(Component):
    def __init__(self, query_info: Dict[str, str], replace_prefixes: Dict[str, str] = None, **kwargs):
        self.query_info = query_info
        self.replace_prefixes = replace_prefixes

    def __call__(self, queries_batch):
        parsed_queries_batch = []
        for query in queries_batch:
            query_split = re.findall("{[ ]?(.*?)[ ]?}", query)
            init_query_triplets, query_triplets = [], []
            if query_split:
                init_query_triplets = query_split[0].split('. ')
            for triplet in init_query_triplets:
                triplet = " ".join([elem.strip("<>") for elem in triplet.strip().split()])
                if self.replace_prefixes:
                    for old_prefix, new_prefix in self.replace_prefixes.items():
                        triplet = triplet.replace(old_prefix, new_prefix)
                query_triplets.append(triplet)
            answer_ent, order_info, filter_from_query = find_query_features(query, order_from_query=True)
            query_info = (query_triplets, answer_ent, filter_from_query, order_info)
            query = merge_sparql_query(query_info, self.query_info)
            parsed_queries_batch.append(query)
        return parsed_queries_batch
