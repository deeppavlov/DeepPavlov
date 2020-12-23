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

import nltk

from deeppavlov.core.common.registry import register
from deeppavlov.models.kbqa.wiki_parser_online import WikiParserOnline
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.rel_ranking_bert_infer import RelRankerBertInfer
from deeppavlov.models.kbqa.utils import \
    extract_year, extract_number, make_combs, fill_online_query
from deeppavlov.models.kbqa.query_generator_base import QueryGeneratorBase

log = getLogger(__name__)


@register('query_generator_online')
class QueryGeneratorOnline(QueryGeneratorBase):
    """
        Class for query generation online using Wikidata query service
    """

    def __init__(self, wiki_parser: WikiParserOnline,
                 rel_ranker: Union[RelRankerInfer, RelRankerBertInfer],
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 7,
                 return_answers: bool = False, *args, **kwargs) -> None:
        """

        Args:
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            rel_ranker: component deeppavlov.models.kbqa.rel_ranking_infer
            entities_to_leave: how many entities to leave after entity linking
            rels_to_leave: how many relations to leave after relation ranking
            return_answers: whether to return answers or candidate answers
        """
        self.wiki_parser = wiki_parser
        self.rel_ranker = rel_ranker
        self.entities_to_leave = entities_to_leave
        self.rels_to_leave = rels_to_leave
        self.return_answers = return_answers
        super().__init__(wiki_parser=self.wiki_parser, rel_ranker=self.rel_ranker,
                         entities_to_leave=self.entities_to_leave, rels_to_leave=self.rels_to_leave,
                         return_answers=self.return_answers, *args, **kwargs)

        self.load()

    def __call__(self, question_batch: List[str],
                 question_san_batch: List[str],
                 template_type_batch: List[str],
                 entities_from_ner_batch: List[List[str]],
                 types_from_ner_batch: List[List[str]]) -> List[Union[List[Tuple[str, Any]], List[str]]]:

        candidate_outputs_batch = []
        for question, question_sanitized, template_type, entities_from_ner, types_from_ner in \
                zip(question_batch, question_san_batch, template_type_batch,
                    entities_from_ner_batch, types_from_ner_batch):
            candidate_outputs, _ = self.find_candidate_answers(question, question_sanitized,
                                                            template_type, entities_from_ner, types_from_ner)
            candidate_outputs_batch.append(candidate_outputs)
        if self.return_answers:
            answers = self.rel_ranker(question_batch, candidate_outputs_batch)
            log.debug(f"(__call__)answers: {answers}")
            return answers
        else:
            log.debug(f"(__call__)candidate_outputs_batch: {[output[:5] for output in candidate_outputs_batch]}")
            return candidate_outputs_batch

    def query_parser(self, question: str, query_info: Dict[str, str],
                     entities_and_types_select: List[str],
                     entity_ids: List[List[str]], type_ids: List[List[str]],
                     rels_from_template: Optional[List[Tuple[str]]] = None) -> List[Tuple[str]]:
        question_tokens = nltk.word_tokenize(question)
        query = query_info["query_template"].lower().replace("wdt:p31", "wdt:P31")
        rels_for_search = query_info["rank_rels"]
        rel_types = query_info["rel_types"]
        rels_for_filter = query_info["filter_rels"]
        property_types = query_info["property_types"]
        query_seq_num = query_info["query_sequence"]
        return_if_found = query_info["return_if_found"]
        log.debug(f"(query_parser)query: {query}, {rels_for_search}, {query_seq_num}, {return_if_found}")
        query_triplets = re.findall("{[ ]?(.*?)[ ]?}", query)[0].split(' . ')
        log.debug(f"(query_parser)query_triplets: {query_triplets}")
        query_triplets = [triplet.split(' ')[:3] for triplet in query_triplets]
        triplet_info_list = [("forw" if triplet[2].startswith('?') else "backw", search_source, rel_type)
                             for search_source, triplet, rel_type in zip(rels_for_search, query_triplets, rel_types) if
                             search_source != "do_not_rank"]
        log.debug(f"(query_parser)rel_directions: {triplet_info_list}")
        rel_variables = re.findall(":(r[\d]{1,2})", query)
        entity_ids = [entity[:self.entities_to_leave] for entity in entity_ids]
        if rels_from_template is not None:
            rels = [[(rel, 1.0) for rel in rel_list] for rel_list in rels_from_template]
        else:
            rels = [self.find_top_rels(question, entity_ids, triplet_info)
                    for triplet_info in triplet_info_list]
        rels_list_for_filter = []
        rels_list_for_fill = []
        filter_rel_variables = []
        fill_rel_variables = []
        for rel_variable, rel_list, is_filter in zip(rel_variables, rels, rels_for_filter):
            if is_filter:
                rels_list_for_filter.append(rel_list)
                filter_rel_variables.append(rel_variable)
            else:
                rels_list_for_fill.append(rel_list)
                fill_rel_variables.append(rel_variable)
        log.debug(f"(query_parser)rels: {rels}")
        log.debug(f"rel_variables {rel_variables}, filter_rel_variables: {filter_rel_variables}")
        log.debug(f"rels_list_for_filter: {rels_list_for_filter}")
        log.debug(f"rels_list_for_fill: {rels_list_for_fill}")
        rels_from_query = list(set([triplet[1] for triplet in query_triplets if triplet[1].startswith('?')]))
        if "count" in query:
            answer_ent = re.findall("as (\?[\S]+)", query)
        else:
            answer_ent = re.findall("select [\(]?([\S]+) ", query)

        filter_from_query = re.findall("contains\((\?\w), (.+?)\)", query)
        log.debug(f"(query_parser)filter_from_query: {filter_from_query}")

        year = extract_year(question_tokens, question)
        number = extract_number(question_tokens, question)
        log.debug(f"year {year}, number {number}")
        if year:
            for elem in filter_from_query:
                query = query.replace(f"{elem[0]}, n", f"YEAR({elem[0]}), {year}")
        elif number:
            for elem in filter_from_query:
                query = query.replace(f"{elem[0]}, n", f"{elem[0]}, {number}")
        query = query.replace(" where", f" {' '.join(rels_from_query)} where")

        log.debug(f"(query_parser)query_with_filtering: {query}")
        rel_combs = make_combs(rels_list_for_fill, permut=False)
        log.debug(f"(query_parser)rel_combs: {rel_combs[:3]}")
        import datetime
        start_time = datetime.datetime.now()
        entity_positions, type_positions = [elem.split('_') for elem in entities_and_types_select.split(' ')]
        log.debug(f"entity_positions {entity_positions}, type_positions {type_positions}")
        selected_entity_ids = [entity_ids[int(pos) - 1] for pos in entity_positions if int(pos) > 0]
        selected_type_ids = [type_ids[int(pos) - 1] for pos in type_positions if int(pos) > 0]
        entity_combs = make_combs(selected_entity_ids, permut=True)
        log.debug(f"(query_parser)entity_combs: {entity_combs[:3]}")
        type_combs = make_combs(selected_type_ids, permut=False)
        log.debug(f"(query_parser)type_combs: {type_combs[:3]}")
        confidence = 0.0
        queries_list = []
        parser_info_list = []
        all_combs_list = list(itertools.product(entity_combs, type_combs, rel_combs))
        for comb_num, combs in enumerate(all_combs_list):
            filled_query, filter_rels = fill_online_query(query, combs[0], combs[1], combs[2], fill_rel_variables,
                                                          filter_rel_variables, rels_list_for_filter)
            if comb_num == 0:
                log.debug(f"\n___________________________\nfilled query: {filled_query}\n___________________________\n")
            queries_list.append((filled_query, return_if_found))
            parser_info_list.append("query_execute")

        candidate_outputs_list = self.wiki_parser(parser_info_list, queries_list)
        outputs_len = len(candidate_outputs_list)
        all_combs_list = all_combs_list[:outputs_len]
        out_vars = filter_rels + rels_from_query + answer_ent

        candidate_outputs = []
        for combs, candidate_output in zip(all_combs_list, candidate_outputs_list):
            candidate_output = [output for output in candidate_output
                                if (all([filter_value in output[filter_var[1:]]["value"]
                                         for filter_var, filter_value in property_types.items()])
                                    and all([not output[ent[1:]]["value"].startswith("http://www.wikidata.org/value")
                                             for ent in answer_ent]))]
            candidate_outputs += [combs[2][:-1] + [output[var[1:]]["value"] for var in out_vars] + [confidence]
                                  for output in candidate_output]

        log.debug(f"(query_parser)loop time: {datetime.datetime.now() - start_time}")
        log.debug(f"(query_parser)final outputs: {candidate_outputs[:3]}")

        return candidate_outputs
