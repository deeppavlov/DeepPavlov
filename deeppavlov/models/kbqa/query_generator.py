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
from collections import namedtuple

import nltk

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import read_json
from deeppavlov.models.kbqa.template_matcher import TemplateMatcher
from deeppavlov.models.kbqa.entity_linking import EntityLinker
from deeppavlov.models.kbqa.wiki_parser import WikiParser
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.rel_ranking_bert_infer import RelRankerBertInfer
from deeppavlov.models.kbqa.utils import \
    extract_year, extract_number, order_of_answers_sorting, make_combs, fill_query

log = getLogger(__name__)


@register('query_generator')
class QueryGenerator(Component, Serializable):
    """
        This class takes as input entity substrings, defines the template of the query and
        fills the slots of the template with candidate entities and relations.
    """

    def __init__(self, template_matcher: TemplateMatcher,
                 linker_entities: EntityLinker,
                 linker_types: EntityLinker,
                 wiki_parser: WikiParser,
                 rel_ranker: Union[RelRankerInfer, RelRankerBertInfer],
                 load_path: str,
                 rank_rels_filename_1: str,
                 rank_rels_filename_2: str,
                 sparql_queries_filename: str,
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 7,
                 return_answers: bool = False, **kwargs) -> None:
        """

        Args:
            template_matcher: component deeppavlov.models.kbqa.template_matcher
            linker_entities: component deeppavlov.models.kbqa.entity_linking for linking of entities
            linker_types: component deeppavlov.models.kbqa.entity_linking for linking of types
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            rel_ranker: component deeppavlov.models.kbqa.rel_ranking_infer
            load_path: path to folder with wikidata files
            rank_rels_filename_1: file with list of rels for first rels in questions with ranking 
            rank_rels_filename_2: file with list of rels for second rels in questions with ranking
            sparql_queries_filename: file with sparql query templates
            entities_to_leave: how many entities to leave after entity linking
            rels_to_leave: how many relations to leave after relation ranking
            sparql_queries_filename: file with a dict of sparql queries
            return_answers: whether to return answers or candidate answers
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.template_matcher = template_matcher
        self.linker_entities = linker_entities
        self.linker_types = linker_types
        self.wiki_parser = wiki_parser
        self.rel_ranker = rel_ranker
        self.rank_rels_filename_1 = rank_rels_filename_1
        self.rank_rels_filename_2 = rank_rels_filename_2
        self.rank_list_0 = []
        self.rank_list_1 = []
        self.entities_to_leave = entities_to_leave
        self.rels_to_leave = rels_to_leave
        self.sparql_queries_filename = sparql_queries_filename
        self.return_answers = return_answers

        self.load()

    def load(self) -> None:
        with open(self.load_path / self.rank_rels_filename_1, 'r') as fl1:
            lines = fl1.readlines()
            self.rank_list_0 = [line.split('\t')[0] for line in lines]

        with open(self.load_path / self.rank_rels_filename_2, 'r') as fl2:
            lines = fl2.readlines()
            self.rank_list_1 = [line.split('\t')[0] for line in lines]

        self.template_queries = read_json(self.load_path / self.sparql_queries_filename)

    def save(self) -> None:
        pass

    def __call__(self, question_batch: List[str],
                 template_type_batch: List[str],
                 entities_from_ner_batch: List[List[str]],
                 types_from_ner_batch: List[List[str]]) -> List[Union[List[Tuple[str, Any]], List[str]]]:

        candidate_outputs_batch = []
        for question, template_type, entities_from_ner, types_from_ner in \
                zip(question_batch, template_type_batch, entities_from_ner_batch, types_from_ner_batch):

            candidate_outputs = []
            self.template_num = template_type

            replace_tokens = [(' - ', '-'), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''),
                              (')', ''), ('–', '-')]
            for old, new in replace_tokens:
                question = question.replace(old, new)

            entities_from_template, types_from_template, rels_from_template, rel_dirs_from_template, \
            query_type_template = self.template_matcher(question)
            self.template_num = query_type_template

            log.debug(f"question: {question}\n")
            log.debug(f"template_type {self.template_num}")

            if entities_from_template or types_from_template:
                entity_ids = self.get_entity_ids(entities_from_template, "entities")
                type_ids = self.get_entity_ids(types_from_template, "types")
                log.debug(f"entities_from_template {entities_from_template}")
                log.debug(f"types_from_template {types_from_template}")
                log.debug(f"rels_from_template {rels_from_template}")
                log.debug(f"entity_ids {entity_ids}")
                log.debug(f"type_ids {type_ids}")

                candidate_outputs = self.find_candidate_answers(question, entity_ids, type_ids, rels_from_template,
                                                                rel_dirs_from_template)

            if not candidate_outputs and entities_from_ner:
                log.debug(f"(__call__)entities_from_ner: {entities_from_ner}")
                log.debug(f"(__call__)types_from_ner: {types_from_ner}")
                entity_ids = self.get_entity_ids(entities_from_ner, "entities")
                type_ids = self.get_entity_ids(types_from_ner, "types")
                log.debug(f"(__call__)entity_ids: {entity_ids}")
                log.debug(f"(__call__)type_ids: {type_ids}")
                self.template_num = template_type[0]
                log.debug(f"(__call__)self.template_num: {self.template_num}")
                candidate_outputs = self.find_candidate_answers(question, entity_ids[:2], type_ids)
            candidate_outputs_batch.append(candidate_outputs)
        if self.return_answers:
            answers = self.rel_ranker(question_batch, candidate_outputs_batch)
            log.debug(f"(__call__)answers: {answers}")
            return answers
        else:
            log.debug(f"(__call__)candidate_outputs_batch: {[output[:5] for output in candidate_outputs_batch]}")
            return candidate_outputs_batch

    def get_entity_ids(self, entities: List[str], what_to_link: str) -> List[List[str]]:
        entity_ids = []
        for entity in entities:
            entity_id = []
            if what_to_link == "entities":
                entity_id, confidences = self.linker_entities(entity)
            if what_to_link == "types":
                entity_id, confidences = self.linker_types(entity)
            entity_ids.append(entity_id[:15])
        return entity_ids

    def find_candidate_answers(self, question: str,
                               entity_ids: List[List[str]],
                               type_ids: List[List[str]],
                               rels_from_template: Optional[List[Tuple[str]]] = None,
                               rel_dirs_from_template: Optional[List[str]] = None) -> List[Tuple[str]]:
        candidate_outputs = []
        log.debug(f"(find_candidate_answers)self.template_num: {self.template_num}")
        templates = [template for num, template in self.template_queries.items() if
                     template["template_num"] == self.template_num]
        templates = [template for template in templates if (template["exact_entity_type_match"] and
                                                            template["entities_and_types_num"] == [len(entity_ids),
                                                                                                   len(type_ids)])
                     or not template["exact_entity_type_match"]]
        if not templates:
            return candidate_outputs
        if rels_from_template is not None:
            query_template = {}
            for template in templates:
                if template["rel_dirs"] == rel_dirs_from_template:
                    query_template = template
            if query_template:
                candidate_outputs = self.query_parser(question, query_template, entity_ids, type_ids,
                                                      rels_from_template)
        else:
            for template in templates:
                candidate_outputs = self.query_parser(question, template, entity_ids, type_ids, rels_from_template)
                if candidate_outputs:
                    return candidate_outputs

            if not candidate_outputs:
                log.debug(f"(find_candidate_answers)templates: {templates}")
                alternative_templates = templates[0]["alternative_templates"]
                for template in alternative_templates:
                    candidate_outputs = self.query_parser(question, template, entity_ids, type_ids, rels_from_template)
                    return candidate_outputs

        log.debug("candidate_rels_and_answers:\n" + '\n'.join([str(output) for output in candidate_outputs[:5]]))

        return candidate_outputs

    def query_parser(self, question: str, query_info: Dict[str, str],
                     entity_ids: List[List[str]], type_ids: List[List[str]],
                     rels_from_template: Optional[List[Tuple[str]]] = None) -> List[Tuple[str]]:
        candidate_outputs = []
        question_tokens = nltk.word_tokenize(question)
        query = query_info["query_template"].lower().replace("wdt:p31", "wdt:P31")
        rels_for_search = query_info["rank_rels"]
        query_seq_num = query_info["query_sequence"]
        return_if_found = query_info["return_if_found"]
        log.debug(f"(query_parser)quer: {query}, {rels_for_search}, {query_seq_num}, {return_if_found}")
        query_triplets = re.findall("{[ ]?(.*?)[ ]?}", query)[0].split(' . ')
        log.debug(f"(query_parser)query_triplets: {query_triplets}")
        query_triplets = [triplet.split(' ')[:3] for triplet in query_triplets]
        query_sequence_dict = {num: triplet for num, triplet in zip(query_seq_num, query_triplets)}
        query_sequence = []
        for i in range(1, max(query_seq_num) + 1):
            query_sequence.append(query_sequence_dict[i])
        log.debug(f"(query_parser)query_sequence: {query_sequence}")
        triplet_info_list = [("forw" if triplet[2].startswith('?') else "backw", search_source)
                             for search_source, triplet in zip(rels_for_search, query_triplets) if
                             search_source != "do_not_rank"]
        log.debug(f"(query_parser)rel_directions: {triplet_info_list}")
        entity_ids = [entity[:self.entities_to_leave] for entity in entity_ids]
        entity_combs = make_combs(entity_ids, permut=True)
        log.debug(f"(query_parser)entity_combs: {entity_combs[:3]}")
        type_combs = make_combs(type_ids, permut=False)
        log.debug(f"(query_parser)type_combs: {type_combs[:3]}")
        if rels_from_template is not None:
            rels = rels_from_template
        else:
            rels = [self.find_top_rels(question, entity_ids, triplet_info)
                    for triplet_info in triplet_info_list]
        log.debug(f"(query_parser)rels: {rels}")
        rels_from_query = [triplet[1] for triplet in query_triplets if triplet[1].startswith('?')]
        answer_ent = re.findall("select [\(]?([\S]+) ", query)
        order_info_nt = namedtuple("order_info", ["variable", "sorting_order"])
        order_variable = re.findall("order by (asc|desc)\((.*)\)", query)
        answers_sorting_order = order_of_answers_sorting(question)
        if order_variable:
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
        log.debug(f"(query_parser)filter_from_query: {filter_from_query}")
        rel_combs = make_combs(rels, permut=False)
        import datetime
        start_time = datetime.datetime.now()
        for combs in itertools.product(entity_combs, type_combs, rel_combs):
            query_hdt_seq = [
                fill_query(query_hdt_elem, combs[0], combs[1], combs[2]) for query_hdt_elem in query_sequence]
            candidate_output = self.wiki_parser(
                rels_from_query + answer_ent, query_hdt_seq, filter_info, order_info)
            candidate_outputs += [combs[2][:-1] + output for output in candidate_output]
            if return_if_found and candidate_output:
                return candidate_outputs
        log.debug(f"(query_parser)loop time: {datetime.datetime.now() - start_time}")
        log.debug(f"(query_parser)final outputs: {candidate_outputs[:3]}")

        return candidate_outputs

    def find_top_rels(self, question: str, entity_ids: List[List[str]], triplet_info: namedtuple) -> List[str]:
        ex_rels = []
        direction, source = triplet_info
        if source == "wiki":
            for entity_id in entity_ids:
                for entity in entity_id[:self.entities_to_leave]:
                    ex_rels += self.wiki_parser.find_rels(entity, direction)
            ex_rels = list(set(ex_rels))
            ex_rels = [rel.split('/')[-1] for rel in ex_rels]
        elif source == "rank_list_1":
            ex_rels = self.rank_list_0
        elif source == "rank_list_2":
            ex_rels = self.rank_list_1
        scores = self.rel_ranker.rank_rels(question, ex_rels)
        top_rels = [score[0] for score in scores]
        return top_rels[:self.rels_to_leave]
