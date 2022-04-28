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

import json
from logging import getLogger
from typing import Tuple, List, Optional, Union, Any, Set

from bs4 import BeautifulSoup
from whapi import search, get_html

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.entity_extraction.entity_linking import EntityLinker
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.template_matcher import TemplateMatcher

log = getLogger(__name__)


class QueryGeneratorBase(Component, Serializable):
    """
        This class takes as input entity substrings, defines the template of the query and
        fills the slots of the template with candidate entities and relations.
    """

    def __init__(self, template_matcher: TemplateMatcher,
                 entity_linker: EntityLinker,
                 rel_ranker: RelRankerInfer,
                 load_path: str,
                 rank_rels_filename_1: str,
                 rank_rels_filename_2: str,
                 sparql_queries_filename: str,
                 wiki_parser=None,
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 7,
                 syntax_structure_known: bool = False,
                 use_wp_api_requester: bool = False,
                 use_el_api_requester: bool = False,
                 use_alt_templates: bool = True,
                 use_add_templates: bool = False, *args, **kwargs) -> None:
        """

        Args:
            template_matcher: component deeppavlov.models.kbqa.template_matcher
            entity_linker: component deeppavlov.models.entity_extraction.entity_linking for linking of entities
            rel_ranker: component deeppavlov.models.kbqa.rel_ranking_infer
            load_path: path to folder with wikidata files
            rank_rels_filename_1: file with list of rels for first rels in questions with ranking 
            rank_rels_filename_2: file with list of rels for second rels in questions with ranking
            sparql_queries_filename: file with sparql query templates
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            entities_to_leave: how many entities to leave after entity linking
            rels_to_leave: how many relations to leave after relation ranking
            syntax_structure_known: if syntax tree parser was used to define query template type
            use_wp_api_requester: whether deeppavlov.models.api_requester.api_requester component will be used for
                Wiki Parser
            use_el_api_requester: whether deeppavlov.models.api_requester.api_requester component will be used for
                Entity Linking
            use_alt_templates: whether to use alternative templates if no answer was found for default query template
        """
        super().__init__(save_path=None, load_path=load_path)
        self.template_matcher = template_matcher
        self.entity_linker = entity_linker
        self.wiki_parser = wiki_parser
        self.rel_ranker = rel_ranker
        self.rank_rels_filename_1 = rank_rels_filename_1
        self.rank_rels_filename_2 = rank_rels_filename_2
        self.rank_list_0 = []
        self.rank_list_1 = []
        self.entities_to_leave = entities_to_leave
        self.rels_to_leave = rels_to_leave
        self.syntax_structure_known = syntax_structure_known
        self.use_wp_api_requester = use_wp_api_requester
        self.use_el_api_requester = use_el_api_requester
        self.use_alt_templates = use_alt_templates
        self.use_add_templates = use_add_templates
        self.sparql_queries_filename = sparql_queries_filename

        self.load()

    def load(self) -> None:
        with open(self.load_path / self.rank_rels_filename_1, 'r') as fl1:
            lines = fl1.readlines()
            self.rank_list_0 = [line.split('\t')[0] for line in lines]

        with open(self.load_path / self.rank_rels_filename_2, 'r') as fl2:
            lines = fl2.readlines()
            self.rank_list_1 = [line.split('\t')[0] for line in lines]

        self.template_queries = read_json(str(expand_path(self.sparql_queries_filename)))

    def save(self) -> None:
        pass

    def find_candidate_answers(self, question: str,
                               question_sanitized: str,
                               template_types: Union[List[str], str],
                               entities_from_ner: List[str],
                               entity_tags: List[str],
                               answer_types: Set[str]) -> Tuple[Union[Union[List[List[Union[str, float]]],
                                                                            List[Any]], Any],
                                                                Union[str, Any], Union[List[Any], Any]]:
        candidate_outputs = []
        self.template_nums = template_types

        replace_tokens = [(' - ', '-'), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''),
                          (')', ''), ('–', '-')]
        for old, new in replace_tokens:
            question = question.replace(old, new)

        entities_from_template, types_from_template, rels_from_template, rel_dirs_from_template, query_type_template, \
        entity_types, template_answer, answer_types, template_found = self.template_matcher(question_sanitized,
                                                                                            entities_from_ner)
        self.template_nums = [query_type_template]
        templates_nums = []

        log.debug(
            f"question: {question} entities_from_template {entities_from_template} template_type {self.template_nums} "
            f"types from template {types_from_template} rels_from_template {rels_from_template}")

        if entities_from_template or types_from_template:
            if rels_from_template[0][0] == "PHOW":
                how_to_content = self.find_answer_wikihow(entities_from_template[0])
                candidate_outputs = [["PHOW", how_to_content, 1.0]]
            else:
                entity_ids = self.get_entity_ids(entities_from_template, entity_tags, question)
                log.debug(f"entities_from_template {entities_from_template}")
                log.debug(f"entity_types {entity_types}")
                log.debug(f"types_from_template {types_from_template}")
                log.debug(f"rels_from_template {rels_from_template}")
                log.debug(f"entity_ids {entity_ids}")
                candidate_outputs, templates_nums = \
                    self.sparql_template_parser(question_sanitized, entity_ids, [], answer_types,
                                                rels_from_template, rel_dirs_from_template)

        if not candidate_outputs and entities_from_ner:
            log.debug(f"(__call__)entities_from_ner: {entities_from_ner}")
            entity_ids = self.get_entity_ids(entities_from_ner, entity_tags, question)
            log.debug(f"(__call__)entity_ids: {entity_ids}")
            self.template_nums = template_types
            log.debug(f"(__call__)self.template_nums: {self.template_nums}")
            if not self.syntax_structure_known:
                entity_ids = entity_ids[:3]
            candidate_outputs, templates_nums = self.sparql_template_parser(question_sanitized, entity_ids, [],
                                                                            answer_types)
        return candidate_outputs, template_answer, templates_nums

    def get_entity_ids(self, entities: List[str], tags: List[str], question: str) -> List[List[str]]:
        entity_ids = []
        el_output = []
        try:
            el_output = self.entity_linker([entities], [tags], [[question]], [None], [None])
        except json.decoder.JSONDecodeError:
            log.info("not received output from entity linking")
        if el_output:
            if self.use_el_api_requester:
                el_output = el_output[0]
            if el_output:
                if isinstance(el_output[0], dict):
                    entity_ids = [entity_info.get("entity_ids", []) for entity_info in el_output]
                if isinstance(el_output[0], list):
                    entity_ids, *_ = el_output
            if not self.use_el_api_requester and entity_ids:
                entity_ids = entity_ids[0]

        return entity_ids

    def sparql_template_parser(self, question: str,
                               entity_ids: List[List[str]],
                               type_ids: List[List[str]],
                               answer_types: List[str],
                               rels_from_template: Optional[List[Tuple[str]]] = None,
                               rel_dirs_from_template: Optional[List[str]] = None) -> Tuple[Union[None, List[Any]],
                                                                                            List[Any]]:
        candidate_outputs = []
        log.debug(f"use alternative templates {self.use_alt_templates}")
        log.debug(f"(find_candidate_answers)self.template_nums: {self.template_nums}")
        templates = []
        templates_nums = []
        for template_num in self.template_nums:
            for num, template in self.template_queries.items():
                if (num == template_num and self.syntax_structure_known) or \
                        (template["template_num"] == template_num and not self.syntax_structure_known):
                    templates.append(template)
                    templates_nums.append(num)
        new_templates = []
        new_templates_nums = []
        for template, template_num in zip(templates, templates_nums):
            if (not self.syntax_structure_known and [len(entity_ids), len(type_ids)] == template[
                "entities_and_types_num"]) or self.syntax_structure_known:
                new_templates.append(template)
                new_templates_nums.append(template_num)

        templates = new_templates
        templates_nums = new_templates_nums

        templates_string = '\n'.join([template["query_template"] for template in templates])
        log.debug(f"{templates_string}")
        if not templates:
            return candidate_outputs, []
        if rels_from_template is not None:
            query_template = {}
            for template in templates:
                if template["rel_dirs"] == rel_dirs_from_template:
                    query_template = template
            if query_template:
                entities_and_types_select = query_template["entities_and_types_select"]
                candidate_outputs = self.query_parser(question, query_template, entities_and_types_select,
                                                      entity_ids, type_ids, answer_types, rels_from_template)
        else:
            for template in templates:
                entities_and_types_select = template["entities_and_types_select"]
                candidate_outputs = self.query_parser(question, template, entities_and_types_select,
                                                      entity_ids, type_ids, answer_types, rels_from_template)
                if self.use_add_templates:
                    additional_templates = template.get("additional_templates", [])
                    templates_nums += additional_templates
                    for add_template_num in additional_templates:
                        candidate_outputs += self.query_parser(question, self.template_queries[add_template_num],
                                                               entities_and_types_select, entity_ids, type_ids,
                                                               answer_types, rels_from_template)
                if candidate_outputs:
                    templates_nums = list(set(templates_nums))
                    return candidate_outputs, templates_nums

            if not candidate_outputs and self.use_alt_templates:
                alternative_templates = templates[0]["alternative_templates"]
                for template_num, entities_and_types_select in alternative_templates:
                    candidate_outputs = self.query_parser(question, self.template_queries[template_num],
                                                          entities_and_types_select, entity_ids, type_ids,
                                                          answer_types, rels_from_template)
                    templates_nums.append(template_num)
                    if candidate_outputs:
                        templates_nums = list(set(templates_nums))
                        return candidate_outputs, templates_nums

        log.debug("candidate_rels_and_answers:\n" + '\n'.join([str(output) for output in candidate_outputs[:5]]))

        templates_nums = list(set(templates_nums))
        return candidate_outputs, templates_nums

    def find_top_rels(self, question: str, entity_ids: List[List[str]], triplet_info: Tuple) -> List[Tuple[str, Any]]:
        ex_rels = []
        direction, source, rel_type = triplet_info
        if source == "wiki":
            queries_list = list({(entity, direction, rel_type) for entity_id in entity_ids
                                 for entity in entity_id[:self.entities_to_leave]})
            parser_info_list = ["find_rels" for i in range(len(queries_list))]
            try:
                ex_rels = self.wiki_parser(parser_info_list, queries_list)
            except json.decoder.JSONDecodeError:
                log.info("find_top_rels, not received output from wiki parser")
            if self.use_wp_api_requester and ex_rels:
                ex_rels = [rel[0] for rel in ex_rels]
            ex_rels = list(set(ex_rels))
            ex_rels = [rel.split('/')[-1] for rel in ex_rels]
        elif source == "rank_list_1":
            ex_rels = self.rank_list_0
        elif source == "rank_list_2":
            ex_rels = self.rank_list_1
        rels_with_scores = []
        ex_rels = [rel for rel in ex_rels if rel.startswith("P")]
        if ex_rels:
            rels_with_scores = self.rel_ranker.rank_rels(question, ex_rels)
        return rels_with_scores[:self.rels_to_leave]

    def find_answer_wikihow(self, howto_sentence: str) -> str:
        tags = []
        search_results = search(howto_sentence, 5)
        if search_results:
            article_id = search_results[0]["article_id"]
            html = get_html(article_id)
            page = BeautifulSoup(html, 'lxml')
            tags = list(page.find_all(['p']))
        if tags:
            howto_content = f"{tags[0].text.strip()}@en"
        else:
            howto_content = "Not Found"
        return howto_content

    def query_parser(self, question, query_template, entities_and_types_select, entity_ids, type_ids, answer_types,
                     rels_from_template):
        raise NotImplementedError
