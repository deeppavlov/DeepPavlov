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
from deeppavlov.models.kbqa.utils import preprocess_template_queries

log = getLogger(__name__)


class QueryGeneratorBase(Component, Serializable):
    """
        This class takes as input entity substrings, defines the template of the query and
        fills the slots of the template with candidate entities and relations.
    """

    def __init__(self, template_matcher: TemplateMatcher,
                 rel_ranker: RelRankerInfer,
                 load_path: str,
                 sparql_queries_filename: str,
                 entity_linker: EntityLinker,
                 rank_rels_filename_1: str = None,
                 rank_rels_filename_2: str = None,
                 wiki_parser=None,
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 7,
                 syntax_structure_known: bool = False,
                 use_wp_api_requester: bool = False,
                 use_el_api_requester: bool = False,
                 use_alt_templates: bool = True,
                 use_add_templates: bool = False,
                 return_answers: bool = True,
                 delete_rel_prefix: bool = True,
                 map_query_str_to_kb: List[Tuple[str, str]] = None, *args, **kwargs) -> None:
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
        self.return_answers = return_answers
        self.delete_rel_prefix = delete_rel_prefix
        self.map_query_str_to_kb = map_query_str_to_kb

        self.load()

    def load(self) -> None:
        if self.rank_rels_filename_1 is not None:
            with open(self.load_path / self.rank_rels_filename_1, 'r') as fl1:
                lines = fl1.readlines()
                self.rank_list_0 = [line.split('\t')[0] for line in lines]

        if self.rank_rels_filename_2 is not None:
            with open(self.load_path / self.rank_rels_filename_2, 'r') as fl2:
                lines = fl2.readlines()
                self.rank_list_1 = [line.split('\t')[0] for line in lines]

        template_queries = read_json(str(expand_path(self.sparql_queries_filename)))
        self.template_queries = preprocess_template_queries(template_queries)

    def save(self) -> None:
        pass

    def find_candidate_answers(self, question: str,
                               question_sanitized: str,
                               template_types: Union[List[str], str],
                               entities_from_ner: List[str],
                               types_from_ner: List[str],
                               entity_tags: List[str],
                               probas: List[float],
                               answer_types: Set[str]) -> Tuple[Union[Union[List[List[Union[str, float]]],
                                                                            List[Any]], Any],
                                                                Union[str, Any], Union[List[Any], Any]]:
        candidate_outputs = []
        self.template_nums = [template_types]

        replace_tokens = [(' - ', '-'), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''),
                          (')', ''), ('–', '-')]
        for old, new in replace_tokens:
            question = question.replace(old, new)

        entities_from_template, types_from_template, rels_from_template, rel_dirs_from_template, query_type_template, \
        entity_types, template_answer, template_answer_types, template_found = self.template_matcher(
            question_sanitized, entities_from_ner)
        if query_type_template:
            self.template_nums = [query_type_template]

        log.debug(
            f"question: {question} entities_from_template {entities_from_template} template_type {self.template_nums} "
            f"types from template {types_from_template} rels_from_template {rels_from_template} entities_from_ner "
            f"{entities_from_ner} types_from_ner {types_from_ner} answer_types {answer_types}")

        if entities_from_template or types_from_template:
            if rels_from_template[0][0] == "PHOW":
                how_to_content = self.find_answer_wikihow(entities_from_template[0])
                candidate_outputs = [["PHOW", how_to_content, 1.0]]
            else:
                entity_ids = self.get_entity_ids(entities_from_template, entity_tags, probas, question)
                type_ids = self.get_entity_ids(types_from_template, ["t" for _ in types_from_template],
                                               [1.0 for _ in types_from_template], question)
                log.debug(f"entities_from_template {entities_from_template}")
                log.debug(f"entity_types {entity_types}")
                log.debug(f"types_from_template {types_from_template}")
                log.debug(f"rels_from_template {rels_from_template}")
                log.debug(f"entity_ids {entity_ids}")
                candidate_outputs = self.sparql_template_parser(question_sanitized, entity_ids, type_ids,
                    template_answer_types, rels_from_template, rel_dirs_from_template)

        if not candidate_outputs and (entities_from_ner or types_from_ner):
            log.debug(f"(__call__)entities_from_ner: {entities_from_ner}")
            entity_ids = self.get_entity_ids(entities_from_ner, entity_tags, probas, question)
            type_ids = self.get_entity_ids(types_from_ner, ["t" for _ in types_from_ner],
                                           [1.0 for _ in types_from_ner], question)
            log.debug(f"(__call__)entity_ids: {entity_ids} type_ids {type_ids}")
            self.template_nums = template_types
            log.debug(f"(__call__)self.template_nums: {self.template_nums}")
            if not self.syntax_structure_known:
                entity_ids = entity_ids[:3]
            candidate_outputs = self.sparql_template_parser(question_sanitized, entity_ids, type_ids, answer_types)
        return candidate_outputs, template_answer

    def get_entity_ids(self, entities: List[str], tags: List[str], probas: List[float], question: str) -> List[List[str]]:
        entity_ids, el_output = [], []
        try:
            el_output = self.entity_linker([entities], [tags], [probas], [[question]], [None], [None])
        except json.decoder.JSONDecodeError:
            log.info("not received output from entity linking")
        if el_output:
            if self.use_el_api_requester:
                el_output = el_output[0]
            if el_output:
                if isinstance(el_output[0], dict):
                    entity_ids = [entity_info.get("entity_ids", []) for entity_info in el_output]
                if isinstance(el_output[0], list):
                    entity_substr, entity_ids, *_ = el_output
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
        log.debug(f"query_parser, answer_types {answer_types}")
        candidate_outputs = []
        if isinstance(self.template_nums, str):
            self.template_nums = [self.template_nums]
        template_log_list = [str([num, self.template_queries[num]["query_template"]]) for num in self.template_nums]
        log.debug(f"(find_candidate_answers)self.template_nums: {' --- '.join(template_log_list)}")
        init_templates = []
        for template_num in self.template_nums:
            for num, template in self.template_queries.items():
                if (num == template_num and self.syntax_structure_known) or \
                        (template["template_num"] == template_num and not self.syntax_structure_known):
                    init_templates.append(template)
        templates = [template for template in init_templates if
                     (not self.syntax_structure_known and [len(entity_ids), len(type_ids)] == template[
                         "entities_and_types_num"])
                     or self.syntax_structure_known]
        if not templates:
            templates = [template for template in init_templates if
                         (not self.syntax_structure_known and [len(entity_ids), 0] == template[
                             "entities_and_types_num"])
                         or self.syntax_structure_known]
        if not templates:
            return candidate_outputs
        if rels_from_template is not None:
            query_template = {}
            for template in templates:
                if template["rel_dirs"] == rel_dirs_from_template:
                    query_template = template
            if query_template:
                candidate_outputs = self.query_parser(question, [query_template], entity_ids, type_ids, answer_types,
                                                      rels_from_template)
        else:
            candidate_outputs = []
            for priority in range(1, 3):
                pr_templates = [template for template in templates if template["priority"] == priority]
                candidate_outputs = self.query_parser(question, pr_templates, entity_ids, type_ids, answer_types,
                                                      rels_from_template)
                if candidate_outputs:
                    return candidate_outputs

            if not candidate_outputs:
                alt_template_nums = templates[0].get("alternative_templates", [])
                log.info(f"Using alternative templates {alt_template_nums}")
                alt_templates = [self.template_queries[num] for num in alt_template_nums]
                candidate_outputs = self.query_parser(question, alt_templates, entity_ids, type_ids, answer_types,
                                                      rels_from_template)
                if candidate_outputs:
                    return candidate_outputs

        log.debug("candidate_rels_and_answers:\n" + '\n'.join([str(output) for output in candidate_outputs[:5]]))
        return candidate_outputs

    def find_top_rels(self, question: str, entity_ids: List[List[str]], triplet_info: Tuple) -> List[Tuple[str, Any]]:
        ex_rels, entity_rel_conn = [], set()
        direction, source, rel_type, n_hop = triplet_info
        if source == "wiki":
            queries_list = list({(entity, direction, rel_type) for entity_id in entity_ids
                                 for entity in entity_id[:self.entities_to_leave]})
            entity_ids_list = [elem[0] for elem in queries_list]
            parser_info_list = ["find_rels" for i in range(len(queries_list))]
            ex_rels = self.wiki_parser(parser_info_list, queries_list)
            for ex_rels_elem, entity_id in zip(ex_rels, entity_ids_list):
                for rel in ex_rels_elem:
                    entity_rel_conn.add((entity_id, rel.split("/")[-1]))
            if self.use_wp_api_requester and ex_rels:
                ex_rels = [rel[0] for rel in ex_rels]
            ex_rels = list(set(itertools.chain.from_iterable(ex_rels)))
            if n_hop == "2-hop":
                queries_list = list({(entity, "backw", rel_type) for entity_id in entity_ids
                                     for entity in entity_id[:self.entities_to_leave]})
                entity_ids_list = [elem[0] for elem in queries_list]
                parser_info_list = ["find_rels" for i in range(len(queries_list))]
                ex_rels_backw = self.wiki_parser(parser_info_list, queries_list)
                for ex_rels_elem, entity_id in zip(ex_rels_backw, entity_ids_list):
                    for rel in ex_rels_elem:
                        entity_rel_conn.add((entity_id, rel.split("/")[-1]))
                ex_rels_backw = list(set(itertools.chain.from_iterable(ex_rels_backw)))
                ex_rels += ex_rels_backw
            if self.delete_rel_prefix:
                ex_rels = [rel.split('/')[-1] for rel in ex_rels]
        elif source in {"rank_list_1", "rel_list_1"}:
            ex_rels = self.rank_list_0
        elif source in {"rank_list_2", "rel_list_2"}:
            ex_rels = self.rank_list_1
        rels_with_scores = self.rel_ranker.rank_rels(question, ex_rels)
        if n_hop == "2-hop" and rels_with_scores and entity_ids and entity_ids[0]:
            rels_1hop = [rel for rel, score in rels_with_scores]
            queries_list = [(entity_ids[0], rels_1hop[:5])]
            parser_info_list = ["find_rels_2hop"]
            ex_rels_2hop = self.wiki_parser(parser_info_list, queries_list)
            if self.delete_rel_prefix:
                ex_rels_2hop = [rel.split('/')[-1] for rel in ex_rels_2hop]
            rels_12hop = list(set(rels_1hop + ex_rels_2hop))
            rels_with_scores = self.rel_ranker.rank_rels(question, rels_12hop)

        return rels_with_scores[:self.rels_to_leave], entity_rel_conn

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

    def query_parser(self, question, query_templates, entity_ids, type_ids, answer_types, rels_from_template):
        raise NotImplementedError
