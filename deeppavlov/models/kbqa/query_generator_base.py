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
from typing import Tuple, List, Optional, Union, Any

from whapi import search, get_html
from bs4 import BeautifulSoup

from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.kbqa.template_matcher import TemplateMatcher
from deeppavlov.models.kbqa.entity_linking import EntityLinker
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.rel_ranking_bert_infer import RelRankerBertInfer

log = getLogger(__name__)


class QueryGeneratorBase(Component, Serializable):
    """
        This class takes as input entity substrings, defines the template of the query and
        fills the slots of the template with candidate entities and relations.
    """

    def __init__(self, template_matcher: TemplateMatcher,
                 linker_entities: EntityLinker,
                 linker_types: EntityLinker,
                 rel_ranker: Union[RelRankerInfer, RelRankerBertInfer],
                 load_path: str,
                 rank_rels_filename_1: str,
                 rank_rels_filename_2: str,
                 sparql_queries_filename: str,
                 wiki_parser=None,
                 wiki_file_format: str = "hdt",
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 7,
                 syntax_structure_known: bool = False,
                 use_api_requester: bool = False,
                 return_answers: bool = False, *args, **kwargs) -> None:
        """

        Args:
            template_matcher: component deeppavlov.models.kbqa.template_matcher
            linker_entities: component deeppavlov.models.kbqa.entity_linking for linking of entities
            linker_types: component deeppavlov.models.kbqa.entity_linking for linking of types
            rel_ranker: component deeppavlov.models.kbqa.rel_ranking_infer
            load_path: path to folder with wikidata files
            rank_rels_filename_1: file with list of rels for first rels in questions with ranking 
            rank_rels_filename_2: file with list of rels for second rels in questions with ranking
            sparql_queries_filename: file with sparql query templates
            wiki_file_format: format of wikidata file
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            entities_to_leave: how many entities to leave after entity linking
            rels_to_leave: how many relations to leave after relation ranking
            syntax_structure_known: if syntax tree parser was used to define query template type
            use_api_requester: whether deeppavlov.models.api_requester.api_requester component will be used for
                Entity Linking and Wiki Parser
            return_answers: whether to return answers or candidate answers
        """
        super().__init__(save_path=None, load_path=load_path)
        self.template_matcher = template_matcher
        self.linker_entities = linker_entities
        self.linker_types = linker_types
        self.wiki_parser = wiki_parser
        self.wiki_file_format = wiki_file_format
        self.rel_ranker = rel_ranker
        self.rank_rels_filename_1 = rank_rels_filename_1
        self.rank_rels_filename_2 = rank_rels_filename_2
        self.rank_list_0 = []
        self.rank_list_1 = []
        self.entities_to_leave = entities_to_leave
        self.rels_to_leave = rels_to_leave
        self.syntax_structure_known = syntax_structure_known
        self.use_api_requester = use_api_requester
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

        self.template_queries = read_json(str(expand_path(self.sparql_queries_filename)))

    def save(self) -> None:
        pass

    def find_candidate_answers(self, question: str,
                               question_sanitized: str,
                               template_types: Union[List[str], str],
                               entities_from_ner: List[str],
                               types_from_ner: List[str]) -> Union[List[Tuple[str, Any]], List[str]]:

        candidate_outputs = []
        self.template_nums = template_types

        replace_tokens = [(' - ', '-'), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''),
                          (')', ''), ('–', '-')]
        for old, new in replace_tokens:
            question = question.replace(old, new)

        entities_from_template, types_from_template, rels_from_template, rel_dirs_from_template, query_type_template, \
        entity_types, template_answer, template_found = self.template_matcher(question_sanitized, entities_from_ner)
        self.template_nums = [query_type_template]

        log.debug(f"question: {question}\n")
        log.debug(f"template_type {self.template_nums}")

        if entities_from_template or types_from_template:
            if rels_from_template[0][0] == "PHOW":
                how_to_content = self.find_answer_wikihow(entities_from_template[0])
                candidate_outputs = [["PHOW", how_to_content, 1.0]]
            else:
                entity_ids = self.get_entity_ids(entities_from_template, "entities", template_found, question,
                                                 entity_types)
                type_ids = self.get_entity_ids(types_from_template, "types")
                log.debug(f"entities_from_template {entities_from_template}")
                log.debug(f"entity_types {entity_types}")
                log.debug(f"types_from_template {types_from_template}")
                log.debug(f"rels_from_template {rels_from_template}")
                log.debug(f"entity_ids {entity_ids}")
                log.debug(f"type_ids {type_ids}")

                candidate_outputs = self.sparql_template_parser(question_sanitized, entity_ids, type_ids,
                                                                rels_from_template,
                                                                rel_dirs_from_template)

        if not candidate_outputs and entities_from_ner:
            log.debug(f"(__call__)entities_from_ner: {entities_from_ner}")
            log.debug(f"(__call__)types_from_ner: {types_from_ner}")
            entity_ids = self.get_entity_ids(entities_from_ner, "entities", question=question)
            type_ids = self.get_entity_ids(types_from_ner, "types")
            log.debug(f"(__call__)entity_ids: {entity_ids}")
            log.debug(f"(__call__)type_ids: {type_ids}")
            self.template_nums = template_types
            log.debug(f"(__call__)self.template_nums: {self.template_nums}")
            if not self.syntax_structure_known:
                entity_ids = entity_ids[:3]
            candidate_outputs = self.sparql_template_parser(question_sanitized, entity_ids, type_ids)
        return candidate_outputs, template_answer

    def get_entity_ids(self, entities: List[str],
                       what_to_link: str,
                       template_found: str = None,
                       question: str = None,
                       entity_types: List[List[str]] = None) -> List[List[str]]:
        entity_ids = []
        if what_to_link == "entities":
            if entity_types:
                el_output = self.linker_entities([entities], [template_found], [question], [entity_types])
            else:
                el_output = self.linker_entities([entities], [template_found], [question])
            if self.use_api_requester:
                el_output = el_output[0]
            entity_ids, _ = el_output
            if not self.use_api_requester and entity_ids:
                entity_ids = entity_ids[0]
        if what_to_link == "types":
            entity_ids, _ = self.linker_types([entities])
            entity_ids = entity_ids[0]

        return entity_ids

    def sparql_template_parser(self, question: str,
                               entity_ids: List[List[str]],
                               type_ids: List[List[str]],
                               rels_from_template: Optional[List[Tuple[str]]] = None,
                               rel_dirs_from_template: Optional[List[str]] = None) -> List[Tuple[str]]:
        candidate_outputs = []
        log.debug(f"(find_candidate_answers)self.template_nums: {self.template_nums}")
        templates = []
        for template_num in self.template_nums:
            for num, template in self.template_queries.items():
                if (num == template_num and self.syntax_structure_known) or \
                        (template["template_num"] == template_num and not self.syntax_structure_known):
                    templates.append(template)
        templates = [template for template in templates if
                     (not self.syntax_structure_known and [len(entity_ids), len(type_ids)] == template[
                         "entities_and_types_num"])
                     or self.syntax_structure_known]
        templates_string = '\n'.join([template["query_template"] for template in templates])
        log.debug(f"{templates_string}")
        if not templates:
            return candidate_outputs
        if rels_from_template is not None:
            query_template = {}
            for template in templates:
                if template["rel_dirs"] == rel_dirs_from_template:
                    query_template = template
            if query_template:
                entities_and_types_select = query_template["entities_and_types_select"]
                candidate_outputs = self.query_parser(question, query_template, entities_and_types_select,
                                                      entity_ids, type_ids, rels_from_template)
        else:
            for template in templates:
                entities_and_types_select = template["entities_and_types_select"]
                candidate_outputs = self.query_parser(question, template, entities_and_types_select,
                                                      entity_ids, type_ids, rels_from_template)
                if candidate_outputs:
                    return candidate_outputs

            if not candidate_outputs:
                alternative_templates = templates[0]["alternative_templates"]
                for template_num, entities_and_types_select in alternative_templates:
                    candidate_outputs = self.query_parser(question, self.template_queries[template_num],
                                                          entities_and_types_select, entity_ids, type_ids,
                                                          rels_from_template)
                    if candidate_outputs:
                        return candidate_outputs

        log.debug("candidate_rels_and_answers:\n" + '\n'.join([str(output) for output in candidate_outputs[:5]]))

        return candidate_outputs

    def find_top_rels(self, question: str, entity_ids: List[List[str]], triplet_info: Tuple) -> List[Tuple[str, Any]]:
        ex_rels = []
        direction, source, rel_type = triplet_info
        if source == "wiki":
            queries_list = list({(entity, direction, rel_type) for entity_id in entity_ids
                                 for entity in entity_id[:self.entities_to_leave]})
            parser_info_list = ["find_rels" for i in range(len(queries_list))]
            ex_rels = self.wiki_parser(parser_info_list, queries_list)
            if self.use_api_requester and ex_rels:
                ex_rels = [rel[0] for rel in ex_rels]
            ex_rels = list(set(ex_rels))
            ex_rels = [rel.split('/')[-1] for rel in ex_rels]
        elif source == "rank_list_1":
            ex_rels = self.rank_list_0
        elif source == "rank_list_2":
            ex_rels = self.rank_list_1
        rels_with_scores = self.rel_ranker.rank_rels(question, ex_rels)
        return rels_with_scores[:self.rels_to_leave]

    def find_answer_wikihow(self, howto_sentence: str) -> str:
        search_results = search(howto_sentence, 5)
        article_id = search_results[0]["article_id"]
        html = get_html(article_id)
        page = BeautifulSoup(html, 'lxml')
        tags = list(page.find_all(['p']))
        if tags:
            howto_content = f"{tags[0].text.strip()}@en"
        else:
            howto_content = "Not Found"
        return howto_content
