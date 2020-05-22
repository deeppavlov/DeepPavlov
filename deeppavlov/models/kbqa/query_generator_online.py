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
from typing import Tuple, List, Any

import nltk

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.kbqa.template_matcher import TemplateMatcher
from deeppavlov.models.kbqa.entity_linking import EntityLinker
from deeppavlov.models.kbqa.wiki_parser_online import WikiParserOnline, get_answer
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.utils import extract_year, extract_number, asc_desc, make_combs

log = getLogger(__name__)


@register('query_generator_online')
class QueryGeneratorOnline(Component, Serializable):
    """
        This class takes as input entity substrings, defines the template of the query and
        fills the slots of the template with candidate entities and relations.
        The class uses Wikidata query service
    """

    def __init__(self, template_matcher: TemplateMatcher,
                 linker: EntityLinker,
                 wiki_parser: WikiParserOnline,
                 rel_ranker: RelRankerInfer,
                 load_path: str,
                 rank_rels_filename_1: str,
                 rank_rels_filename_2: str,
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 10, **kwargs):
        """

        Args:
            template_matcher: component deeppavlov.models.kbqa.template_matcher
            linker: component deeppavlov.models.kbqa.entity_linking
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            rel_ranker: component deeppavlov.models.kbqa.rel_ranking_infer
            load_path: path to folder with wikidata files
            rank_rels_filename_1: file with list of rels for first rels in questions with ranking 
            rank_rels_filename_2: file with list of rels for second rels in questions with ranking
            entities_to_leave: how many entities to leave after entity linking
            rels_to_leave: how many relations to leave after relation ranking
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.template_matcher = template_matcher
        self.linker = linker
        self.wiki_parser = wiki_parser
        self.rel_ranker = rel_ranker
        self.rank_rels_filename_1 = rank_rels_filename_1
        self.rank_rels_filename_2 = rank_rels_filename_2
        self.entities_to_leave = entities_to_leave
        self.rels_to_leave = rels_to_leave
        self.load()

    def load(self) -> None:
        with open(self.load_path / self.rank_rels_filename_1, 'r') as fl1:
            lines = fl1.readlines()
            self.rank_list_0 = [line.split('\t')[0] for line in lines]

        with open(self.load_path / self.rank_rels_filename_2, 'r') as fl2:
            lines = fl2.readlines()
            self.rank_list_1 = [line.split('\t')[0] for line in lines]

    def save(self) -> None:
        pass

    def __call__(self, question_batch: List[str],
                 template_type_batch: List[str],
                 entities_from_ner_batch: List[List[str]]) -> List[Tuple[str]]:

        candidate_outputs_batch = []
        for question, template_type, entities_from_ner in \
                     zip(question_batch, template_type_batch, entities_from_ner_batch):
            candidate_outputs = []
            self.template_num = int(template_type)

            entities_from_template, rels_from_template, query_type_template = self.template_matcher(question)
            if query_type_template == "simple":
                self.template_num = 7

            log.debug(f"template_type {self.template_num}")

            if entities_from_template:
                entity_ids = self.get_entity_ids(entities_from_template)
                log.debug(f"entities_from_template {entities_from_template}")
                log.debug(f"rels_from_template {rels_from_template}")
                log.debug(f"entity_ids {entity_ids}")

                candidate_outputs = self.find_candidate_answers(question, entity_ids, rels_from_template)

            if not candidate_outputs and entities_from_ner:
                entity_ids = self.get_entity_ids(entities_from_ner)
                log.debug(f"entities_from_ner {entities_from_ner}")
                log.debug(f"entity_ids {entity_ids}")
                candidate_outputs = self.find_candidate_answers(question, entity_ids, rels_from_template=None)
            candidate_outputs_batch.append(candidate_outputs)

        return candidate_outputs_batch

    def get_entity_ids(self, entities: List[str]) -> List[List[str]]:
        entity_ids = []
        for entity in entities:
            entity_id, confidences = self.linker(entity)
            entity_ids.append(entity_id[:15])
        return entity_ids

    def find_candidate_answers(self, question: str,
                               entity_ids: List[List[str]],
                               rels_from_template: List[Tuple[str]]) -> List[Tuple[str]]:
        candidate_outputs = []

        if self.template_num == 0 or self.template_num == 1:
            candidate_outputs = self.complex_question_with_number_solver(question, entity_ids)

        if self.template_num == 2 or self.template_num == 3:
            candidate_outputs = self.complex_question_with_qualifier_solver(entity_ids)
            if not candidate_outputs:
                self.template_num = 7

        if self.template_num == 4:
            candidate_outputs = self.questions_with_count_solver(question, entity_ids)

        if self.template_num == 5:
            candidate_outputs = self.maxmin_one_entity_solver(question, entity_ids[0][:self.entities_to_leave])

        if self.template_num == 6:
            candidate_outputs = self.maxmin_two_entities_solver(question, entity_ids)

        if self.template_num == 7:
            candidate_outputs = self.two_hop_solver(question, entity_ids, rels_from_template)

        log.debug("candidate_rels_and_answers:\n" + '\n'.join([str(output) for output in candidate_outputs[:10]]))

        return candidate_outputs

    def complex_question_with_number_solver(self, question: str, entity_ids: List[List[str]]) -> List[Tuple[str]]:
        question_tokens = nltk.word_tokenize(question)
        year = extract_year(question_tokens, question)
        number = False
        if not year:
            number = extract_number(question_tokens, question)
        log.debug(f"year {year}, number {number}")

        candidate_outputs = []

        if year:
            if self.template_num == 0:
                template = "SELECT ?obj ?p1 ?p2 ?p3 WHERE {{ wd:{} ?p1 ?s . ?s ?p2 ?obj . ?s ?p3 ?x " + \
                           "filter(contains(YEAR(?x),'{}')) }}"
            if self.template_num == 1:
                template = "SELECT ?obj ?p1 ?p2 ?p3 WHERE {{ wd:{} ?p1 ?s . ?s ?p2 ?x " + \
                           "filter(contains(YEAR(?x),{})) . ?s ?p3 ?obj }}"
            candidate_outputs = self.find_relevant_subgraph_cqwn(template, entity_ids[0][:self.entities_to_leave], year)
        if number:
            if self.template_num == 0:
                template = "SELECT ?obj ?p1 ?p2 ?p3 WHERE {{ wd:{} ?p1 ?s . ?s ?p2 ?obj . ?s ?p3 ?x " + \
                           "filter(contains(?x,'{}')) }}"
            if self.template_num == 1:
                template = "SELECT ?obj ?p1 ?p2 ?p3 WHERE {{ wd:{} ?p1 ?s . ?s ?p2 ?x " + \
                           "filter(contains(?x,{})) . ?s ?p3 ?obj }}"

            candidate_outputs = self.find_relevant_subgraph_cqwn(template, entity_ids[0][:self.entities_to_leave],
                                                                 number)

        return candidate_outputs

    def complex_question_with_qualifier_solver(self, entity_ids: List[List[str]]) -> List[Tuple[str]]:
        candidate_outputs = []

        if len(entity_ids) > 1:
            ent_combs = make_combs(entity_ids)
            if self.template_num == 2:
                query = "SELECT ?obj ?p1 ?p2 ?p3 WHERE {{ wd:{} ?p1 ?s . ?s ?p2 wd:{} . ?s ?p3 ?obj }}"

            if self.template_num == 3:
                query = "SELECT ?obj ?p1 ?p2 ?p3 WHERE {{ wd:{} ?p1 ?s . ?s ?p2 ?obj . ?s ?p3 wd:{} }}"

            candidate_outputs = self.find_relevant_subgraph_cqwq(query, ent_combs)

        return candidate_outputs

    def questions_with_count_solver(self, question: str, entity_ids: List[List[str]]) -> List[Tuple[str, Any]]:
        candidate_outputs = []

        ex_rels = []
        for entity_id in entity_ids:
            for entity in entity_id[:self.entities_to_leave]:
                ex_rels += self.wiki_parser("rels", "forw", entity)
                ex_rels += self.wiki_parser("rels", "backw", entity)

        ex_rels = list(set(ex_rels))
        scores = self.rel_ranker.rank_rels(question, ex_rels)
        top_rels = [score[0] for score in scores]
        log.debug(f"top scored rels: {top_rels}")
        for entity_id in entity_ids:
            for entity in entity_id[:self.entities_to_leave]:
                for rel in top_rels[:self.rels_to_leave]:
                    template = "SELECT (COUNT(?obj) AS ?value) {{ wd:{} wdt:{} ?obj }}"
                    query = template.format(entity, rel)
                    answers = get_answer(query)
                    if answers:
                        candidate_outputs.append((rel, answers[0]["value"]["value"]))
                        return candidate_outputs
                    else:
                        template = "SELECT (COUNT(?obj) AS ?value) {{ ?obj wdt:{} wd:{} }}"
                        query = template.format(rel, entity)
                        answers = get_answer(query)
                        if answers:
                            candidate_outputs.append((rel, answers[0]["value"]["value"]))
                            return candidate_outputs

        return candidate_outputs

    def maxmin_one_entity_solver(self, question: str, entities_list: List[str]) -> List[Tuple[str, str]]:
        scores = self.rel_ranker.rank_rels(question, self.rank_list_0)
        top_rels = [score[0] for score in scores]
        rel_list = ["?p = wdt:{}".format(rel) for rel in top_rels]
        rel_str = " || ".join(rel_list)
        log.debug(f"top scored rels: {top_rels}")
        ascending = asc_desc(question)
        if ascending:
            query = "SELECT ?ent ?p WHERE {{ ?ent wdt:P31 wd:{} . ?ent ?p ?obj . FILTER({}) }} " + \
                    "ORDER BY ASC(?obj)LIMIT 1"
        else:
            query = "SELECT ?ent ?p WHERE {{ ?ent wdt:P31 wd:{} . ?ent ?p ?obj . FILTER({}) }} " + \
                    "ORDER BY DESC(?obj)LIMIT 1"
        candidate_outputs = self.find_relevant_subgraph_maxmin_one(query, entities_list, rel_str)

        return candidate_outputs

    def maxmin_two_entities_solver(self, question: str, entity_ids: List[List[str]]) -> List[Tuple[str]]:
        ex_rels = []
        for entities_list in entity_ids:
            for entity in entities_list:
                ex_rels += self.wiki_parser("rels", "backw", entity)

        ex_rels = list(set(ex_rels))
        scores_1 = self.rel_ranker.rank_rels(question, ex_rels)
        top_rels_1 = [score[0] for score in scores_1]
        rel_list_1 = ["?p1 = wdt:{}".format(rel) for rel in top_rels_1]
        rel_str_1 = " || ".join(rel_list_1)
        log.debug(f"top scored first rels: {top_rels_1}")

        scores_2 = self.rel_ranker.rank_rels(question, self.rank_list_1)
        top_rels_2 = [score[0] for score in scores_2]
        rel_list_2 = ["?p2 = wdt:{}".format(rel) for rel in top_rels_2]
        rel_str_2 = " || ".join(rel_list_2)
        log.debug(f"top scored second rels: {top_rels_2}")

        candidate_outputs = []

        if len(entity_ids) > 1:
            ent_combs = make_combs(entity_ids)

            ascending = asc_desc(question)
            if ascending:
                query = "SELECT ?ent WHERE {{ ?ent wdt:P31 wd:{} . ?ent ?p1 ?obj . ?ent ?p2 wd:{} . " + \
                        "FILTER(({})&&({})) }} ORDER BY ASC(?obj)LIMIT 1"
            else:
                query = "SELECT ?ent WHERE {{ ?ent wdt:P31 wd:{} . ?ent ?p1 ?obj . ?ent ?p2 wd:{} . " + \
                        "FILTER(({})&&({})) }} ORDER BY DESC(?obj)LIMIT 1"

            candidate_outputs = self.find_relevant_subgraph_maxmin_two(query, ent_combs, rel_str_1, rel_str_2)

        return candidate_outputs

    def two_hop_solver(self, question: str,
                       entity_ids: List[List[str]],
                       rels_from_template: List[Tuple[str]] = None):
        candidate_outputs = []
        if len(entity_ids) == 1:
            if rels_from_template is not None:
                candidate_outputs = self.from_template_one_ent(entity_ids, rels_from_template)

            else:
                ex_rels = []
                for entity in entity_ids[0][:self.entities_to_leave]:
                    ex_rels += self.wiki_parser("rels", "forw", entity)
                    ex_rels += self.wiki_parser("rels", "backw", entity)

                ex_rels = list(set(ex_rels))
                scores = self.rel_ranker.rank_rels(question, ex_rels)
                top_rels = [score[0] for score in scores]
                rel_list_1 = ["?p1 = wdt:{}".format(rel) for rel in top_rels[:self.rels_to_leave]]
                rel_str_1 = " || ".join(rel_list_1)
                log.debug(f"top scored rels: {top_rels}")

                ex_rels_2 = []
                for entity in entity_ids[0][:self.entities_to_leave]:
                    template = "SELECT ?p2 WHERE {{ wd:{} ?p1 ?mid . ?mid ?p2 ?obj . FILTER({})}}"
                    query = template.format(entity, rel_str_1)
                    answers = get_answer(query)
                    if answers:
                        for ans in answers:
                            ex_rels_2.append(ans["p2"]["value"].split('/')[-1])
                    template = "SELECT ?p2 WHERE {{ ?mid ?p1 wd:{} . ?mid ?p2 ?obj . FILTER({})}}"
                    query = template.format(entity, rel_str_1)
                    answers = get_answer(query)
                    if answers:
                        for ans in answers:
                            ex_rels_2.append(ans["p2"]["value"].split('/')[-1])

                ex_rels_2 = list(set(ex_rels_2))
                scores_2 = self.rel_ranker.rank_rels(question, ex_rels_2)
                top_rels_2 = [score[0] for score in scores_2]
                rel_list_2 = ["?p2 = wdt:{}".format(rel) for rel in top_rels_2[:self.rels_to_leave]]
                rel_str_2 = " || ".join(rel_list_2)
                log.debug(f"top scored second rels: {top_rels_2}")

                for entity in entity_ids[0][:self.entities_to_leave]:
                    template = "SELECT ?p1 ?obj WHERE {{ wd:{} ?p1 ?obj . FILTER({})}}"
                    query = template.format(entity, rel_str_1)
                    answers_forw = get_answer(query)
                    template = "SELECT ?p1 ?obj WHERE {{ ?obj ?p1 wd:{} . FILTER({})}}"
                    query = template.format(entity, rel_str_1)
                    answers_backw = get_answer(query)
                    if answers_forw:
                        for ans in answers_forw:
                            p1 = ans["p1"]["value"].split('/')[-1]
                            obj = ans["obj"]["value"].split('/')[-1]
                            candidate_outputs.append((p1, obj))
                    if answers_backw:
                        for ans in answers_backw:
                            p1 = ans["p1"]["value"].split('/')[-1]
                            obj = ans["obj"]["value"].split('/')[-1]
                            candidate_outputs.append((p1, obj))

                for entity in entity_ids[0][:self.entities_to_leave]:
                    template = "SELECT ?p1 ?p2 ?obj WHERE {{ wd:{} ?p1 ?mid . ?mid ?p2 ?obj . FILTER(({}) && ({})) }}"
                    query = template.format(entity, rel_str_1, rel_str_2)
                    answers_forw = get_answer(query)
                    if answers_forw:
                        for ans in answers_forw:
                            p1 = ans["p1"]["value"].split('/')[-1]
                            p2 = ans["p2"]["value"].split('/')[-1]
                            obj = ans["obj"]["value"].split('/')[-1]
                            candidate_outputs.append((p1, p2, obj))
                    template = "SELECT ?p1 ?p2 ?obj WHERE {{ ?mid ?p1 wd:{} . ?mid ?p2 ?obj . FILTER(({}) && ({})) }}"
                    query = template.format(entity, rel_str_1, rel_str_2)
                    answers_backw = get_answer(query)
                    if answers_backw:
                        for ans in answers_backw:
                            p1 = ans["p1"]["value"].split('/')[-1]
                            p2 = ans["p2"]["value"].split('/')[-1]
                            obj = ans["obj"]["value"].split('/')[-1]
                            candidate_outputs.append((p1, p2, obj))

        if len(entity_ids) == 2:
            ent_combs = make_combs(entity_ids)
            if rels_from_template is not None:
                candidate_outputs = self.from_template_two_ent(ent_combs, rels_from_template)

            else:
                for ent_comb in ent_combs[:60]:
                    template = "SELECT ?p ?obj WHERE {{ wd:{} ?p ?obj . ?obj wdt:P31 wd:{} }}"
                    query = template.format(ent_comb[1], ent_comb[0])
                    answers_forw = get_answer(query)
                    if answers_forw:
                        for ans in answers_forw:
                            p = ans["p"]["value"].split('/')[-1]
                            obj = ans["obj"]["value"].split('/')[-1]
                            candidate_outputs.append((p, obj))
                        if candidate_outputs:
                            return candidate_outputs

                    template = "SELECT ?p ?obj WHERE {{ ?obj ?p wd:{} . ?obj wdt:P31 wd:{} }}"
                    query = template.format(ent_comb[1], ent_comb[0])
                    answers_backw = get_answer(query)
                    if answers_backw:
                        for ans in answers_backw:
                            p = ans["p"]["value"].split('/')[-1]
                            obj = ans["obj"]["value"].split('/')[-1]
                            candidate_outputs.append((p, obj))
                        if candidate_outputs:
                            return candidate_outputs

                    template = "SELECT ?p1 ?p2 ?obj WHERE {{ wd:{} ?p1 ?obj . ?obj ?p2 wd:{} }}"
                    query = template.format(ent_comb[1], ent_comb[0])
                    answers_forw = get_answer(query)
                    if answers_forw:
                        for ans in answers_forw:
                            p1 = ans["p1"]["value"].split('/')[-1]
                            p2 = ans["p2"]["value"].split('/')[-1]
                            obj = ans["obj"]["value"].split('/')[-1]
                            candidate_outputs.append((p1, p2, obj))
                        if candidate_outputs:
                            return candidate_outputs

        return candidate_outputs

    def find_relevant_subgraph_cqwn(self, template: str,
                                    entities_list: List[str],
                                    num: str) -> List[Tuple[str, str, str]]:
        candidate_outputs = []

        for entity in entities_list:
            query = template.format(entity, num)
            answers = get_answer(query)
            if answers:
                for ans in answers:
                    p1 = ans["p1"]["value"]
                    p2 = ans["p2"]["value"]
                    p3 = ans["p3"]["value"]
                    obj = ans["obj"]["value"]
                    if "statement" in p2 and "qualifier" in p3:
                        p1 = p1.split('/')[-1]
                        p3 = p3.split('/')[-1]
                        obj = obj.split('/')[-1]
                        candidate_outputs.append((p1, p3, obj))
                if candidate_outputs:
                    return candidate_outputs

        return candidate_outputs

    def find_relevant_subgraph_cqwq(self, template: str,
                                    ent_combs: List[Tuple[str, str, int]]) -> List[Tuple[str, str, str]]:
        candidate_outputs = []

        for ent_comb in ent_combs:
            query = template.format(ent_comb[0], ent_comb[1])
            answers = get_answer(query)
            if answers:
                for ans in answers:
                    p1 = ans["p1"]["value"]
                    p2 = ans["p2"]["value"]
                    p3 = ans["p3"]["value"]
                    obj = ans["obj"]["value"]
                    if (obj.startswith('Q') or obj.endswith("00:00:00Z") or obj.isdigit()) \
                            and "statement" in p2 and "qualifier" in p3:
                        p1 = p1.split('/')[-1]
                        p3 = p3.split('/')[-1]
                        obj = obj.split('/')[-1]
                        candidate_outputs.append((p1, p3, obj))
                if candidate_outputs:
                    return candidate_outputs

        return candidate_outputs

    def find_relevant_subgraph_maxmin_one(self, template: str,
                                          entities_list: List[str],
                                          rel_str: str) -> List[Tuple[str, str]]:
        candidate_answers = []

        for entity in entities_list:
            query = template.format(entity, rel_str)
            answers = get_answer(query)
            if answers:
                for answer in answers:
                    ent = answer["ent"]["value"].split('/')[-1]
                    p = answer["p"]["value"].split('/')[-1]
                    candidate_answers.append((p, ent))
                if candidate_answers:
                    return candidate_answers

        return candidate_answers

    def find_relevant_subgraph_maxmin_two(self, template: str,
                                          ent_combs: List[Tuple[str, str, int]],
                                          rel_str_1: str,
                                          rel_str_2: str) -> List[Tuple[str, str, str]]:
        candidate_answers = []

        for ent_comb in ent_combs:
            query = template.format(ent_comb[0], ent_comb[1], rel_str_1, rel_str_2)
            answers = get_answer(query)
            if answers:
                for answer in answers:
                    ent = answer["ent"]["value"].split('/')[-1]
                    p1 = answer["p1"]["value"].split('/')[-1]
                    p2 = answer["p2"]["value"].split('/')[-1]
                    candidate_answers.append((p1, p2, ent))
                if candidate_answers:
                    return candidate_answers

        return candidate_answers

    def from_template_one_ent(self, entity_ids: List[List[str]],
                              rels_from_template: List[Tuple[str]]) -> List[Tuple[str, str]]:
        candidate_outputs = []
        if len(rels_from_template) == 1:
            relation = rels_from_template[0][0]
            direction = rels_from_template[0][1]
            for entity in entity_ids[0]:
                objects = self.wiki_parser("objects", direction, entity, relation)
                if objects:
                    candidate_outputs.append((relation, objects[0]))
                    return candidate_outputs

        if len(rels_from_template) == 2:
            relation_1 = rels_from_template[0][0]
            direction_1 = rels_from_template[0][1]
            relation_2 = rels_from_template[1][0]
            direction_2 = rels_from_template[1][1]
            for entity in entity_ids[0]:
                objects_1 = self.wiki_parser("objects", direction_1, entity, relation_1)
                for object_1 in objects_1:
                    objects_2 = self.wiki_parser("objects", direction_2, object_1, relation_2)
                    if objects_2:
                        for object_2 in objects_2:
                            candidate_outputs.append((relation_1, relation_2, object_2))
                            return candidate_outputs

        return candidate_outputs

    def from_template_two_ent(self, ent_combs: List[Tuple[str, str, int]],
                              rels_from_template: List[Tuple[str]]) -> List[Tuple[str, str]]:
        candidate_outputs = []
        if len(rels_from_template) == 1:
            relation = rels_from_template[0][0]
            direction = rels_from_template[0][1]
            for ent_comb in ent_combs:
                objects_1 = self.wiki_parser("objects", direction, ent_comb[1], relation)
                if objects_1:
                    for object_1 in objects_1:
                        objects_2 = self.wiki_parser("objects", direction, object_1, "P31", obj=ent_comb[0])
                        if objects_2:
                            candidate_outputs.append((relation, objects_2[0]))
                            return candidate_outputs

        if len(rels_from_template) == 2:
            relation_1 = rels_from_template[0][0]
            direction_1 = rels_from_template[0][1]
            relation_2 = rels_from_template[1][0]
            direction_2 = rels_from_template[1][1]
            for ent_comb in ent_combs:
                objects_1 = self.wiki_parser("objects", direction_1, ent_comb[0], relation_1)
                objects_2 = self.wiki_parser("objects", direction_2, ent_comb[1], relation_2)
                objects_intersect = list(set(objects_1) & set(objects_2))
                if objects_intersect:
                    return [(relation_1, relation_2, objects_intersect[0])]

        return candidate_outputs
