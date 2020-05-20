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
from typing import Tuple, List, Any, Optional, Union

import re
import nltk

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.kbqa.template_matcher import TemplateMatcher
from deeppavlov.models.kbqa.entity_linking import EntityLinker
from deeppavlov.models.kbqa.wiki_parser import WikiParser
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.rel_ranking_bert_infer import RelRankerBertInfer
from deeppavlov.models.kbqa.utils import extract_year, extract_number, asc_desc, make_entity_combs, fill_query

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
                 entities_to_leave: int = 5,
                 rels_to_leave: int = 10,
                 rels_to_leave_2hop: int = 7,
                 return_answers: bool = False, **kwargs) -> None:
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
            rels_to_leave_2hop: how many relations to leave in 2-hop questions
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
        self.entities_to_leave = entities_to_leave
        self.rels_to_leave = rels_to_leave
        self.rels_to_leave_2hop = rels_to_leave_2hop
        self.return_answers = return_answers
        self.template_queries = {
            0: ("SELECT ?obj WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?obj . ?s ?p ?x filter(contains(?x, N)&&contains(?p, 'qualifier')) }", (1, 0, 0), True),
            1: ("SELECT ?value WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?x filter(contains(?x, N)) . ?s ?p ?value filter(contains(?p, 'qualifier')) }", (1, 0, 0), True),
            2: ("SELECT ?value WHERE { wd:E1 p:R1 ?s . ?s ps:R1 wd:E2 . ?s ?p ?value filter(contains(?p, 'qualifier')) }", (1, 0, 0), True),
            3: ("SELECT ?obj WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?obj . ?s ?p wd:E2 filter(contains(?p, 'qualifier')) }", (1, 0, 0), True),
            4: ("SELECT (COUNT(?obj) AS ?value ) { wd:E1 wdt:R1 ?obj }", (1,), True),
            5: ("SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 ?obj } ORDER BY ASC(?obj) LIMIT 5", (0, 2), True),
            6: ("SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 ?obj . ?ent wdt:R2 wd:E1 } ORDER BY ASC(?obj) LIMIT 5", (0, 2, 3), True),
            8: ("SELECT ?ent WHERE { ?ent_mid wdt:P31 wd:T1 . ?ent wdt:R1 ?obj . ?ent_mid wdt:R2 ?ent } ORDER BY ASC(?obj) LIMIT 5", (0, 2, 3), True)
        }
        self.two_hop_queries = {
            (2, 0): {("forw", "forw"): ("SELECT ?ent WHERE { wd:E1 wdt:R1 ?ent . wd:E2 wdt:R2 ?ent }", (1, 1), False),
                     ("forw", "backw"): ("SELECT ?ent WHERE { wd:E1 wdt:R1 ?ent . ?ent wdt:R2 wd:E2 }", (1, 1), False)},
            (1, 1): {("forw",): ("SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . wd:E1 wdt:R1 ?ent }", (0, 1), False),
                     ("backw",): ("SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 wd:E1 }", (0, 1), False)},
            (1, 0): {("forw",): ("SELECT ?ent WHERE { wd:E1 wdt:R1 ?ent }", (1,), False),
                     ("backw",): ("SELECT ?ent WHERE { ?ent wdt:R1 wd:E1 }", (1,), False)}}
        
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
                 entities_from_ner_batch: List[List[str]],
                 types_from_ner_batch: List[List[str]]) -> List[Tuple[str]]:

        candidate_outputs_batch = []
        for question, template_type, entities_from_ner, types_from_ner in \
            zip(question_batch, template_type_batch, entities_from_ner_batch, types_from_ner_batch):

            candidate_outputs = []
            self.template_num = int(template_type)

            replace_tokens = [(' - ', '-'), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''),
                              (')', ''), ('–', '-')]
            for old, new in replace_tokens:
                question = question.replace(old, new)

            entities_from_template, types_from_template, rels_from_template, query_type_template = self.template_matcher(question)
            if query_type_template.isdigit():
                self.template_num = int(query_type_template)

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

                candidate_outputs = self.find_candidate_answers(question, entity_ids, type_ids, rels_from_template)

            if not candidate_outputs and entities_from_ner:
                log.debug(f"(__call__)entities_from_ner: {entities_from_ner}")
                log.debug(f"(__call__)types_from_ner: {types_from_ner}")
                entity_ids = self.get_entity_ids(entities_from_ner, "entities")
                type_ids = self.get_entity_ids(types_from_ner, "types")
                log.debug(f"(__call__)entity_ids: {entity_ids}")
                log.debug(f"(__call__)type_ids: {type_ids}")
                self.template_num = int(template_type[0])
                if self.template_num == 6 and len(type_ids) == 2:
                    self.template_num = 7
                log.debug(f"(__call__)self.template_num: {self.template_num}")
                candidate_outputs = self.find_candidate_answers(question, entity_ids, type_ids, rels_from_template=None)
            candidate_outputs_batch.append(candidate_outputs)
        if self.return_answers:
            answers = self.rel_ranker(question_batch, candidate_outputs_batch)
            log.debug(f"(__call__)answers: {answers}")
            return answers
        else:
            log.debug(f"(__call__)candidate_outputs_batch: {candidate_outputs_batch}")
            return candidate_outputs_batch

    def get_entity_ids(self, entities: List[str], what_to_link: str) -> List[List[str]]:
        entity_ids = []
        for entity in entities:
            if what_to_link == "entities":
                entity_id, confidences = self.linker_entities(entity)
            if what_to_link == "types":
                entity_id, confidences = self.linker_types(entity)
            entity_ids.append(entity_id[:15])
        return entity_ids


    def find_candidate_answers(self, question: str,
                               entity_ids: List[List[str]],
                               type_ids: List[List[str]],
                               rels_from_template: List[Tuple[str]]) -> List[Tuple[str]]:
        candidate_outputs = []
        log.debug(f"(find_candidate_answers)self.template_num: {self.template_num}")

        if self.template_num != 7:
            template = self.template_queries[self.template_num]
            candidate_outputs = self.query_parser(question, template,
                        entity_ids, type_ids, rels_from_template)

        if self.template_num == 7:
            templates = self.two_hop_queries[(len(entity_ids), len(type_ids))]
            if rels_from_template is not None:
                rel_dirs = tuple([rel_list[-1] for rel_list in rels_from_template])
                template = templates[rel_dirs]
                candidate_outputs = candidate_outputs = self.query_parser(question, template,
                        entity_ids, type_ids, rels_from_template)
            else:
                for rel_dirs in templates:
                    candidate_outputs = candidate_outputs = self.query_parser(question, templates[rel_dirs],
                        entity_ids, type_ids, rels_from_template)
                    if candidate_outputs:
                        break

        log.debug("candidate_rels_and_answers:\n" + '\n'.join([str(output) for output in candidate_outputs]))

        return candidate_outputs
    
    def query_parser(self, question, query_info, entity_ids, type_ids, rels_from_template):
        candidate_outputs = []
        question_tokens = nltk.word_tokenize(question)
        query, rels_for_search, return_if_found = query_info
        print("query", query, rels_for_search, return_if_found)
        query_triplets = query[query.find('{')+1:query.find('}')].strip(' ').split(' . ')
        print("query_triplets", query_triplets)
        query_triplets = [tuple(triplet.split(' ')[:3]) for triplet in query_triplets]
        known_query_triplets = [triplet for triplet in query_triplets if not all([elem.startswith('?') for elem in triplet])]
        unknown_query_triplets = [triplet for triplet in query_triplets if all([elem.startswith('?') for elem in triplet])]
        print("known_query_triplets", known_query_triplets)
        print("unknown_query_triplets", unknown_query_triplets)
        rel_directions = [("forw" if triplet[2].startswith('?') else "backw", search_or_not) \
            for search_or_not, triplet in zip(rels_for_search, query_triplets) if search_or_not]
        print("rel_directions", rel_directions)
        entity_combs = make_entity_combs(entity_ids, permut=True)
        print("entity_combs", entity_combs[:3])
        type_combs = make_entity_combs(type_ids, permut=False)
        print("type_combs", type_combs[:3])
        rels = []
        if rels_from_template is not None:
            rels = [rel_list[:-1] for rel_list in rels_from_template]
        else:
            rels = [self.find_top_rels(question, entity_ids, d) for d in rel_directions]
        print("rels", rels)
        rels_from_query = [triplet[1] for triplet in query_triplets if triplet[1].startswith('?')]
        print("rels_from_query", rels_from_query)
        answer_ent = re.findall("SELECT [\(]?([\S]+) ", query)
        order_from_query = re.findall("ORDER BY ([A-Z]{3,4})\((.*)\)", query)
        ascending = asc_desc(question)
        print("question, ascending", question, ascending)
        if not ascending:
            order_from_query = [("DESC", elem[1]) for elem in order_from_query]
        print("answer_ent", answer_ent, "order_from_query", order_from_query)
        filter_from_query = re.findall("contains\((.*)\) ", query)
        print("filter_from_query", filter_from_query)

        year = extract_year(question_tokens, question)
        number = extract_number(question_tokens, question)
        if year:
            filter_from_query = [elem.replace("N", year) for elem in filter_from_query]
        if number:
            filter_from_query = [elem.replace("N", number) for elem in filter_from_query]
        print("filter_from_query", filter_from_query)

        rel_combs = make_entity_combs(rels, permut=False)
        for entity_comb in entity_combs:
            for type_comb in type_combs:
                for rel_comb in rel_combs:
                    query_hdt = fill_query(known_query_triplets, entity_comb, type_comb, rel_comb)
                    print("query_hdt", query_hdt)
                    candidate_output = self.wiki_parser(rels_from_query + answer_ent, query_hdt, 
                        unknown_query_triplets, filter_from_query, order_from_query)
                    candidate_outputs += [rel_comb[:-1] + tuple(output) for output in candidate_output]
                    print("outputs", candidate_outputs)
                    if return_if_found and candidate_output:
                        return candidate_outputs
        print("final outputs", candidate_outputs)

        return candidate_outputs

    def find_top_rels(self, question, entity_ids, triplet_direction):
        if triplet_direction[1] == 1:
            ex_rels = []
            for entity_id in entity_ids:
                for entity in entity_id[:self.entities_to_leave]:
                    ex_rels += self.wiki_parser.find_rels(entity, triplet_direction[0])
            ex_rels = list(set(ex_rels))
            ex_rels = [rel.split('/')[-1] for rel in ex_rels]
        if triplet_direction[1] == 2:
            ex_rels = self.rank_list_0
        if triplet_direction[1] == 3:
            ex_rels = self.rank_list_1
        scores = self.rel_ranker.rank_rels(question, ex_rels)
        top_rels = [score[0] for score in scores]
        return top_rels
