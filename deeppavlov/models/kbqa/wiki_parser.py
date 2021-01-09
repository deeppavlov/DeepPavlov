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

import datetime
import random
import re
from logging import getLogger
from typing import List, Tuple, Dict, Any, Union
from collections import namedtuple

from hdt import HDTDocument

from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register('wiki_parser')
class WikiParser:
    """This class extract relations, objects or triplets from Wikidata HDT file"""

    def __init__(self, wiki_filename: str,
                       file_format: str = "hdt",
                       prefixes: Dict[str, Union[str, Dict[str, str]]] = {
                           "entity": "http://we",
                           "label": "http://wl",
                           "alias": "http://wal",
                           "description": "http://wd",
                           "rels": {"direct": "http://wpd",
                                    "no_type": "http://wp",
                                    "statement": "http://wps",
                                    "qualifier": "http://wpq"
                                   },
                           "statement": "http://ws"
                       },
                       dialog_mode: bool = False,
                       lang: str = "@en", **kwargs) -> None:
        """

        Args:
            wiki_filename: file with Wikidata
            file_format: format of Wikidata file
            lang: Russian or English language
            **kwargs:
        """
        
        self.prefixes = prefixes
        self.file_format = file_format
        self.wiki_filename = str(expand_path(wiki_filename))
        if self.file_format == "hdt":
            self.document = HDTDocument(self.wiki_filename)
        elif self.file_format == "pickle":
            self.document = load_pickle(self.wiki_filename)
            self.parsed_document = {}
        else:
            raise ValueError("Unsupported file format")
        
        self.grounded_entities_set = set()
        self.connected_entities_set = set()
        self.lang = lang

    def __call__(self, parser_info_list: List[str], queries_list: List[Any]) -> List[Any]:
        wiki_parser_output = []
        query_answer_types = []
        for parser_info, query in zip(parser_info_list, queries_list):
            if parser_info == "query_execute":
                what_return, query_seq, filter_info, order_info, answer_types, return_if_found = query
                if answer_types:
                    query_answer_types = answer_types
                candidate_output = {}
                try:
                    candidate_output = self.execute(what_return, query_seq, filter_info, order_info,
                                                                 query_answer_types)
                except:
                    log.info("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(candidate_output)
                if return_if_found and candidate_output:
                    return wiki_parser_output
            elif parser_info == "find_rels":
                rels = []
                try:
                    rels = self.find_rels(*query)
                except:
                    log.info("Wrong arguments are passed to wiki_parser")
                wiki_parser_output += rels
            elif parser_info == "find_label":
                label = ""
                try:
                    label = self.find_label(*query)
                except:
                    log.info("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(label)
            elif parser_info == "find_types":
                types = []
                try:
                    types = self.find_types(query)
                except:
                    log.info("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(types)
            elif parser_info == "retrieve_paths":
                paths = []
                try:
                    paths = self.retrieve_paths(*query)
                except:
                    log.info("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(paths)
            elif parser_info == "add_entities_to_dialogue_subgraph":
                log.debug(f"adding entities to dialogue subgraph, {query}")
                for entity in query:
                    self.grounded_entities_set.add(entity)
                    if self.file_format == "hdt":
                        tr, cnt = self.document.search_triples(f"{self.prefixes['entity']}/{entity}", "", "")
                        if cnt < 10000:
                            for triplet in tr:
                                obj = triplet[2].split('/')[-1].split("^^")[0].strip('"')
                                if re.findall("Q[\d]+", obj):
                                    self.connected_entities_set.add(obj)
                        tr, cnt = self.document.search_triples("", "", f"{self.prefixes['entity']}/{entity}")
                        if cnt < 10000:
                            for triplet in tr:
                                obj = triplet[0].split('/')[-1].split("^^")[0].strip('"')
                                if re.findall("Q[\d]+", obj):
                                    self.connected_entities_set.add(obj)
                    if self.file_format == "pickle":
                        triplets = self.document.get(entity, {})
                        if triplets:
                            if "forw" in triplets:
                                uncompressed_triplets = self.uncompress(triplets["forw"])
                                for triplet in uncompressed_triplets:
                                    for obj in triplet[1:]:
                                        self.connected_entities_set.add(obj)
                            if "backw" in triplets:
                                uncompressed_triplets = self.uncompress(triplets["backw"])
                                for triplet in uncompressed_triplets:
                                    for obj in triplet[1:]:
                                        self.connected_entities_set.add(obj)
            elif parser_info == "find_entities_in_subgraph":
                entities = set(query).intersection(self.connected_entities_set)
                if not entities:
                    entities = query
                log.debug(f"found intersection with subgraph, {entities}")
                wiki_parser_output.append(list(entities))
            elif parser_info == "retrieve_grounded_entities":
                log.debug(f"retrieve grounded entities, {self.grounded_entities_set}")
                wiki_parser_output.append(list(self.grounded_entities_set))
            elif parser_info == "find_triplets":
                if self.file_format == "hdt":
                    triplets = []
                    try:
                        triplets_forw, c = self.document.search_triples(f"{self.prefixes['entity']}/{query}", "", "")
                        triplets.extend([triplet for triplet in triplets_forw
                                             if not triplet[2].startswith(self.prefixes["statement"])])
                        triplets_backw, c = self.document.search_triples("", "", f"{self.prefixes['entity']}/{query}")
                        triplets.extend([triplet for triplet in triplets_forw
                                             if not triplet[0].startswith(self.prefixes["statement"])])
                    except:
                        log.info("Wrong arguments are passed to wiki_parser")
                    wiki_parser_output.append(list(triplets))
                else:
                    triplets = {}
                    try:
                        triplets = self.document.get(query, {})
                    except:
                        log.info("Wrong arguments are passed to wiki_parser")
                    uncompressed_triplets = {}
                    if triplets:
                        if "forw" in triplets:
                            uncompressed_triplets["forw"] = self.uncompress(triplets["forw"])
                        if "backw" in triplets:
                            uncompressed_triplets["backw"] = self.uncompress(triplets["backw"])
                    wiki_parser_output.append(uncompressed_triplets)
            elif parser_info == "parse_triplets" and self.file_format == "pickle":
                for entity in query:
                    self.parse_triplets(entity)
                wiki_parser_output.append("ok")
            else:
                raise ValueError("Unsupported query type")
        
        return wiki_parser_output

    def execute(self, what_return: List[str],
                query_seq: List[List[str]],
                filter_info: List[Tuple[str]] = None,
                order_info: namedtuple = None,
                answer_types: List[str] = None) -> List[List[str]]:
        """
            Let us consider an example of the question "What is the deepest lake in Russia?"
            with the corresponding SPARQL query            
            "SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 ?obj . ?ent wdt:R2 wd:E1 } ORDER BY ASC(?obj) LIMIT 5"

            arguments:
                what_return: ["?obj"]
                query_seq: [["?ent", "http://www.wikidata.org/prop/direct/P17", "http://www.wikidata.org/entity/Q159"]
                            ["?ent", "http://www.wikidata.org/prop/direct/P31", "http://www.wikidata.org/entity/Q23397"],
                            ["?ent", "http://www.wikidata.org/prop/direct/P4511", "?obj"]]
                filter_info: []
                order_info: order_info(variable='?obj', sorting_order='asc')
        """
        extended_combs = []
        combs = []
        
        for n, query in enumerate(query_seq):
            unknown_elem_positions = [(pos, elem) for pos, elem in enumerate(query) if elem.startswith('?')]
            """
                n = 0, query = ["?ent", "http://www.wikidata.org/prop/direct/P17",
                                                                            "http://www.wikidata.org/entity/Q159"]
                       unknown_elem_positions = ["?ent"]
                n = 1, query = ["?ent", "http://www.wikidata.org/prop/direct/P31",
                                                                        "http://www.wikidata.org/entity/Q23397"]
                       unknown_elem_positions = [(0, "?ent")]
                n = 2, query = ["?ent", "http://www.wikidata.org/prop/direct/P4511", "?obj"]
                       unknown_elem_positions = [(0, "?ent"), (2, "?obj")]
            """
            if n == 0:
                combs = self.search(query, unknown_elem_positions)
                # combs = [{"?ent": "http://www.wikidata.org/entity/Q5513"}, ...]
            else:
                if combs:
                    known_elements = []
                    extended_combs = []
                    if query[0].startswith("?") and query[2].startswith("?"):
                        for elem in query:
                            if elem in combs[0].keys():
                                known_elements.append(elem)
                        for comb in combs:
                            """
                                n = 1
                                query = ["?ent", "http://www.wikidata.org/prop/direct/P31",
                                                                            "http://www.wikidata.org/entity/Q23397"]
                                comb = {"?ent": "http://www.wikidata.org/entity/Q5513"}
                                known_elements = ["?ent"], known_values = ["http://www.wikidata.org/entity/Q5513"]
                                filled_query = ["http://www.wikidata.org/entity/Q5513", 
                                                "http://www.wikidata.org/prop/direct/P31", 
                                                "http://www.wikidata.org/entity/Q23397"]
                                new_combs = [["http://www.wikidata.org/entity/Q5513", 
                                              "http://www.wikidata.org/prop/direct/P31", 
                                              "http://www.wikidata.org/entity/Q23397"], ...]
                                extended_combs = [{"?ent": "http://www.wikidata.org/entity/Q5513"}, ...]
                            """
                            known_values = [comb[known_elem] for known_elem in known_elements]
                            for known_elem, known_value in zip(known_elements, known_values):
                                filled_query = [elem.replace(known_elem, known_value) for elem in query]
                                new_combs = self.search(filled_query, unknown_elem_positions)
                                for new_comb in new_combs:
                                    extended_combs.append({**comb, **new_comb})
                    else:
                        new_combs = self.search(query, unknown_elem_positions)
                        for comb in combs:
                            for new_comb in new_combs:
                                extended_combs.append({**comb, **new_comb})
                combs = extended_combs

        if combs:
            if filter_info:
                for filter_elem, filter_value in filter_info:
                    if filter_value == "qualifier":
                        filter_value = "wpq/"
                    combs = [comb for comb in combs if filter_value in comb[filter_elem]]

            if order_info and not isinstance(order_info, list) and order_info.variable is not None:
                reverse = True if order_info.sorting_order == "desc" else False
                sort_elem = order_info.variable
                for i in range(len(combs)):
                    value_str = combs[i][sort_elem].split('^^')[0].strip('"')
                    fnd = re.findall("[\d]{3,4}-[\d]{1,2}-[\d]{1,2}")
                    if fnd:
                        combs[i][sort_elem] = fnd[0]
                    else:
                        combs[i][sort_elem] = float(value_str)
                combs = sorted(combs, key=lambda x: x[sort_elem], reverse=reverse)
                combs = [combs[0]]

            if what_return[-1].startswith("count"):
                combs = [[combs[0][key] for key in what_return[:-1]] + [len(combs)]]
            else:
                combs = [[elem[key] for key in what_return] for elem in combs]
            
            if answer_types:
                answer_types = set(answer_types)
                combs = [[entity for entity in comb
                              if answer_types.intersection(self.find_types(entity))] for comb in combs]
                combs = [comb for comb in combs if any([entity for entity in comb])]

        return combs

    def search(self, query: List[str], unknown_elem_positions: List[Tuple[int, str]]) -> List[Dict[str, str]]:
        query = list(map(lambda elem: "" if elem.startswith('?') else elem, query))
        subj, rel, obj = query
        if self.file_format == "hdt":
            triplets, c = self.document.search_triples(subj, rel, obj)
            if rel == self.prefixes["description"]:
                triplets = [triplet for triplet in triplets if triplet[2].endswith(self.lang)]
            combs = [{elem: triplet[pos] for pos, elem in unknown_elem_positions} for triplet in triplets]
        else:
            if subj:
                subj, triplets = self.find_triplets(subj, "forw")
                triplets = [[subj, triplet[0], obj] for triplet in triplets for obj in triplet[1:]]
            if obj:
                obj, triplets = self.find_triplets(obj, "backw")
                triplets = [[subj, triplet[0], obj] for triplet in triplets for subj in triplet[1:]]
            if rel:
                if rel == self.prefixes["description"]:
                    triplets = [triplet for triplet in triplets if triplet[1] == "descr_en"]
                else:
                    rel = rel.split('/')[-1]
                    triplets = [triplet for triplet in triplets if triplet[1] == rel]
            combs = [{elem: triplet[pos] for pos, elem in unknown_elem_positions} for triplet in triplets]
        return combs

    def find_label(self, entity: str, question: str) -> str:
        entity = str(entity).replace('"', '')
        if self.file_format == "hdt":
            if entity.startswith("Q") or entity.startswith("P"):
                # example: "Q5513"
                entity = f"{self.prefixes['entity']}/{entity}"
                # "http://www.wikidata.org/entity/Q5513"

            if entity.startswith(self.prefixes["entity"]):
                labels, c = self.document.search_triples(entity, self.prefixes["label"], "")
                # labels = [["http://www.wikidata.org/entity/Q5513", "http://www.w3.org/2000/01/rdf-schema#label",
                #                                                    '"Lake Baikal"@en'], ...]
                for label in labels:
                    if label[2].endswith(self.lang):
                        found_label = label[2].strip(self.lang).replace('"', '')
                        return found_label

            elif entity.endswith(self.lang):
                # entity: '"Lake Baikal"@en'
                entity = entity[:-3]
                return entity

            elif "^^" in entity:
                """
                    examples:
                        '"1799-06-06T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>' (date)
                        '"+1642"^^<http://www.w3.org/2001/XMLSchema#decimal>' (number)
                """
                entity = entity.split("^^")[0]
                for token in ["T00:00:00Z", "+"]:
                    entity = entity.replace(token, '')
                entity = self.format_date(entity, question)
                return entity

            elif entity.isdigit():
                return entity
        if self.file_format == "pickle":
            if entity:
                if entity.startswith("Q") or entity.startswith("P"):
                    triplets = self.document.get(entity, {}).get("forw", [])
                    triplets = self.uncompress(triplets)
                    for triplet in triplets:
                        if triplet[0] == "name_en":
                            return triplet[1]
                else:
                    entity = self.format_date(entity, question)
                    return entity

        return "Not Found"

    def format_date(self, entity, question):
        date_info = re.findall("([\d]{3,4})-([\d]{1,2})-([\d]{1,2})", entity)
        if date_info:
            year, month, day = date_info[0]
            if "how old" in question.lower():
                entity = datetime.datetime.now().year - int(year)
            elif day != "00":
                date = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                entity = date.strftime("%d %B %Y")
            else:
                entity = year
            return entity
        entity = entity.lstrip('+-')
        return entity

    def find_alias(self, entity: str) -> List[str]:
        aliases = []
        if entity.startswith(self.prefixes["entity"]):
            labels, cardinality = self.document.search_triples(entity, self.prefixes["alias"], "")
            aliases = [label[2].strip(self.lang).strip('"') for label in labels if label[2].endswith(self.lang)]
        return aliases

    def find_rels(self, entity: str, direction: str, rel_type: str = "no_type", save: bool = False) -> List[str]:
        rels = []
        if self.file_format == "hdt":
            if not rel_type:
                rel_type = "direct"
            if direction == "forw":
                query = [f"{self.prefixes['entity']}/{entity}", "", ""]
            else:
                query = ["", "", f"{self.prefixes['entity']}/{entity}"]
            triplets, c = self.document.search_triples(*query)

            start_str = f"{self.prefixes['rels'][rel_type]}/P"
            rels = {triplet[1] for triplet in triplets if triplet[1].startswith(start_str)}
            rels = list(rels)
        if self.file_format == "pickle":
            triplets = self.document.get(entity, {}).get(direction, [])
            triplets = self.uncompress(triplets)
            rels = [triplet[0] for triplet in triplets if triplet[0].startswith("P")]
        return rels
        
    def retrieve_paths(self, entity: str, paths: List[List[str]]) -> List[List[str]]:
        retrieved_paths = []
        for path in paths:
            rel_1 = path[0]
            if rel_1.startswith("~"):
                dir_1 = "backw"
            else:
                dir_1 = "forw"
            rel_1 = rel_1.strip("~")
            if self.file_format == "hdt":
                if dir_1 == "forw":
                    query_1 = [f"{self.prefixes['entity']}/{entity}", f"{self.prefixes['rels']['direct']}/{rel_1}", ""]
                else:
                    query_1 = ["", f"{self.prefixes['rels']['direct']}/{rel_1}", f"{self.prefixes['entity']}/{entity}"]
                tr, cnt = self.document.search_triples(*query_1)
                
                if dir_1 == "forw":
                    objects_1 = [triplet[2].split('/')[-1] for triplet in tr]
                else:
                    objects_1 = [triplet[0].split('/')[-1] for triplet in tr]
                    
                if objects_1:
                    if len(path) == 1:
                        chosen_obj = random.choice(objects_1)
                        if dir_1 == "forw":
                            retrieved_paths.append([self.find_label(rel_1, ""), self.find_label(chosen_obj, "")])
                        else:
                            retrieved_paths.append([self.find_label(chosen_obj, ""), self.find_label(rel_1, "")])
                    else:
                        cur_paths = []
                        rel_2 = path[1]
                        if rel_2.startswith("~"):
                            dir_2 = "backw"
                        else:
                            dir_2 = "forw"
                        rel_2 = rel_2.strip("~")
                        for obj_1 in objects_1:
                            if obj_1.startswith("Q"):
                                if dir_2 == "forw":
                                    query_2 = [f"{self.prefixes['entity']}/{obj_1}", f"{self.prefixes['rels']['direct']}/{rel_2}", ""]
                                else:
                                    query_2 = ["", f"{self.prefixes['rels']['direct']}/{rel_2}", f"{self.prefixes['entity']}/{obj_1}"]
                                tr, cnt = self.document.search_triples(*query_2)
                                
                                if dir_2 == "forw":
                                    objects_2 = [triplet[2].split('/')[-1] for triplet in tr]
                                else:
                                    objects_2 = [triplet[0].split('/')[-1] for triplet in tr]
            
                                if objects_2:
                                    chosen_obj = random.choice(objects_2)
                                    if dir_1 == "forw" and dir_2 == "forw":
                                        cur_paths.append([self.find_label(rel_1, ""), self.find_label(obj_1, ""),
                                                                self.find_label(rel_2, ""), self.find_label(chosen_obj, "")])
                                    elif dir_1 == "forw" and dir_2 == "backw":
                                        cur_paths.append([self.find_label(rel_1, ""), self.find_label(obj_1, ""),
                                                                self.find_label(chosen_obj, ""), self.find_label(rel_2, ""),
                                                                self.find_label(obj_1, "")])
                                    elif dir_1 == "backw" and dir_2 == "forw":
                                        cur_paths.append([self.find_label(obj_1, ""), self.find_label(rel_1, ""),
                                                                self.find_label(obj_1, ""), self.find_label(rel_2, ""),
                                                                self.find_label(chosen_obj, "")])
                                    else:
                                        cur_paths.append([self.find_label(obj_1, ""), self.find_label(rel_1, ""),
                                                                self.find_label(chosen_obj, ""), self.find_label(rel_2, ""),
                                                                self.find_label(obj_1, "")])
                        
                        retrieved_paths.append(random.choice(cur_paths))
                        
        return retrieved_paths
        
        
    def find_types(self, entity: str):
        types = []
        if self.file_format == "hdt":
            if not entity.startswith("http"):
                entity = f"{self.prefixes['entity']}/{entity}"
            tr, c = self.document.search_triples(entity, f"{self.prefixes['rels']['direct']}/P31", "")
            types = [triplet[2].split('/')[-1] for triplet in tr]
        if self.file_format == "pickle":
            entity = entity.split('/')[-1]
            triplets = self.document.get(entity, {}).get("forw", [])
            triplets = self.uncompress(triplets)
            for triplet in triplets:
                if triplet[0] == "P31":
                    types = triplet[1:]
        types = set(types)
        return types

    def uncompress(self, triplets: Union[str, List[List[str]]]) -> List[List[str]]:
        if isinstance(triplets, str):
            triplets = triplets.split('\t')
            triplets = [triplet.strip().split("  ") for triplet in triplets]
        return triplets

    def parse_triplets(self, entity):
        triplets = self.document.get(entity, {})
        for direction in ["forw", "backw"]:
            if direction in triplets:
                dir_triplets = triplets[direction]
                dir_triplets = self.uncompress(dir_triplets)
                if entity in self.parsed_document:
                    self.parsed_document[entity][direction] = dir_triplets
                else:
                    self.parsed_document[entity] = {direction: dir_triplets}

    def find_triplets(self, subj: str, direction: str) -> Tuple[str, List[List[str]]]:
        subj = subj.split('/')[-1]
        if subj in self.parsed_document:
            triplets = self.parsed_document.get(subj, {}).get(direction, [])
        else:
            triplets = self.document.get(subj, {}).get(direction, [])
            triplets = self.uncompress(triplets)
        return subj, triplets
