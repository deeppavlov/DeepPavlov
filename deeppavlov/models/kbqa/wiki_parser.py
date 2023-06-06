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
import re
from collections import namedtuple
from logging import getLogger
from typing import List, Tuple, Dict, Any, Union

from hdt import HDTDocument

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import load_pickle, read_json
from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register('wiki_parser')
class WikiParser:
    """This class extract relations, objects or triplets from Wikidata HDT file."""

    def __init__(self, wiki_filename: str,
                 file_format: str = "hdt",
                 prefixes: Dict[str, Union[str, Dict[str, str]]] = None,
                 rel_q2name_filename: str = None,
                 max_comb_num: int = 1e6,
                 lang: str = "@en", **kwargs) -> None:
        """

        Args:
            wiki_filename: file with Wikidata
            file_format: format of Wikidata file
            lang: Russian or English language
            **kwargs:
        """

        if prefixes is None:
            prefixes = {
                "entity": "http://we",
                "label": "http://wl",
                "alias": "http://wal",
                "description": "http://wd",
                "rels": {
                    "direct": "http://wpd",
                    "no_type": "http://wp",
                    "statement": "http://wps",
                    "qualifier": "http://wpq",
                    "type": "http://wpd/P31"
                },
                "statement": "http://ws"
            }
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
        self.used_rels = set()
        self.rel_q2name = dict()
        if rel_q2name_filename:
            if rel_q2name_filename.endswith("json"):
                self.rel_q2name = read_json(str(expand_path(rel_q2name_filename)))
            elif rel_q2name_filename.endswith("pickle"):
                self.rel_q2name = load_pickle(str(expand_path(rel_q2name_filename)))
            else:
                raise ValueError(f"Unsupported file format: {rel_q2name_filename}")

        self.max_comb_num = max_comb_num
        self.lang = lang
        self.replace_tokens = [('"', ''), (self.lang, " "), ('$', ' '), ('  ', ' ')]

    def __call__(self, parser_info_list: List[str], queries_list: List[Any]) -> List[Any]:
        wiki_parser_output = self.execute_queries_list(parser_info_list, queries_list)
        return wiki_parser_output

    def execute_queries_list(self, parser_info_list: List[str], queries_list: List[Any]):
        wiki_parser_output = []
        query_answer_types = []
        for parser_info, query in zip(parser_info_list, queries_list):
            if parser_info == "query_execute":
                answers, found_rels, found_combs = [], [], []
                try:
                    what_return, rels_from_query, query_seq, filter_info, order_info, answer_types, rel_types, \
                    return_if_found = query
                    if answer_types:
                        query_answer_types = answer_types
                    answers, found_rels, found_combs = \
                        self.execute(what_return, rels_from_query, query_seq, filter_info, order_info,
                                     query_answer_types, rel_types)
                except ValueError:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append([answers, found_rels, found_combs])
            elif parser_info == "find_rels":
                rels = []
                try:
                    rels = self.find_rels(*query)
                except:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(rels)
            elif parser_info == "find_rels_2hop":
                rels = []
                try:
                    rels = self.find_rels_2hop(*query)
                except ValueError:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output += rels
            elif parser_info == "find_object":
                objects = []
                try:
                    objects = self.find_object(*query)
                except:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(objects)
            elif parser_info == "check_triplet":
                check_res = False
                try:
                    check_res = self.check_triplet(*query)
                except:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(check_res)
            elif parser_info == "find_label":
                label = ""
                try:
                    label = self.find_label(*query)
                except:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(label)
            elif parser_info == "find_types":
                types = []
                try:
                    types = self.find_types(query)
                except:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(types)
            elif parser_info == "fill_triplets":
                filled_triplets = []
                try:
                    filled_triplets = self.fill_triplets(*query)
                except ValueError:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(filled_triplets)
            elif parser_info == "find_triplets":
                if self.file_format == "hdt":
                    triplets = []
                    try:
                        triplets_forw, c = self.document.search_triples(f"{self.prefixes['entity']}/{query}", "", "")
                        triplets.extend([triplet for triplet in triplets_forw
                                         if not triplet[2].startswith(self.prefixes["statement"])])
                        triplets_backw, c = self.document.search_triples("", "", f"{self.prefixes['entity']}/{query}")
                        triplets.extend([triplet for triplet in triplets_backw
                                         if not triplet[0].startswith(self.prefixes["statement"])])
                    except:
                        log.warning("Wrong arguments are passed to wiki_parser")
                    wiki_parser_output.append(list(triplets))
                else:
                    triplets = {}
                    try:
                        triplets = self.document.get(query, {})
                    except:
                        log.warning("Wrong arguments are passed to wiki_parser")
                    uncompressed_triplets = {}
                    if triplets:
                        if "forw" in triplets:
                            uncompressed_triplets["forw"] = self.uncompress(triplets["forw"])
                        if "backw" in triplets:
                            uncompressed_triplets["backw"] = self.uncompress(triplets["backw"])
                    wiki_parser_output.append(uncompressed_triplets)
            elif parser_info == "find_triplets_for_rel":
                found_triplets = []
                try:
                    found_triplets, c = \
                        self.document.search_triples("", f"{self.prefixes['rels']['direct']}/{query}", "")
                except:
                    log.warning("Wrong arguments are passed to wiki_parser")
                wiki_parser_output.append(list(found_triplets))
            elif parser_info == "parse_triplets" and self.file_format == "pickle":
                for entity in query:
                    self.parse_triplets(entity)
                wiki_parser_output.append("ok")
            else:
                raise ValueError("Unsupported query type")

        return wiki_parser_output

    def execute(self, what_return: List[str],
                rels_from_query: List[str],
                query_seq: List[List[str]],
                filter_info: List[Tuple[str]] = None,
                order_info: namedtuple = None,
                answer_types: List[str] = None,
                rel_types: List[str] = None):
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
        answers, found_rels, found_combs = [], [], []

        for n, (query, rel_type) in enumerate(zip(query_seq, rel_types)):
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
                combs, triplets = self.search(query, unknown_elem_positions, rel_type)
                # combs = [{"?ent": "http://www.wikidata.org/entity/Q5513"}, ...]
            else:
                if combs:
                    known_elements = []
                    extended_combs = []
                    if query[0].startswith("?"):
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
                            if comb:
                                known_values = [comb[known_elem] for known_elem in known_elements]
                                for known_elem, known_value in zip(known_elements, known_values):
                                    filled_query = [elem.replace(known_elem, known_value) for elem in query]
                                    new_combs, triplets = self.search(filled_query, unknown_elem_positions, rel_type)
                                    for new_comb in new_combs:
                                        extended_combs.append(self.merge_combs(comb, new_comb))
                    else:
                        new_combs, triplets = self.search(query, unknown_elem_positions, rel_type)
                        for comb in combs:
                            for new_comb in new_combs:
                                extended_combs.append(self.merge_combs(comb, new_comb))
                combs = extended_combs

        is_boolean = self.define_is_boolean(query_seq)
        if combs or is_boolean:
            if filter_info:
                for filter_elem, filter_value in filter_info:
                    if filter_value == "qualifier":
                        filter_value = "wpq/"
                    combs = [comb for comb in combs if filter_value in comb[filter_elem]]

            if order_info and not isinstance(order_info, list) and order_info.variable is not None:
                reverse = True if order_info.sorting_order == "desc" else False
                sort_elem = order_info.variable
                if combs and "?p" in combs[0]:
                    rel_combs = {}
                    for comb in combs:
                        if comb["?p"] not in rel_combs:
                            rel_combs[comb["?p"]] = []
                        rel_combs[comb["?p"]].append(comb)
                    rel_combs_list = rel_combs.values()
                else:
                    rel_combs_list = [combs]
                new_rel_combs_list = []
                for rel_combs in rel_combs_list:
                    new_rel_combs = []
                    for rel_comb in rel_combs:
                        value_str = rel_comb[sort_elem].split('^^')[0].strip('"+')
                        fnd_date = re.findall(r"[\d]{3,4}-[\d]{1,2}-[\d]{1,2}", value_str)
                        fnd_num = re.findall(r"([\d]+)\.([\d]+)", value_str)
                        if fnd_date:
                            rel_comb[sort_elem] = fnd_date[0]
                        elif fnd_num or value_str.isdigit():
                            rel_comb[sort_elem] = float(value_str)
                        new_rel_combs.append(rel_comb)
                    new_rel_combs = [(elem, n) for n, elem in enumerate(new_rel_combs)]
                    new_rel_combs = sorted(new_rel_combs, key=lambda x: (x[0][sort_elem], x[1]), reverse=reverse)
                    new_rel_combs = [elem[0] for elem in new_rel_combs]
                    new_rel_combs_list.append(new_rel_combs)
                combs = [new_rel_combs[0] for new_rel_combs in new_rel_combs_list]

            if what_return and what_return[-1].startswith("count"):
                answers = [[len(combs)]]
            else:
                answers = [[elem[key] for key in what_return if key in elem] for elem in combs]

            if answer_types:
                if list(answer_types) == ["date"]:
                    answers = [[entity for entity in answer
                                if re.findall(r"[\d]{3,4}-[\d]{1,2}-[\d]{1,2}", entity)] for answer in answers]
                elif list(answer_types) == ["not_date"]:
                    answers = [[entity for entity in answer
                                if not re.findall(r"[\d]{3,4}-[\d]{1,2}-[\d]{1,2}", entity)] for answer in answers]
                else:
                    answer_types = set(answer_types)
                    answers = [[entity for entity in answer
                                if answer_types.intersection(self.find_types(entity))] for answer in answers]
            if is_boolean:
                answers = [["Yes" if len(triplets) > 0 else "No"]]
            found_rels = [[elem[key] for key in rels_from_query if key in elem] for elem in combs]
            ans_rels_combs = [(answer, rel, comb) for answer, rel, comb in zip(answers, found_rels, combs)
                              if any([entity for entity in answer])]
            answers = [elem[0] for elem in ans_rels_combs]
            found_rels = [elem[1] for elem in ans_rels_combs]
            found_combs = [elem[2] for elem in ans_rels_combs]

        return answers, found_rels, found_combs

    @staticmethod
    def define_is_boolean(query_hdt_seq):
        return len(query_hdt_seq) == 1 and all([not query_hdt_seq[0][i].startswith("?") for i in [0, 2]])

    @staticmethod
    def merge_combs(comb1, comb2):
        new_comb = {}
        for key in comb1:
            if (key in comb2 and comb1[key] == comb2[key]) or key not in comb2:
                new_comb[key] = comb1[key]
        for key in comb2:
            if (key in comb1 and comb2[key] == comb1[key]) or key not in comb1:
                new_comb[key] = comb2[key]
        return new_comb

    def search(self, query: List[str], unknown_elem_positions: List[Tuple[int, str]], rel_type):
        query = list(map(lambda elem: "" if elem.startswith('?') else elem, query))
        subj, rel, obj = query
        if self.file_format == "hdt":
            combs = []
            triplets, cnt = self.document.search_triples(subj, rel, obj)
            if cnt < self.max_comb_num:
                triplets = list(triplets)
                if rel == self.prefixes["description"] or rel == self.prefixes["label"]:
                    triplets = [triplet for triplet in triplets if triplet[2].endswith(self.lang)]
                    combs = [{elem: triplet[pos] for pos, elem in unknown_elem_positions} for triplet in triplets]
                else:
                    if isinstance(self.prefixes["rels"][rel_type], str):
                        combs = [{elem: triplet[pos] for pos, elem in unknown_elem_positions} for triplet in triplets
                                 if (triplet[1].startswith(self.prefixes["rels"][rel_type])
                                     or triplet[1].startswith(self.prefixes["rels"]["type"]))]
                    else:
                        combs = [{elem: triplet[pos] for pos, elem in unknown_elem_positions} for triplet in triplets
                                 if (any(triplet[1].startswith(tp) for tp in self.prefixes["rels"][rel_type])
                                     or triplet[1].startswith(self.prefixes["rels"]["type"]))]
            else:
                log.debug("max comb num exceeds")
        else:
            triplets = []
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

        return combs, triplets

    def find_label(self, entity: str, question: str = "") -> str:
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
                        found_label = label[2].strip(self.lang)
                        for old_tok, new_tok in self.replace_tokens:
                            found_label = found_label.replace(old_tok, new_tok)
                        found_label = found_label.strip()
                        return found_label

            elif entity.endswith(self.lang):
                # entity: '"Lake Baikal"@en'
                entity = entity[:-3].replace('$', ' ').replace('  ', ' ')
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
                entity = self.format_date(entity, question).replace('$', '')
                return entity

            elif re.findall(r"[\d]{3,4}-[\d]{2}-[\d]{2}", entity):
                entity = self.format_date(entity, question).replace('$', '')
                return entity

            elif entity in ["Yes", "No"]:
                return entity

            elif entity.isdigit():
                entity = entity.replace('.', ',')
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
        dates_dict = {"January": "января", "February": "февраля", "March": "марта", "April": "апреля", "May": "мая",
                      "June": "июня", "July": "июля", "August": "августа", "September": "сентября",
                      "October": "октября",
                      "November": "ноября", "December": "декабря"}
        date_info = re.findall("([\d]{3,4})-([\d]{1,2})-([\d]{1,2})", entity)
        if date_info:
            year, month, day = date_info[0]
            if "how old" in question.lower() or "сколько лет" in question.lower():
                entity = datetime.datetime.now().year - int(year)
            elif "в каком году" in question.lower():
                entity = year
            elif "в каком месяце" in question.lower():
                entity = month
            elif day not in {"00", "0"}:
                date = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                entity = date.strftime("%d %B %Y")
            else:
                entity = year
            if self.lang == "@ru":
                for mnth, mnth_replace in dates_dict.items():
                    entity = entity.replace(mnth, mnth_replace)
            return str(entity)
        entity = entity.lstrip('+-')
        return entity

    def find_alias(self, entity: str) -> List[str]:
        aliases = []
        if entity.startswith(self.prefixes["entity"]):
            labels, cardinality = self.document.search_triples(entity, self.prefixes["alias"], "")
            aliases = [label[2].strip(self.lang).strip('"') for label in labels if label[2].endswith(self.lang)]
        return aliases

    def find_rels(self, entity: str, direction: str, rel_type: str = "no_type") -> List[str]:
        rels = []
        if self.file_format == "hdt":
            if not rel_type:
                rel_type = "direct"
            if direction == "forw":
                query = [f"{self.prefixes['entity']}/{entity}", "", ""]
            else:
                query = ["", "", f"{self.prefixes['entity']}/{entity}"]
            triplets, c = self.document.search_triples(*query)
            triplets = list(triplets)
            if isinstance(self.prefixes['rels'][rel_type], str):
                start_str = f"{self.prefixes['rels'][rel_type]}/P"
                rels = {triplet[1] for triplet in triplets if triplet[1].startswith(start_str)}
            else:
                rels = {triplet[1] for triplet in triplets
                        if any([triplet[1].startswith(tp) for tp in self.prefixes['rels'][rel_type]])}
            rels = list(rels)
            if self.used_rels:
                rels = [rel for rel in rels if rel.split("/")[-1] in self.used_rels]
        return rels

    def find_rels_2hop(self, entity_ids, rels_1hop):
        rels = []
        for entity_id in entity_ids:
            for rel_1hop in rels_1hop:
                triplets, cnt = self.document.search_triples(f"{self.prefixes['entity']}/{entity_id}", rel_1hop, "")
                triplets = [triplet for triplet in triplets if triplet[2].startswith(self.prefixes['entity'])]
                objects_1hop = [triplet[2].split("/")[-1] for triplet in triplets]
                triplets, cnt = self.document.search_triples("", rel_1hop, f"{self.prefixes['entity']}/{entity_id}")
                triplets = [triplet for triplet in triplets if triplet[0].startswith(self.prefixes['entity'])]
                objects_1hop += [triplet[0].split("/")[-1] for triplet in triplets]
                for object_1hop in objects_1hop[:5]:
                    tr_2hop, cnt = self.document.search_triples(f"{self.prefixes['entity']}/{object_1hop}", "", "")
                    rels_2hop = [elem[1] for elem in tr_2hop if elem[1] != rel_1hop]
                    if self.used_rels:
                        rels_2hop = [elem for elem in rels_2hop if elem.split("/")[-1] in self.used_rels]
                    rels += rels_2hop
                    tr_2hop, cnt = self.document.search_triples("", "", f"{self.prefixes['entity']}/{object_1hop}")
                    rels_2hop = [elem[1] for elem in tr_2hop if elem[1] != rel_1hop]
                    if self.used_rels:
                        rels_2hop = [elem for elem in rels_2hop if elem.split("/")[-1] in self.used_rels]
                    rels += rels_2hop
        rels = list(set(rels))
        return rels

    def find_object(self, entity: str, rel: str, direction: str) -> List[str]:
        objects = []
        if not direction:
            direction = "forw"
        if self.file_format == "hdt":
            entity = f"{self.prefixes['entity']}/{entity.split('/')[-1]}"
            rel = f"{self.prefixes['rels']['direct']}/{rel}"
            if direction == "forw":
                triplets, cnt = self.document.search_triples(entity, rel, "")
                if cnt < self.max_comb_num:
                    objects.extend([triplet[2].split('/')[-1] for triplet in triplets])
            else:
                triplets, cnt = self.document.search_triples("", rel, entity)
                objects.extend([triplet[0].split('/')[-1] for triplet in triplets])
        else:
            entity = entity.split('/')[-1]
            rel = rel.split('/')[-1]
            triplets = self.document.get(entity, {}).get(direction, [])
            triplets = self.uncompress(triplets)
            for found_rel, *objects in triplets:
                if rel == found_rel:
                    objects.extend(objects)
        return objects

    def check_triplet(self, subj: str, rel: str, obj: str) -> bool:
        if self.file_format == "hdt":
            subj = f"{self.prefixes['entity']}/{subj}"
            rel = f"{self.prefixes['rels']['direct']}/{rel}"
            obj = f"{self.prefixes['entity']}/{obj}"
            triplets, cnt = self.document.search_triples(subj, rel, obj)
            if cnt > 0:
                return True
            else:
                return False
        else:
            subj = subj.split('/')[-1]
            rel = rel.split('/')[-1]
            obj = obj.split('/')[-1]
            triplets = self.document.get(subj, {}).get("forw", [])
            triplets = self.uncompress(triplets)
            for found_rel, *objects in triplets:
                if found_rel == rel:
                    for found_obj in objects:
                        if found_obj == obj:
                            return True
            return False

    def find_types(self, entity: str):
        types = []
        if self.file_format == "hdt":
            if not entity.startswith("http"):
                entity = f"{self.prefixes['entity']}/{entity}"
            tr, c = self.document.search_triples(entity, f"{self.prefixes['rels']['direct']}/P31", "")
            types = [triplet[2].split('/')[-1] for triplet in tr]
            for rel in ["P106", "P21"]:
                tr, c = self.document.search_triples(entity, f"{self.prefixes['rels']['direct']}/{rel}", "")
                types += [triplet[2].split('/')[-1] for triplet in tr]

        if self.file_format == "pickle":
            entity = entity.split('/')[-1]
            triplets = self.document.get(entity, {}).get("forw", [])
            triplets = self.uncompress(triplets)
            for triplet in triplets:
                if triplet[0] == "P31":
                    types = triplet[1:]
        types = set(types)
        return types

    def find_subclasses(self, entity: str):
        types = []
        if self.file_format == "hdt":
            if not entity.startswith("http"):
                entity = f"{self.prefixes['entity']}/{entity}"
            tr, c = self.document.search_triples(entity, f"{self.prefixes['rels']['direct']}/P279", "")
            types = [triplet[2].split('/')[-1] for triplet in tr]
        if self.file_format == "pickle":
            entity = entity.split('/')[-1]
            triplets = self.document.get(entity, {}).get("forw", [])
            triplets = self.uncompress(triplets)
            for triplet in triplets:
                if triplet[0] == "P279":
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

    def fill_triplets(self, init_triplets, what_to_return, comb):
        filled_triplets = []
        for n, (subj, rel, obj) in enumerate(init_triplets):
            if "statement" in self.prefixes and subj.startswith("?") \
                    and comb.get(subj, "").startswith(self.prefixes["statement"]) and not rel.startswith("?") \
                    and (obj == what_to_return[0] or re.findall(r"[\d]{3,4}", comb.get(what_to_return[0], ""))):
                continue
            else:
                if "statement" in self.prefixes and subj.startswith("?") \
                        and str(comb.get(subj, "")).startswith(self.prefixes["statement"]):
                    if not comb.get(what_to_return[0], "").startswith("http") \
                            and re.findall(r"[\d]{3,4}", comb.get(what_to_return[0], "")):
                        subj = init_triplets[1][2]
                    else:
                        subj = what_to_return[0]
                if "statement" in self.prefixes and obj.startswith("?") \
                        and str(comb.get(obj, "")).startswith(self.prefixes["statement"]):
                    if not str(comb.get(what_to_return[0], "")).startswith("http") \
                            and re.findall(r"[\d]{3,4}", str(comb.get(what_to_return[0], ""))):
                        obj = init_triplets[1][2]
                    else:
                        obj = what_to_return[0]
                subj, obj = str(subj), str(obj)
                if subj.startswith("?"):
                    subj = comb.get(subj, "")
                if obj.startswith("?"):
                    obj = comb.get(obj, "")
                if rel.startswith("?"):
                    rel = comb.get(rel, "")
                subj_label = self.find_label(subj)
                obj_label = self.find_label(obj)
                if rel in self.rel_q2name:
                    rel_label = self.rel_q2name[rel]
                elif rel.split("/")[-1] in self.rel_q2name:
                    rel_label = self.rel_q2name[rel.split("/")[-1]]
                else:
                    rel_label = self.find_label(rel)
                if isinstance(rel_label, list) and rel_label:
                    rel_label = rel_label[0]
                filled_triplets.append([subj_label, rel_label, obj_label])
        return filled_triplets
