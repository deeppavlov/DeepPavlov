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
from collections import namedtuple
from typing import List, Tuple, Dict, Any


def find_query_features(query, qualifier_rels=None, question=None, order_from_query=None):
    query = query.lower().replace("select distinct", "select")
    answer_ent = re.findall(r"select [\(]?([\S]+) ", query)
    order_info_nt = namedtuple("order_info", ["variable", "sorting_order"])
    order_variable = re.findall("order by (asc|desc)\((.*)\)", query)
    if order_variable:
        if (qualifier_rels and len(qualifier_rels[0][4:]) > 1) or order_from_query:
            answers_sorting_order = order_variable[0][0]
        else:
            answers_sorting_order = order_of_answers_sorting(question)
        order_info = order_info_nt(order_variable[0][1], answers_sorting_order)
    else:
        order_info = order_info_nt(None, None)
    filter_from_query = re.findall("contains\((\?\w), (.+?)\)", query)
    return answer_ent, order_info, filter_from_query


def extract_year(question_tokens: List[str], question: str) -> str:
    question_patterns = [r'.*\d{1,2}/\d{1,2}/(\d{4}).*', r'.*\d{1,2}-\d{1,2}-(\d{4}).*', r'.*(\d{4})-\d{1,2}-\d{1,2}.*']
    from_to_patterns = [r"from ([\d]{3,4}) to [\d]{3,4}", r"с ([\d]{3,4}) по [\d]{3,4}"]
    token_patterns = [r'(\d{4})', r'^(\d{4})-.*', r'.*-(\d{4})$']
    year = ""
    for pattern in question_patterns:
        fnd = re.search(pattern, question)
        if fnd is not None:
            year = fnd.group(1)
            break
    else:
        for pattern in from_to_patterns:
            fnd = re.findall(pattern, question)
            if fnd:
                return fnd[0]
        for token in question_tokens:
            for pattern in token_patterns:
                fnd = re.search(pattern, token)
                if fnd is not None:
                    return fnd.group(1)
    return year


def extract_number(question_tokens: List[str], question: str) -> str:
    number = ""
    fnd = re.search(r'.*(\d\.\d+e\+\d+)\D*', question)
    if fnd is not None:
        number = fnd.group(1)
    else:
        for tok in question_tokens:
            if tok[0].isdigit():
                number = tok
                break

    number = number.replace('1st', '1').replace('2nd', '2').replace('3rd', '3')
    number = number.strip(".0")

    return number


def order_of_answers_sorting(question: str) -> str:
    question_lower = question.lower()
    max_words = ["maximum", "highest", "max ", "greatest", "most", "longest", "biggest", "deepest", "завершил",
                 "закончил", "завершает"]
    for word in max_words:
        if word in question_lower:
            return "desc"
    return "asc"


def make_combs(entity_ids: List[List[str]], permut: bool) -> List[List[str]]:
    entity_ids = [[(entity, n) for n, entity in enumerate(entities_list)] for entities_list in entity_ids]
    entity_ids = list(itertools.product(*entity_ids))
    entity_ids = [comb for comb in entity_ids if not
    (all([comb[i][0][0].split("/")[-1] == comb[0][0][0].split("/")[-1] for i in range(len(comb))])
     and not all([comb[i][0][0] == comb[0][0][0] for i in range(len(comb))]))]
    entity_ids_permut = []
    if permut:
        for comb in entity_ids:
            entity_ids_permut += itertools.permutations(comb)
    else:
        entity_ids_permut = entity_ids
    entity_ids = sorted(entity_ids_permut, key=lambda x: sum([elem[1] for elem in x]))
    ent_combs = [[elem[0] for elem in comb] + [sum([elem[1] for elem in comb])] for comb in entity_ids]
    return ent_combs


def fill_slots(query: str, entity_comb: List[str], type_comb: List[str], rel_comb: List[Tuple[str, float]],
               delete_rel_prefix: bool = False) -> str:
    for n, entity in enumerate(entity_comb[:-1]):
        query = query.replace(f"e{n + 1}", entity)
    for n, entity_type in enumerate(type_comb[:-1]):  # type_entity
        query = query.replace(f"t{n + 1}", entity_type)
    for n, (rel, score) in enumerate(rel_comb[:-1]):
        if not rel.startswith("?"):
            if delete_rel_prefix:
                rel = rel.split("/")[-1]
            query = query.replace(f"r{n + 1}", rel)
    return query


def correct_variables(query_triplets: List[str], answer_ent: List[str], query_info: Dict[str, str]):
    for i in range(len(query_triplets)):
        for ent_var in answer_ent:
            triplet_elements = query_triplets[i].split()
            for j in range(len(triplet_elements)):
                if triplet_elements[j] not in ent_var and triplet_elements[j].startswith("?"):
                    triplet_elements[j] = query_info["mid_var"]
                    break
                if triplet_elements[j].startswith("?") \
                        and triplet_elements[j] not in [query_info["mid_var"], query_info["unk_var"]]:
                    triplet_elements[j] = query_info["unk_var"]
                    break
            query_triplets[i] = " ".join(triplet_elements)
            query_triplets[i] = query_triplets[i].replace(ent_var, query_info["unk_var"])
    return query_triplets


def query_from_triplets(query_triplets: List[str], answer_ent: List[str], query_info: Dict[str, str]) -> str:
    filled_query = " . ".join(query_triplets)
    if answer_ent and answer_ent[0].lower().startswith("count"):
        filled_query = f"SELECT COUNT({query_info['unk_var']}) " + \
                       f"WHERE {{ {filled_query}. }}"
    else:
        filled_query = f"SELECT {query_info['unk_var']} WHERE {{ {filled_query}. }}"
    filled_query = filled_query.replace(" ..", ".")
    return filled_query


def fill_query(query: List[str], entity_comb: List[str], type_comb: List[str], rel_comb: List[Tuple[str, float]],
               map_query_str_to_kb) -> List[str]:
    ''' example of query: ["wd:E1", "p:R1", "?s"]
                   entity_comb: ["Q159"]
                   type_comb: []
                   rel_comb: ["P17"]
        map_query_str_to_kb = [("P0", "http://wd"),
                               ("P00", "http://wl"),
                               ("wd:", "http://we/"),
                               ("wdt:", "http://wpd/"),
                               (" p:", " http://wp/"),
                               ("ps:", "http://wps/"),
                               ("pq:", "http://wpq/")]
    '''
    query = " ".join(query)

    for query_str, wikidata_str in map_query_str_to_kb:
        query = query.replace(query_str, wikidata_str)
    query = fill_slots(query, entity_comb, type_comb, rel_comb)
    query = query.replace("http://wpd/P0", "http://wd")
    query = query.replace("http://wpd/P00", "http://wl")
    query = query.split(' ')
    return query


def make_sparql_query(query_info: Tuple[List[str], List[str], List[str], Dict[str, Any], Dict[str, Any]],
                      entities: List[str], rels: List[Tuple[str, float]], types: List[str],
                      query_info_dict: Dict[str, str]) -> List[str]:
    query_triplets, filled_triplets, answer_ent, filter_info, order_info = query_info
    query_triplets = [fill_slots(elem, entities, types, rels, delete_rel_prefix=True) for elem in query_triplets]
    query_triplets = correct_variables(query_triplets, answer_ent, query_info_dict)
    filled_queries = []
    if any(["qualifier" in filter_info_element for filter_info_element in filter_info]):
        filled_queries.append(query_from_triplets(query_triplets, answer_ent, query_info_dict))
    else:
        for triplets_p in list(itertools.permutations(query_triplets)):
            filled_queries.append(query_from_triplets(triplets_p, answer_ent, query_info_dict))
    return filled_queries


def merge_sparql_query(query_info: Tuple[List[str], List[str], Dict[str, Any], Dict[str, Any]],
                       query_info_dict: Dict[str, str]) -> str:
    query_triplets, answer_ent, filter_info, order_info = query_info
    query = query_from_triplets(query_triplets, answer_ent, query_info_dict)
    return query


def preprocess_template_queries(template_queries: Dict[str, Any], kb_prefixes: Dict[str, str]) -> Dict[str, Any]:
    for template_num in template_queries:
        template = template_queries[template_num]
        query = template["query_template"]
        q_triplets = re.findall("{[ ]?(.*?)[ ]?}", query)[0].split(' . ')
        q_triplets = [triplet.split(' ')[:3] for triplet in q_triplets]
        if not "rel_types" in template:
            template["rel_types"] = ["direct" for _ in q_triplets]
        rel_types = template["rel_types"]
        rel_dirs, n_hops, entities, types, gr_ent, mod_ent, q_ent = [], [], set(), set(), set(), set(), set()

        for n, (triplet, rel_type) in enumerate(zip(q_triplets, rel_types)):
            if not triplet[1].startswith(kb_prefixes["type_rel"]):
                if triplet[2].startswith("?"):
                    rel_dirs.append("forw")
                else:
                    rel_dirs.append("backw")
            for ind in [0, 2]:
                if triplet[ind].startswith(kb_prefixes["entity"]):
                    entities.add(triplet[ind])
                elif triplet[ind].startswith(kb_prefixes["type"]):
                    types.add(triplet[ind])
            if rel_type in {"qualifier", "statement"}:
                if triplet[2].startswith(kb_prefixes["entity"]):
                    q_ent.add(triplet[2])
            else:
                if triplet[0].startswith(kb_prefixes["entity"]):
                    gr_ent.add(triplet[0])
                elif triplet[2].startswith(kb_prefixes["entity"]):
                    mod_ent.add(triplet[2])
            if triplet[1].startswith(kb_prefixes["rel"]) and triplet[0].startswith("?") and triplet[2].startswith("?"):
                n_hops.append("2-hop")
            elif n == 0 and len(q_triplets) == 2 and q_triplets[1][1].startswith(kb_prefixes["rel"]) \
                    and q_triplets[1][0].startswith("?") and q_triplets[1][2].startswith("?"):
                n_hops.append("1-of-2-hop")
            else:
                n_hops.append("1-hop")
        syntax_structure = {"gr_ent": len(gr_ent), "types": len(types), "mod_ent": len(mod_ent),
                            "q_ent": len(q_ent), "year_or_number": False, "count": False, "order": False}
        if "filter" in query.lower():
            syntax_structure["year_or_number"] = True
        if "order" in query.lower():
            syntax_structure["order"] = True
        if "count" in query.lower():
            syntax_structure["count"] = True
        if not "query_sequence" in template:
            template["query_sequence"] = list(range(1, len(q_triplets) + 1))
        template["rel_dirs"] = rel_dirs
        template["n_hops"] = n_hops
        template["entities_and_types_num"] = [len(entities), len(types)]
        if entities:
            entities_str = '_'.join([str(num) for num in list(range(1, len(entities) + 1))])
        else:
            entities_str = "0"
        if types:
            types_str = '_'.join([str(num) for num in list(range(1, len(types) + 1))])
        else:
            types_str = "0"
        template["entities_and_types_select"] = f"{entities_str} {types_str}"
        template["syntax_structure"] = syntax_structure
        if "return_if_found" not in template:
            template["return_if_found"] = False
        if "priority" not in template:
            template["priority"] = 1
        template_queries[template_num] = template
    return template_queries
