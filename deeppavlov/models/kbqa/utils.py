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

import re
import itertools
from typing import List, Tuple


def extract_year(question_tokens: List[str], question: str) -> str:
    question_patterns = [r'.*\d{1,2}/\d{1,2}/(\d{4}).*', r'.*\d{1,2}-\d{1,2}-(\d{4}).*', r'.*(\d{4})-\d{1,2}-\d{1,2}.*']
    token_patterns = [r'(\d{4})', r'^(\d{4})-.*', r'.*-(\d{4})$']
    year = ""
    for pattern in question_patterns:
        fnd = re.search(pattern, question)
        if fnd is not None:
            year = fnd.group(1)
            break
    else:
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
    max_words = ["maximum", "highest", "max ", "greatest", "most", "longest", "biggest", "deepest"]

    for word in max_words:
        if word in question_lower:
            return "desc"

    return "asc"


def make_combs(entity_ids: List[List[str]], permut: bool) -> List[List[str]]:
    entity_ids = [[(entity, n) for n, entity in enumerate(entities_list)] for entities_list in entity_ids]
    entity_ids = list(itertools.product(*entity_ids))
    entity_ids_permut = []
    if permut:
        for comb in entity_ids:
            entity_ids_permut += itertools.permutations(comb)
    else:
        entity_ids_permut = entity_ids
    entity_ids = sorted(entity_ids_permut, key=lambda x: sum([elem[1] for elem in x]))
    ent_combs = [[elem[0] for elem in comb] + [sum([elem[1] for elem in comb])] for comb in entity_ids]
    return ent_combs


def fill_query(query: List[str], entity_comb: List[str], type_comb: List[str], rel_comb: List[str]) -> List[str]:
    ''' example of query: ["wd:E1", "p:R1", "?s"]
                   entity_comb: ["Q159"]
                   type_comb: []
                   rel_comb: ["P17"]
    '''
    query = " ".join(query)
    map_query_str_to_wikidata = [("P0", "http://schema.org/description"),
                                 ("wd:", "http://www.wikidata.org/entity/"),
                                 ("wdt:", "http://www.wikidata.org/prop/direct/"),
                                 (" p:", " http://www.wikidata.org/prop/"),
                                 ("wdt:", "http://www.wikidata.org/prop/direct/"),
                                 ("ps:", "http://www.wikidata.org/prop/statement/"),
                                 ("pq:", "http://www.wikidata.org/prop/qualifier/")]

    for query_str, wikidata_str in map_query_str_to_wikidata:
        query = query.replace(query_str, wikidata_str)
    for n, entity in enumerate(entity_comb[:-1]):
        query = query.replace(f"e{n + 1}", entity)
    for n, entity_type in enumerate(type_comb[:-1]):  # type_entity
        query = query.replace(f"t{n + 1}", entity_type)
    for n, (rel, score) in enumerate(rel_comb[:-1]):
        query = query.replace(f"r{n + 1}", rel)
    query = query.replace("http://www.wikidata.org/prop/direct/P0", "http://schema.org/description")
    query = query.split(' ')
    return query


def fill_online_query(query: List[str], entity_comb: List[str], type_comb: List[str],
                      rel_comb: List[str], rels_to_replace: List[str],
                      rels_for_filter: List[str], rel_list_for_filter: List[List[str]]) -> Tuple[str, List[str]]:
    rel_list_for_filter = [[rel for rel, score in rel_list] for rel_list in rel_list_for_filter]
    for n, entity in enumerate(entity_comb[:-1]):
        query = query.replace(f"e{n + 1}", entity)
    for n, entity_type in enumerate(type_comb[:-1]):  # type_entity
        query = query.replace(f"t{n + 1}", entity_type)
    for n, (rel, score) in enumerate(rel_comb[:-1]):
        query = query.replace(rels_to_replace[n], rel)

    candidate_rel_filters = []
    new_rels = []
    if rels_for_filter:
        n = 0
        for rel, candidate_rels in zip(rels_for_filter, rel_list_for_filter):
            rel_types = re.findall(f" ([\S]+:){rel}", query)
            for rel_type in rel_types:
                new_rel = f"?p{n + 1}"
                query = query.replace(f'{rel_type}{rel}', new_rel)
                new_rels.append(new_rel)
                candidate_rels_filled = [f"{new_rel} = {rel_type}{rel_value}" for rel_value in candidate_rels]
                candidate_rel_str = " || ".join(candidate_rels_filled)
                candidate_rel_filters.append(f"({candidate_rel_str})")
                n += 1

        if "filter" in query:
            query = query.replace("filter(", f"filter({'&&'.join(candidate_rel_filters)}&&")
        else:
            query = query.replace(" }", f" filter({'&&'.join(candidate_rel_filters)}) }}")

        query = query.replace(" where", f" {' '.join(new_rels)} where")
        if rel_list_for_filter[0][0] == "P0" and len(entity_comb) == 2:
            query = f"select ?ent ?p1 where {{ wd:{entity_comb[0]} ?p1" + \
                    "?ent filter((?p1=schema:description)&&(lang(?ent)='en'))}}"

    return query, new_rels
