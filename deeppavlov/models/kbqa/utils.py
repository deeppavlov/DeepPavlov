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
from typing import List


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
    map_query_str_to_wikidata = [("P0", "http://wd"),
                                 ("P00", "http://wl"),
                                 ("wd:", "http://we/"),
                                 ("wdt:", "http://wpd/"),
                                 (" p:", " http://wp/"),
                                 ("ps:", "http://wps/"),
                                 ("pq:", "http://wpq/")]

    for query_str, wikidata_str in map_query_str_to_wikidata:
        query = query.replace(query_str, wikidata_str)
    for n, entity in enumerate(entity_comb[:-1]):
        query = query.replace(f"e{n + 1}", entity)
    for n, entity_type in enumerate(type_comb[:-1]):  # type_entity
        query = query.replace(f"t{n + 1}", entity_type)
    for n, (rel, score) in enumerate(rel_comb[:-1]):
        query = query.replace(f"r{n + 1}", rel)
    query = query.replace("http://wpd/P0", "http://wd")
    query = query.replace("http://wpd/P00", "http://wl")
    query = query.split(' ')
    return query
