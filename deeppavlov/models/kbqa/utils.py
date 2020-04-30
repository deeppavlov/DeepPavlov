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
from typing import Tuple, List


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


def asc_desc(question: str) -> bool:
    question_lower = question.lower()
    max_words = ["maximum", "highest", "max(", "greatest", "most", "longest"]

    for word in max_words:
        if word in question_lower:
            return False

    return True


def make_entity_combs(entity_ids: List[List[str]]) -> List[Tuple[str, str, int]]:
    ent_combs = []
    for n, entity_1 in enumerate(entity_ids[0]):
        for m, entity_2 in enumerate(entity_ids[1]):
            ent_combs.append((entity_1, entity_2, (n + m)))
            ent_combs.append((entity_2, entity_1, (n + m)))

    ent_combs = sorted(ent_combs, key=lambda x: x[2])
    return ent_combs
