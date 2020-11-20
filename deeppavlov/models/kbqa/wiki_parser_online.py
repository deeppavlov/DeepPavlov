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
from time import sleep
from typing import List, Dict, Any

import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout

from deeppavlov import __version__ as dp_version
from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register('wiki_parser_online')
class WikiParserOnline:
    """This class extract relations or labels from Wikidata query service"""

    def __init__(self, url: str, timeout: float = 0.5, **kwargs) -> None:
        self.url = url
        self.timeout = timeout
        self.agent_header = {'User-Agent': f'wiki_parser_online/{dp_version} (https://deeppavlov.ai;'
                                           f' info@deeppavlov.ai) deeppavlov/{dp_version}'}

    def __call__(self, parser_info_list: List[str], queries_list: List[Any]) -> List[Any]:
        wiki_parser_output = []
        for parser_info, query in zip(parser_info_list, queries_list):
            if parser_info == "query_execute":
                query_to_execute, return_if_found = query
                candidate_output = self.get_answer(query_to_execute)
                wiki_parser_output.append(candidate_output)
                if return_if_found and candidate_output:
                    return wiki_parser_output
            elif parser_info == "find_rels":
                wiki_parser_output += self.find_rels(*query)
            elif parser_info == "find_label":
                wiki_parser_output.append(self.find_label(*query))
            else:
                raise ValueError("Unsupported query type")
        return wiki_parser_output

    def get_answer(self, query: str) -> List[Dict[str, Dict[str, str]]]:
        data = []
        for i in range(5):
            try:
                resp = requests.get(self.url,
                                    params={'query': query, 'format': 'json'},
                                    timeout=self.timeout,
                                    headers=self.agent_header)
                if resp.status_code != 200:
                    continue
                data_0 = resp.json()
                if "results" in data_0.keys():
                    data = data_0['results']['bindings']
                elif "boolean" in data_0.keys():
                    data = data_0['boolean']
                break
            except (ConnectTimeout, ReadTimeout) as e:
                log.warning(f'TimeoutError: {repr(e)}')
            except ConnectionError as e:
                log.warning(f'Connection error: {repr(e)}\nWaiting 1s...')
                sleep(1)
        return data

    def find_label(self, entity: str, question: str) -> str:
        entity = str(entity).replace('"', '')
        if entity.startswith("http://www.wikidata.org/entity/Q"):
            entity = entity.split('/')[-1]
        if entity.startswith("Q"):
            query = f"SELECT DISTINCT ?label WHERE {{ wd:{entity} rdfs:label ?label . FILTER (lang(?label) = 'en') }}"
            labels = self.get_answer(query)
            if labels:
                labels = [label["label"]["value"] for label in labels]
                return labels[0]
        elif entity.endswith("T00:00:00Z"):
            return entity.split('T00:00:00Z')[0]
        else:
            return entity

    def find_rels(self, entity: str, direction: str, rel_type: str = "no_type") -> List[str]:
        if direction == "forw":
            query = f"SELECT DISTINCT ?rel WHERE {{ wd:{entity} ?rel ?obj . }}"
        else:
            query = f"SELECT DISTINCT ?rel WHERE {{ ?subj ?rel wd:{entity} . }}"
        rels = self.get_answer(query)
        if rels:
            rels = [rel["rel"]["value"] for rel in rels]

        if rel_type != "no_type":
            start_str = f"http://www.wikidata.org/prop/{rel_type}"
        else:
            start_str = "http://www.wikidata.org/prop/P"
        rels = [rel for rel in rels if rel.startswith(start_str)]
        return rels
