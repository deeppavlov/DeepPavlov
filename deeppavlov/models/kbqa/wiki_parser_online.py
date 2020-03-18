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

from typing import List, Dict
import requests
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


def get_answer(query: str) -> Dict[str, Dict[str, str]]:
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    data = []
    for i in range(3):
        try:
            data_0 = requests.get(url, params={'query': query, 'format': 'json'}).json()
            if "results" in data_0.keys():
                data = data_0['results']['bindings']
            if "boolean" in data_0.keys():
                data = data_0['boolean']

        except:
            e = 0

        if data:
            break

    return data


@register('wiki_parser_online')
class WikiParserOnline(Component):
    """
        This class extracts relations or objects from Wikidata query service
    """

    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, what_return: str,
                 direction: str,
                 entity: str,
                 rel: str = None,
                 obj: str = None,
                 find_label: bool = False) -> List[str]:

        if find_label:
            if entity.startswith("Q"):
                template = "SELECT DISTINCT ?label WHERE {{ wd:{} rdfs:label ?label . FILTER (lang(?label) = 'en') }}"
                query = template.format(entity)
                labels = get_answer(query)
                if labels:
                    labels = [label["label"]["value"] for label in labels]
                    return labels[0]

            else:
                return entity

        if direction == "forw":
            if obj is not None:
                query = "ASK WHERE {{ wd:{} wdt:{} wd:{} }}".format(entity, rel, obj)
                res = get_answer(query)
                return res

            if what_return == "rels":
                query = "SELECT DISTINCT ?rel ?obj WHERE {{ wd:{} ?rel ?obj }}".format(entity)
                rel_obj = get_answer(query)
                if rel_obj:
                    rels = [entry["rel"]["value"].split('/')[-1] for entry in rel_obj]
                    return rels

            if what_return == "objects":
                query = "SELECT DISTINCT ?obj WHERE {{ wd:{} wdt:{} ?obj }}".format(entity, rel)
                obj = get_answer(query)
                if obj:
                    objects = [entry["obj"]["value"].split('/')[-1] for entry in obj]
                    return objects

        if direction == "backw":
            if obj is not None:
                query = "ASK WHERE {{ wd:{} wdt:{} wd:{} }}".format(obj, rel, entity)
                res = get_answer(query)
                return res

            if what_return == "rels":
                query = "SELECT DISTINCT ?rel ?obj WHERE {{ ?obj ?rel wd:{} }}".format(entity)
                rel_obj = get_answer(query)
                if rel_obj:
                    rels = [entry["rel"]["value"].split('/')[-1] for entry in rel_obj]
                    return rels

            if what_return == "objects":
                query = "SELECT DISTINCT ?obj WHERE {{ ?obj wdt:{} wd:{} }}".format(entity, rel)
                obj = get_answer(query)
                if obj:
                    objects = [entry["obj"]["value"].split('/')[-1] for entry in obj]
                    return objects

        return []
