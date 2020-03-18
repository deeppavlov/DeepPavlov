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

from typing import List
from hdt import HDTDocument
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('wiki_parser')
class WikiParser(Component):
    """
        This class extract relations, objects or triplets from Wikidata HDT file
    """

    def __init__(self, wiki_filename: str, **kwargs):
        wiki_path = expand_path(wiki_filename)
        self.document = HDTDocument(str(wiki_path))

    def __call__(self, what_return: str,
                 direction: str,
                 entity: str,
                 rel: str = None,
                 obj: str = None,
                 type_of_rel: str = None,
                 filter_obj: str = None,
                 find_label: str = False,
                 find_alias: str = False) -> List[str]:
        """

        Args:
            what_return: "rels" - relations, "obj" - objects or "triplets"
            direction: "forw" - triplets [subject, relation, object]
                       "backw" - triplets [object, relation, subject]
            entity: id of entity
            rel: id of relation
            obj: id of object
            type_of_rel: "direct", "statement" or "qualifier"
            filter_obj: string which the object we want to contain
            find_label: whether to find label of entity from entity id
            find_alias: whether to find alias of entity from entity_id
        """

        if not entity.startswith("http://www.wikidata.org/") and entity.startswith("Q"):
            entity = "http://www.wikidata.org/entity/" + entity

        if find_label:
            if entity.startswith("http://www.wikidata.org/entity/"):
                labels, cardinality = self.document.search_triples(entity,
                                                                   "http://www.w3.org/2000/01/rdf-schema#label", "")
                for label in labels:
                    if label[2].endswith("@en"):
                        found_label = label[2].strip('@en').strip('"')
                        return found_label

            elif "http://www.w3.org/2001/XMLSchema#dateTime" in entity:
                entity = entity.strip("^^<http://www.w3.org/2001/XMLSchema#dateTime>").strip('"').strip("T00:00:00Z")
                return entity

            elif "http://www.w3.org/2001/XMLSchema#decimal>" in entity:
                entity = entity.strip("^^<http://www.w3.org/2001/XMLSchema#decimal>").strip('"').strip('"')
                return entity

            elif entity.isdigit():
                return entity

            return "Not Found"

        if find_alias:
            aliases = []
            if entity.startswith("http://www.wikidata.org/entity/"):
                labels, cardinality = self.document.search_triples(entity,
                                                                   "http://www.w3.org/2004/02/skos/core#altLabel", "")
                for label in labels:
                    if label[2].endswith("@en"):
                        aliases.append(label[2].strip('@en').strip('"'))
            return aliases

        if rel is not None:
            if type_of_rel is None:
                if not rel.startswith("http:"):
                    rel = "http://www.wikidata.org/prop/{}".format(rel)
            else:
                rel = "http://www.wikidata.org/prop/{}/{}".format(type_of_rel, rel)
        else:
            rel = ""

        if obj is not None:
            if not obj.startswith("http://www.wikidata.org/"):
                obj = "http://www.wikidata.org/entity/" + obj
        else:
            obj = ""

        if direction == "forw":
            triplets, cardinality = self.document.search_triples(entity, rel, obj)
        if direction == "backw":
            triplets, cardinality = self.document.search_triples(obj, rel, entity)

        found_triplets = []
        for triplet in triplets:
            if type_of_rel is None or (type_of_rel is not None and type_of_rel in triplet[1]):
                if filter_obj is None or (filter_obj is not None and filter_obj in triplet[2]):
                    found_triplets.append(triplet)

        if what_return == "rels":
            rels = [triplet[1].split('/')[-1] for triplet in found_triplets]
            rels = list(set(rels))
            return rels

        if what_return == "triplets":
            return found_triplets

        if what_return == "objects":
            if direction == "forw":
                objects = [triplet[2] for triplet in found_triplets]
            if direction == "backw":
                objects = [triplet[0] for triplet in found_triplets]

            return objects
