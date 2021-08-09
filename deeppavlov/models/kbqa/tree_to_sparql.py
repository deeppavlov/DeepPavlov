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

from io import StringIO
from typing import Any, List, Tuple, Dict, Union
from logging import getLogger
from collections import defaultdict

import numpy as np
import pymorphy2
import re
from scipy.sparse import csr_matrix
from udapi.block.read.conllu import Conllu
from udapi.core.node import Node

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register('ru_adj_to_noun')
class RuAdjToNoun:
    """
        Class for converting an adjective in Russian to the corresponding noun, for example:
        "московский" -> "Москва", "африканский" -> "Африка"
    """

    def __init__(self, freq_dict_filename: str, candidate_nouns: int = 10, **kwargs):
        """

        Args:
            freq_dict_filename: file with the dictionary of Russian words with the corresponding frequencies
            candidate_nouns: how many candidate nouns to leave after search
            **kwargs:
        """
        self.candidate_nouns = candidate_nouns
        alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя-"
        self.alphabet_length = len(alphabet)
        self.max_word_length = 24
        self.letter_nums = {letter: num for num, letter in enumerate(alphabet)}
        with open(str(expand_path(freq_dict_filename)), 'r') as fl:
            lines = fl.readlines()
        pos_freq_dict = defaultdict(list)
        for line in lines:
            line_split = line.strip('\n').split('\t')
            if re.match("[\d]+\.[\d]+", line_split[2]):
                pos_freq_dict[line_split[1]].append((line_split[0], float(line_split[2])))
        self.nouns_with_freq = pos_freq_dict["s.PROP"]
        self.adj_set = set([word for word, freq in pos_freq_dict["a"]])
        self.nouns = [noun[0] for noun in self.nouns_with_freq]
        self.matrix = self.make_sparse_matrix(self.nouns).transpose()
        self.morph = pymorphy2.MorphAnalyzer()

    def search(self, word: str):
        word = self.morph.parse(word)[0]
        word = word.normal_form
        if word in self.adj_set:
            q_matrix = self.make_sparse_matrix([word])
            scores = q_matrix * self.matrix
            scores = np.squeeze(scores.toarray() + 0.0001)
            indices = np.argsort(-scores)[:self.candidate_nouns]
            scores = list(scores[indices])
            candidates = [self.nouns_with_freq[indices[i]] + (scores[i],) for i in range(len(indices))]
            candidates = sorted(candidates, key=lambda x: x[1] * x[2], reverse=True)
            log.debug(f"AdjToNoun, found nouns: {candidates}")
            if candidates[0][2] > 2.5:
                return candidates[0][0]
        return ""

    def make_sparse_matrix(self, words: List[str]):
        indptr = []
        indices = []
        data = []

        total_length = 0

        for n, word in enumerate(words):
            indptr.append(total_length)
            for cnt, letter in enumerate(word.lower()):
                col = self.alphabet_length * cnt + self.letter_nums[letter]
                indices.append(col)
                init_value = 1.0 - cnt * 0.05
                if init_value < 0:
                    init_value = 0
                data.append(init_value)
            total_length += len(word)

        indptr.append(total_length)

        data = np.array(data)
        indptr = np.array(indptr)
        indices = np.array(indices)

        matrix = csr_matrix((data, indices, indptr), shape=(len(words), self.max_word_length * self.alphabet_length))

        return matrix


@register('tree_to_sparql')
class TreeToSparql(Component):
    """
        Class for building of sparql query template using syntax parser
    """

    def __init__(self, sparql_queries_filename: str, lang: str = "rus", adj_to_noun: RuAdjToNoun = None, **kwargs):
        """

        Args:
            sparql_queries_filename: file with sparql query templates
            lang: english or russian
            adj_to_noun: component deeppavlov.models.kbqa.tree_to_sparql:RuAdjToNoun
            **kwargs:
        """
        self.lang = lang
        if self.lang == "rus":
            self.q_pronouns = {"какой", "какая", "какое", "каком", "каким", "какую", "кто", "что", "как", "когда",
                               "где", "чем", "сколько"}
            self.how_many = "сколько"
            self.change_root_tokens = {"каким был", "какой была"}
            self.first_tokens = {"первый"}
            self.last_tokens = {"последний"}
            self.begin_tokens = {"начинать", "начать"}
            self.end_tokens = {"завершить", "завершать", "закончить"}
            self.ranking_tokens = {"самый"}
        elif self.lang == "eng":
            self.q_pronouns = {"what", "who", "how", "when", "where", "which"}
            self.how_many = "how many"
            self.change_root_tokens = ""
            self.first_tokens = {"first"}
            self.last_tokens = {"last"}
            self.begin_tokens = set()
            self.end_tokens = set()
            self.ranking_tokens = set()
        else:
            raise ValueError(f"unsupported language {lang}")
        self.sparql_queries_filename = expand_path(sparql_queries_filename)
        self.template_queries = read_json(self.sparql_queries_filename)
        self.adj_to_noun = adj_to_noun
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, syntax_tree_batch: List[str],
                 positions_batch: List[List[List[int]]]) -> Tuple[
        List[str], List[List[str]], List[List[str]], List[List[str]]]:
        log.debug(f"positions of entity tokens {positions_batch}")
        query_nums_batch = []
        entities_dict_batch = []
        types_dict_batch = []
        questions_batch = []
        count = False
        for syntax_tree, positions in zip(syntax_tree_batch, positions_batch):
            log.debug(f"\n{syntax_tree}")
            tree = Conllu(filehandle=StringIO(syntax_tree)).read_tree()
            root = self.find_root(tree)
            tree_desc = tree.descendants
            unknown_node = ""
            if root:
                log.debug(f"syntax tree info, root: {root.form}")
                unknown_node, unknown_branch = self.find_branch_with_unknown(root)
            positions = [num for position in positions for num in position]
            if unknown_node:
                log.debug(f"syntax tree info, unknown node: {unknown_node.form}, unknown branch: {unknown_branch.form}")
                log.debug(f"wh_leaf: {self.wh_leaf}")
                clause_node, clause_branch = self.find_clause_node(root, unknown_branch)
                if clause_node:
                    log.debug(f"clause node {clause_node.form}")
                else:
                    log.debug(f"clause node not found")
                modifiers, clause_modifiers = self.find_modifiers_of_unknown(unknown_node)
                modifiers_debug_list = []
                for modifier in modifiers:
                    if isinstance(modifier, str):
                        modifiers_debug_list.append(modifier)
                    else:
                        modifiers_debug_list.append(modifier.form)
                log.debug(f"modifiers: {' '.join(modifiers_debug_list)}")
                if f"{tree_desc[0].form.lower()} {tree_desc[1].form.lower()}" in self.change_root_tokens:
                    new_root = root.children[0]
                else:
                    new_root = root
                root_desc = defaultdict(list)
                for node in new_root.children:
                    if node.deprel not in ["punct", "advmod", "cop", "mark"]:
                        if node == unknown_branch:
                            root_desc[node.deprel].append(node)
                        else:
                            if self.find_entities(node, positions, cut_clause=False) or \
                                    (self.find_year_or_number(node) and node.deprel in ["obl", "nummod"]):
                                root_desc[node.deprel].append(node)

                if root.form.lower() == self.how_many or ("nsubj" in root_desc.keys() and
                                                          self.how_many in [nd.form.lower() for nd in
                                                                            root_desc["nsubj"]]):
                    count = True
                log.debug(f"root_desc {root_desc.keys()}")
                appos_token_nums = sorted(self.find_appos_tokens(root, []))
                appos_tokens = [elem.form for elem in tree_desc if elem.ord in appos_token_nums]
                clause_token_nums = sorted(self.find_clause_tokens(root, clause_node, []))
                clause_tokens = [elem.form for elem in tree_desc if elem.ord in clause_token_nums]
                log.debug(f"appos tokens: {appos_tokens}")
                log.debug(f"clause_tokens: {clause_tokens}")
                self.root_entity = False
                if root.ord - 1 in positions:
                    self.root_entity = True

                temporal_order = self.find_first_last(new_root)
                new_root_nf = self.morph.parse(new_root.form)[0].normal_form
                if new_root_nf in self.begin_tokens or new_root_nf in self.end_tokens:
                    temporal_order = new_root_nf
                ranking_tokens = self.find_ranking_tokens(new_root)
                query_nums, entities_dict, types_dict = self.build_query(new_root, unknown_branch, root_desc,
                                                                         unknown_node, modifiers, clause_modifiers,
                                                                         clause_node, positions, count,
                                                                         temporal_order, ranking_tokens)

                if self.lang == "rus":
                    if ranking_tokens:
                        question = []
                        for node in tree.descendants:
                            if node.ord in ranking_tokens or node.form.lower() in self.q_pronouns:
                                question.append(self.morph.parse(node.form)[0].normal_form)
                            else:
                                question.append(node.form)
                        question = ' '.join(question)
                    else:
                        question = ' '.join([node.form for node in tree.descendants
                                             if
                                             (node.ord not in appos_token_nums or node.ord not in clause_token_nums)])
                else:
                    question = ' '.join([node.form for node in tree.descendants])
                log.debug(f"sanitized question: {question}")
                query_nums_batch.append(query_nums)
                entities_dict_batch.append(entities_dict)
                types_dict_batch.append(types_dict)
                questions_batch.append(question)
        return questions_batch, query_nums_batch, entities_dict_batch, types_dict_batch

    def find_root(self, tree: Node) -> Node:
        for node in tree.descendants:
            if node.deprel == "root" and node.children:
                return node

    def find_branch_with_unknown(self, root: Node) -> Tuple[str, str]:
        self.wh_leaf = False
        self.one_chain = False

        if root.form.lower() in self.q_pronouns:
            if "nsubj" in [node.deprel for node in root.children] or root.form.lower() in self.how_many:
                self.one_chain = True
            else:
                for node in root.children:
                    if node.deprel == "nsubj":
                        return node, node

        if not self.one_chain:
            for node in root.children:
                if node.form.lower() in self.q_pronouns:
                    if node.children:
                        for child in node.children:
                            if child.deprel in ["nmod", "obl"]:
                                return child, node
                    else:
                        self.wh_leaf = True
                else:
                    for child in node.descendants:
                        if child.form.lower() in self.q_pronouns:
                            return child.parent, node

        if self.wh_leaf or self.one_chain:
            for node in root.children:
                if node.deprel in ["nsubj", "obl", "obj", "nmod", "xcomp"] and node.form.lower() not in self.q_pronouns:
                    return node, node

        return "", ""

    def find_modifiers_of_unknown(self, node: Node) -> Tuple[List[Union[str, Any]], list]:
        modifiers = []
        clause_modifiers = []
        for mod in node.children:
            if mod.deprel in ["amod", "nmod"] or (mod.deprel == "appos" and mod.children):
                noun_mod = ""
                if self.adj_to_noun:
                    noun_mod = self.adj_to_noun.search(mod.form)
                if noun_mod:
                    modifiers.append(noun_mod)
                else:
                    modifiers.append(mod)
            if mod.deprel == "acl":
                clause_modifiers.append(mod)
        return modifiers, clause_modifiers

    def find_clause_node(self, root: Node, unknown_branch: Node) -> Tuple[str, str]:
        for node in root.children:
            if node.deprel == "obl" and node != unknown_branch:
                for elem in node.children:
                    if elem.deprel == "acl":
                        return elem, node
        return "", ""

    def find_named_entity(self, node: Node, conj_list: List[Node], desc_list: List[Tuple[str, int]],
                          positions: List[int], cut_clause: bool) -> List[Tuple[str, int]]:
        if node.children:
            if self.find_nmod_appos(node, positions):
                used_desc = [elem for elem in node.children if elem.deprel == "appos"]
            else:
                used_desc = node.children

            for elem in used_desc:
                if self.check_node(elem, conj_list, cut_clause):
                    desc_list = self.find_named_entity(elem, conj_list, desc_list, positions, cut_clause)
        log.debug(f"find_named_entity: node.ord, {node.ord - 1}, {node.form}, positions, {positions}")
        log.debug(f"find nmod appos {self.find_nmod_appos(node, positions)}")
        if node.ord - 1 in positions:
            initials_3 = re.findall("([А-Яа-я]{1}\.)([А-Яа-я]{1}\.)([А-Яа-я]{3,15})", node.form)
            initials_2 = re.findall("([А-Яа-я]{1}\.)([А-Яа-я]{3,15})", node.form)
            if initials_3:
                entity_substring = ' '.join(initials_3[0])
            elif initials_2:
                entity_substring = ' '.join(initials_2[0])
            else:
                entity_substring = node.form
            desc_list.append((entity_substring, node.ord))

        return desc_list

    def check_node(self, elem: Node, conj_list: List[Node], cut_clause: bool) -> bool:
        """
        This function defines whether to go deeper in the syntactic tree to look for named entity tokens
        If all the conditions are true, then we recursively look for named entity tokens in elem's descendants.
        Args:
            elem: node of the syntactic tree for which we decide whether to look for named entities in its descendants
            conj_list: list of nodes, connected with the "elem" node with "conj" deprel
            cut_clause: if cut_clause is True, we do not want to look for named entities in adjective clauses ("acl")
        """
        move_deeper = False
        if not cut_clause or (cut_clause and elem.deprel != "acl"):
            if elem not in conj_list:
                if elem.deprel != "appos" or \
                        (elem.deprel == "appos"
                         and (not elem.children or
                              (len(elem.children) == 1 and elem.children[0].deprel in ["flat:name", "parataxis"]) or
                              (len(elem.children) > 1 and {"«", '"', '``', '('} & {nd.form for nd in
                                                                                   elem.descendants}))):
                    move_deeper = True
        return move_deeper

    def find_conj(self, node: Node, conj_list: List[Node], positions: List[int], cut_clause: bool) -> List[Node]:
        if node.children:
            for elem in node.children:
                if not cut_clause or (cut_clause and elem.deprel != "acl"):
                    conj_list = self.find_conj(elem, conj_list, positions, cut_clause)

        if node.deprel == "conj":
            conj_in_ner = False
            for elem in node.children:
                if elem.deprel == "cc" and (elem.ord - 1) in positions:
                    conj_in_ner = True
            if not conj_in_ner:
                conj_list.append(node)

        return conj_list

    def find_entities(self, node: Node, positions: List[int], cut_clause: bool = True) -> List[str]:
        entities_list = []
        conj_list = self.find_conj(node, [], positions, cut_clause)
        entity = self.find_entity(node, conj_list, positions, cut_clause)
        if entity:
            entities_list.append(entity)
        if conj_list:
            for conj_node in conj_list:
                curr_conj_list = [elem for elem in conj_list if elem != conj_node]
                entity = self.find_entity(conj_node, curr_conj_list, positions, cut_clause)
                entities_list.append(entity)
        log.debug(f"found_entities, {entities_list}")
        return entities_list

    def find_entity(self, node: Node, conj_list: List[Node], positions: List[int], cut_clause: bool) -> str:
        grounded_entity_tokens = self.find_named_entity(node, conj_list, [], positions, cut_clause)
        grounded_entity = sorted(grounded_entity_tokens, key=lambda x: x[1])
        grounded_entity = " ".join([entity[0] for entity in grounded_entity])
        return grounded_entity

    def find_nmod_appos(self, node: Node, positions: List[int]) -> bool:
        node_desc = {elem.deprel: elem for elem in node.children}
        node_deprels = sorted([elem.deprel for elem in node.children if elem.deprel != "case"])
        if node.ord - 1 in positions:
            return False
        elif node_deprels == ["appos", "nmod"] and node_desc["appos"].ord - 1 in positions \
                and node_desc["nmod"].ord in positions:
            return True
        return False

    def find_year_or_number(self, node: Node) -> bool:
        found = False
        for elem in node.descendants:
            if elem.deprel == "nummod":
                return True
        return found

    def find_appos_tokens(self, node: Node, appos_token_nums: List[int]) -> List[int]:
        for elem in node.children:
            if elem.deprel == "appos" and (len(elem.descendants) > 1 and
                                           not {"«", '"', '``', '('} & {nd.form for nd in elem.descendants} or
                                           (len(elem.descendants) == 1 and elem.descendants[0].deprel != "flat:name")):
                appos_token_nums.append(elem.ord)
                for desc in elem.descendants:
                    appos_token_nums.append(desc.ord)
            else:
                appos_token_nums = self.find_appos_tokens(elem, appos_token_nums)
        return appos_token_nums

    def find_clause_tokens(self, node: Node, clause_node: Node, clause_token_nums: List[int]) -> List[int]:
        for elem in node.children:
            if elem != clause_node and elem.deprel == "acl":
                clause_token_nums.append(elem.ord)
                for desc in elem.descendants:
                    clause_token_nums.append(desc.ord)
            else:
                clause_token_nums = self.find_appos_tokens(elem, clause_token_nums)
        return clause_token_nums

    def find_first_last(self, node: Node) -> str:
        first_or_last = ""
        nodes = [node]
        while nodes:
            for node in nodes:
                node_desc = defaultdict(set)
                for elem in node.children:
                    parsed_elem = self.morph.parse(elem.form.lower())[0].inflect({"masc", "sing", "nomn"})
                    if parsed_elem is not None:
                        node_desc[elem.deprel].add(parsed_elem.word)
                    else:
                        node_desc[elem.deprel].add(elem.form)
                if "amod" in node_desc.keys() and "nmod" in node_desc.keys() and \
                        node_desc["amod"].intersection(self.first_tokens | self.last_tokens):
                    first_or_last = ' '.join(node_desc["amod"].intersection(self.first_tokens | self.last_tokens))
                    return first_or_last
            nodes = [elem for node in nodes for elem in node.children]
        return first_or_last

    def find_ranking_tokens(self, node: Node) -> list:
        ranking_tokens = []
        for elem in node.descendants:
            if self.morph.parse(elem.form)[0].normal_form in self.ranking_tokens:
                ranking_tokens.append(elem.ord)
                ranking_tokens.append(elem.parent.ord)
                return ranking_tokens
        return ranking_tokens

    def build_query(self, root: Node, unknown_branch: Node, root_desc: Dict[str, List[Node]],
                    unknown_node: Node, unknown_modifiers: List[Node], clause_modifiers: List[Node],
                    clause_node: Node, positions: List[int],
                    count: bool = False, temporal_order: str = "",
                    ranking_tokens: List[str] = None) -> Tuple[List[str], List[str], List[str]]:
        query_nums = []
        grounded_entities_list = []
        types_list = []
        modifiers_list = []
        qualifier_entities_list = []
        found_year_or_number = False
        order = False
        root_desc_deprels = []
        for key in root_desc.keys():
            for i in range(len(root_desc[key])):
                root_desc_deprels.append(key)
        root_desc_deprels = sorted(root_desc_deprels)
        log.debug(f"build_query: root_desc.keys, {root_desc_deprels}, positions {positions}")
        log.debug(f"temporal order {temporal_order}, ranking tokens {ranking_tokens}")
        if root_desc_deprels in [["nsubj", "obl"],
                                 ["nsubj", "obj"],
                                 ["nsubj", "xcomp"],
                                 ["obj", "xcomp"],
                                 ["nmod", "nsubj"],
                                 ["obj", "obl"],
                                 ["iobj", "nsubj"],
                                 ["acl", "nsubj"],
                                 ["cop", "nsubj", "obl"],
                                 ["obj"],
                                 ["obl"],
                                 ["nmod"],
                                 ["xcomp"],
                                 ["nsubj"]]:
            if self.wh_leaf or self.one_chain:
                if root_desc_deprels == ["nsubj", "obl"]:
                    grounded_entities_list = self.find_entities(root_desc["nsubj"][0], positions, cut_clause=True)
                    if not grounded_entities_list:
                        grounded_entities_list = self.find_entities(root_desc["obl"][0], positions, cut_clause=True)
                else:
                    for nodes in root_desc.values():
                        if nodes[0].form not in self.q_pronouns:
                            grounded_entities_list = self.find_entities(nodes[0], positions, cut_clause=True)
                            if grounded_entities_list:
                                break
            else:
                if self.root_entity:
                    grounded_entities_list = [root.form]
                for nodes in root_desc.values():
                    if nodes[0] != unknown_branch:
                        grounded_entities_list = self.find_entities(nodes[0], positions, cut_clause=True)
                        if grounded_entities_list:
                            type_entity = unknown_node.form
                            types_list.append(type_entity)
                            break

                if unknown_modifiers:
                    for n, modifier in enumerate(unknown_modifiers):
                        if isinstance(modifier, str):
                            modifiers_list.append(modifier)
                        else:
                            modifier_entities = self.find_entities(modifier, positions, cut_clause=True)
                            if modifier_entities:
                                modifiers_list += modifier_entities
                if clause_modifiers:
                    found_year_or_number = self.find_year_or_number(clause_modifiers[0])
                    if found_year_or_number:
                        query_nums.append("0")
                    qualifier_entities_list = self.find_entities(clause_modifiers[0], positions, cut_clause=True)

        if root_desc_deprels == ["nsubj", "obl", "obl"]:
            grounded_entities_list = self.find_entities(root_desc["nsubj"][0], positions, cut_clause=True)
            for node in root_desc["obl"]:
                if node == unknown_branch:
                    types_list.append(node.form)
                else:
                    grounded_entities_list += self.find_entities(node, positions, cut_clause=True)

        if root_desc_deprels == ["obj", "xcomp"]:
            grounded_entities_list = self.find_entities(root_desc["xcomp"][0], positions, cut_clause=True)

        if root_desc_deprels == ["nsubj", "obj", "obl"] or root_desc_deprels == ["obj", "obl"] and self.wh_leaf:
            found_year_or_number = self.find_year_or_number(root_desc["obl"][0])
            if self.wh_leaf:
                grounded_entities_list = self.find_entities(root_desc["obl"][0], positions, cut_clause=True)
                qualifier_entities_list = self.find_entities(root_desc["obj"][0], positions, cut_clause=True)
            else:
                grounded_entities_list = self.find_entities(root_desc["obj"][0], positions, cut_clause=True)
                if found_year_or_number:
                    query_nums.append("0")

        if clause_node:
            for node in clause_node.children:
                if node.deprel == "obj":
                    grounded_entities_list = self.find_entities(node, positions, cut_clause=False)
                if self.find_year_or_number(node):
                    query_nums.append("0")

            if not self.wh_leaf:
                type_entity = unknown_node.form
                types_list.append(type_entity)

        if root_desc_deprels == ["nmod", "nmod"]:
            grounded_entities_list = self.find_entities(root_desc["nmod"][0], positions, cut_clause=True)
            modifiers_list = self.find_entities(root_desc["nmod"][1], positions, cut_clause=True)

        if root_desc_deprels == ["nmod", "nsubj", "nummod"]:
            if not self.wh_leaf:
                grounded_entities_list = self.find_entities(root_desc["nmod"][0], positions, cut_clause=True)
                found_year_or_number = self.find_year_or_number(root_desc["nummod"][0])

        if temporal_order:
            for deprel in root_desc:
                for node in root_desc[deprel]:
                    entities = self.find_entities(node, positions, cut_clause=True)
                    if entities:
                        grounded_entities_list = entities
                        break
                if grounded_entities_list:
                    break
            if temporal_order in self.first_tokens:
                query_nums.append("22")
                query_nums.append("23")
            if temporal_order in self.last_tokens:
                query_nums.append("24")
            if temporal_order in self.begin_tokens:
                query_nums.append("22")
                query_nums.append("25")
            if temporal_order in self.end_tokens:
                query_nums.append("24")
                query_nums.append("26")

        if count:
            grounded_entities_list = self.find_entities(root, positions, cut_clause=True)

        entities_list = grounded_entities_list + qualifier_entities_list + modifiers_list

        grounded_entities_length = len(grounded_entities_list)
        types_length = len(types_list)
        modifiers_length = len(modifiers_list)
        qualifiers_length = len(qualifier_entities_list)
        if qualifiers_length > 0 or modifiers_length or count:
            types_length = 0

        if not temporal_order:
            for num, template in self.template_queries.items():
                if [grounded_entities_length, types_length, modifiers_length,
                    qualifiers_length, found_year_or_number, count, order] == list(
                    template["syntax_structure"].values()):
                    query_nums.append(num)

        log.debug(f"tree_to_sparql, grounded entities {grounded_entities_list}")
        log.debug(f"tree_to_sparql, types {types_list}")
        log.debug(f"tree_to_sparql, modifier entities {modifiers_list}")
        log.debug(f"tree_to_sparql, qualifier entities {qualifier_entities_list}")
        log.debug(f"tree to sparql, query nums {query_nums}")

        return query_nums, entities_list, types_list
