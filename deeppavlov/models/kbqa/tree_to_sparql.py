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

import copy
import re
from collections import defaultdict
from io import StringIO
from logging import getLogger
from typing import Any, List, Tuple, Dict, Union

import spacy
from navec import Navec
from razdel import tokenize
from slovnet import Syntax
from udapi.block.read.conllu import Conllu
from udapi.core.node import Node

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.kbqa.ru_adj_to_noun import RuAdjToNoun
from deeppavlov.models.kbqa.utils import preprocess_template_queries

log = getLogger(__name__)


@register('slovnet_syntax_parser')
class SlovnetSyntaxParser(Component, Serializable):
    """Class for syntax parsing using Slovnet library"""

    def __init__(self, load_path: str, navec_filename: str, syntax_parser_filename: str, tree_patterns_filename: str,
                 **kwargs):
        super().__init__(save_path=None, load_path=load_path)
        self.navec_filename = expand_path(navec_filename)
        self.syntax_parser_filename = expand_path(syntax_parser_filename)
        self.tree_patterns = read_json(expand_path(tree_patterns_filename))
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.pronouns = {"q_pronouns": {"какой", "какая", "какое", "каком", "каким", "какую", "кто", "что", "как",
                                        "когда", "где", "чем", "сколько"},
                         "how_many": {"сколько"}}
        self.first_tokens = {"первый", "первая", "первое"}
        self.nlp = spacy.load("ru_core_news_sm")
        self.load()

    def load(self) -> None:
        navec = Navec.load(self.navec_filename)
        self.syntax = Syntax.load(self.syntax_parser_filename)
        self.syntax.navec(navec)

    def save(self) -> None:
        pass

    def preprocess_sentences(self, sentences, entity_offsets_batch):
        sentences_tokens_batch, replace_dict_batch = [], []
        for sentence, entity_offsets in zip(sentences, entity_offsets_batch):
            if sentence.islower():
                for start, end in entity_offsets:
                    entity_old = sentence[start:end]
                    if entity_old:
                        entity_new = f"{entity_old[0].upper()}{entity_old[1:]}"
                        sentence = sentence.replace(entity_old, entity_new)
                sentence = f"{sentence[0].upper()}{sentence[1:]}"
            names3 = re.findall(r"([\w]{1}\.)([ ]?)([\w]{1}\.)([ ])([\w]{3,})", sentence)
            replace_dict = {}
            for name in names3:
                names_str = "".join(name)
                replace_dict[name[-1]] = (names_str, "name")
                sentence = sentence.replace(names_str, name[-1])
            names2 = re.findall(r"([\w]{1}\.)([ ])([\w]{3,})", sentence)
            for name in names2:
                names_str = "".join(name)
                replace_dict[name[-1]] = (names_str, "name")
                sentence = sentence.replace(names_str, name[-1])
            works_of_art = re.findall(r'(["«])(.*?)(["»])', sentence)
            for symb_start, work_of_art, symb_end in works_of_art:
                work_of_art_tokens = re.findall(self.re_tokenizer, work_of_art)
                if len(work_of_art.split()) > 1:
                    short_substr = ""
                    for tok in work_of_art_tokens:
                        if self.nlp(tok)[0].pos_ == "NOUN":
                            short_substr = tok
                            break
                    if not short_substr:
                        short_substr = work_of_art_tokens[0]
                    replace_dict[short_substr] = (work_of_art, "name")
                    sentence = sentence.replace(work_of_art, short_substr)
            while True:
                tokens = sentence.split()
                found_substr = False
                for i in range(len(tokens) - 2):
                    found = True
                    for j in range(i, i + 3):
                        if len(tokens[j]) < 2 or tokens[j][0] in '("' or tokens[j][-1] in '"),.?':
                            found = False
                    if found and i > 0:
                        token_tags = [self.nlp(tokens[j])[0].pos_ for j in range(i, i + 3)]
                        lemm_tokens = {self.nlp(tok)[0].lemma_ for tok in tokens[i:i + 3]}
                        if token_tags == ["DET", "DET", "NOUN"] and not lemm_tokens & self.first_tokens:
                            long_substr = " ".join(tokens[i:i + 3])
                            replace_dict[tokens[i + 2]] = (long_substr, "adj")
                            sentence = sentence.replace(long_substr, tokens[i + 2])
                            found_substr = True
                    if found_substr:
                        break
                if not found_substr:
                    break
            sentence_tokens = [tok.text for tok in tokenize(sentence)]
            sentences_tokens_batch.append(sentence_tokens)
            log.debug(f"replace_dict: {replace_dict} --- sentence: {sentence_tokens}")
            replace_dict_batch.append(replace_dict)
        return sentences_tokens_batch, replace_dict_batch

    def get_markup(self, proc_syntax_batch, replace_dict_batch):
        markup_batch = []
        for proc_syntax, replace_dict in zip(proc_syntax_batch, replace_dict_batch):
            markup_list = []
            for elem in proc_syntax.tokens:
                markup_list.append({"id": elem.id, "text": elem.text, "head_id": int(elem.head_id), "rel": elem.rel})
            ids, words, head_ids, rels = self.get_elements(markup_list)
            head_ids, markup_list = self.correct_cycle(ids, head_ids, rels, markup_list)
            for substr in replace_dict:
                substr_full, substr_type = replace_dict[substr]
                found_n = -1
                for n, markup_elem in enumerate(markup_list):
                    if markup_elem["text"] == substr:
                        found_n = n
                if found_n > -1:
                    before_markup_list = copy.deepcopy(markup_list[:found_n])
                    after_markup_list = copy.deepcopy(markup_list[found_n + 1:])
                    substr_tokens = [tok.text for tok in tokenize(substr_full)]
                    new_markup_list = []
                    if substr_type == "name":
                        for j in range(len(substr_tokens)):
                            new_markup_elem = {"id": str(found_n + j + 1), "text": substr_tokens[j]}
                            if j == 0:
                                new_markup_elem["rel"] = markup_list[found_n]["rel"]
                                if int(markup_list[found_n]["head_id"]) < found_n + 1:
                                    new_markup_elem["head_id"] = markup_list[found_n]["head_id"]
                                else:
                                    new_markup_elem["head_id"] = str(int(markup_list[found_n]["head_id"]) + len(
                                        substr_tokens) - 1)
                            else:
                                new_markup_elem["rel"] = "flat:name"
                                new_markup_elem["head_id"] = str(found_n + 1)
                            new_markup_list.append(new_markup_elem)
                    elif substr_type == "adj":
                        for j in range(len(substr_tokens)):
                            new_elem = {"id": str(found_n + j + 1), "text": substr_tokens[j]}
                            if j == len(substr_tokens) - 1:
                                new_elem["rel"] = markup_list[found_n]["rel"]
                                if markup_list[found_n]["head_id"] < found_n + 1:
                                    new_elem["head_id"] = markup_list[found_n]["head_id"]
                                else:
                                    new_elem["head_id"] = markup_list[found_n]["head_id"] + len(substr_tokens) - 1
                            else:
                                new_elem["rel"] = "amod"
                                new_elem["head_id"] = str(found_n + len(substr_tokens))
                            new_markup_list.append(new_elem)

                    for j in range(len(before_markup_list)):
                        if int(before_markup_list[j]["head_id"]) > found_n + 1:
                            before_markup_list[j]["head_id"] = int(before_markup_list[j]["head_id"]) + \
                                                               len(substr_tokens) - 1
                        if before_markup_list[j]["head_id"] == found_n + 1 and substr_type == "adj":
                            before_markup_list[j]["head_id"] = found_n + len(substr_tokens)
                    for j in range(len(after_markup_list)):
                        after_markup_list[j]["id"] = str(int(after_markup_list[j]["id"]) + len(substr_tokens) - 1)
                        if int(after_markup_list[j]["head_id"]) > found_n + 1:
                            after_markup_list[j]["head_id"] = int(after_markup_list[j]["head_id"]) + \
                                                              len(substr_tokens) - 1
                        if after_markup_list[j]["head_id"] == found_n + 1 and substr_type == "adj":
                            after_markup_list[j]["head_id"] = found_n + len(substr_tokens)

                    markup_list = before_markup_list + new_markup_list + after_markup_list
            for j in range(len(markup_list)):
                markup_list[j]["head_id"] = str(markup_list[j]["head_id"])
            markup_batch.append(markup_list)
        return markup_batch

    def find_cycle(self, ids, head_ids):
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i < j and head_ids[j] == str(i + 1) and head_ids[i] == str(j + 1):
                    return i + 1
        return -1

    def correct_markup(self, words, head_ids, rels, root_n):
        if len(words) > 3:
            pos = [self.nlp(words[i])[0].pos_ for i in range(len(words))]
            for tree_pattern in self.tree_patterns:
                first_word = tree_pattern.get("first_word", "")
                (r_start, r_end), rel_info = tree_pattern.get("rels", [[0, 0], ""])
                (p_start, p_end), pos_info = tree_pattern.get("pos", [[0, 0], ""])
                if (not first_word or words[0].lower() in self.pronouns[first_word]) \
                        and (not rel_info or rels[r_start:r_end] == rel_info) \
                        and (not pos_info or pos[p_start:p_end] == pos_info):
                    for ind, deprel in tree_pattern.get("rel_ids", {}).items():
                        rels[int(ind)] = deprel
                    for ind, head_id in tree_pattern.get("head_ids", {}).items():
                        head_ids[int(ind)] = head_id
                    root_n = tree_pattern["root_n"]
                    break
            if words[0].lower() in {"какой", "какая", "какое"} and rels[:3] == ["det", "obj", "root"] \
                    and pos[1:3] == ["NOUN", "VERB"] and "nsubj" not in rels:
                rels[1] = "nsubj"
        return head_ids, rels, root_n

    def find_root(self, rels):
        root_n = -1
        for n in range(len(rels)):
            if rels[n] == "root":
                root_n = n + 1
                break
        return root_n

    def get_elements(self, markup_elem):
        ids, words, head_ids, rels = [], [], [], []
        for elem in markup_elem:
            ids.append(elem["id"])
            words.append(elem["text"])
            head_ids.append(elem["head_id"])
            rels.append(elem["rel"])
        return ids, words, head_ids, rels

    def correct_cycle(self, ids, head_ids, rels, markup_elem):
        cycle_num = -1
        for n, (elem_id, head_id) in enumerate(zip(ids, head_ids)):
            if str(head_id) == str(elem_id):
                cycle_num = n
        root_n = self.find_root(rels)
        if cycle_num > 0 and root_n > -1:
            head_ids[cycle_num] = root_n
        markup_elem[cycle_num]["head_id"] = root_n
        return head_ids, markup_elem

    def process_markup(self, markup_batch):
        processed_markup_batch = []
        for markup_elem in markup_batch:
            processed_markup = []
            ids, words, head_ids, rels = self.get_elements(markup_elem)
            if "root" not in {rel.lower() for rel in rels}:
                found_root = False
                for n, (elem_id, head_id) in enumerate(zip(ids, head_ids)):
                    if elem_id == head_id:
                        rels[n] = "root"
                        head_ids[n] = 0
                        found_root = True
                if not found_root:
                    for n in range(len(ids)):
                        if rels[n] == "nsubj":
                            rels[n] = "root"
                            head_ids[n] = 0
                            found_root = True
                if not found_root:
                    for n in range(len(ids)):
                        if self.nlp(words[n])[0].pos_ == "VERB":
                            rels[n] = "root"
                            head_ids[n] = 0

            root_n = self.find_root(rels)
            head_ids, rels, root_n = self.correct_markup(words, head_ids, rels, root_n)
            if words[-1] == "?" and -1 < root_n != head_ids[-1]:
                head_ids[-1] = root_n

            head_ids, markup_elem = self.correct_cycle(ids, head_ids, rels, markup_elem)
            i = self.find_cycle(ids, head_ids)
            if i == 1 and root_n > -1:
                head_ids[i - 1] = root_n
            for elem_id, word, head_id, rel in zip(ids, words, head_ids, rels):
                processed_markup.append(f"{elem_id}\t{word}\t_\t_\t_\t_\t{head_id}\t{rel}\t_\t_")
            processed_markup_batch.append("\n".join(processed_markup))
        return processed_markup_batch

    def __call__(self, sentences, entity_offsets_batch):
        sentences_tokens_batch, substr_dict_batch = self.preprocess_sentences(sentences, entity_offsets_batch)
        proc_syntax_batch = list(self.syntax.map(sentences_tokens_batch))
        markup_batch = self.get_markup(proc_syntax_batch, substr_dict_batch)
        processed_markup_batch = self.process_markup(markup_batch)
        return processed_markup_batch


@register('tree_to_sparql')
class TreeToSparql(Component):
    """
        Class for building of sparql query template using syntax parser
    """

    def __init__(self, sparql_queries_filename: str, syntax_parser: Component, kb_prefixes: Dict[str, str],
                 adj_to_noun: RuAdjToNoun = None, **kwargs):
        """

        Args:
            sparql_queries_filename: file with sparql query templates
            syntax_parser: component for syntactic parsing of the input question
            kb_prefixes: prefixes for entities, relations and types in the knowledge base
            adj_to_noun: component deeppavlov.models.kbqa.tree_to_sparql:RuAdjToNoun
            **kwargs:
        """
        self.q_pronouns = {"какой", "какая", "какое", "каком", "каким", "какую", "кто", "что", "как", "когда",
                           "где", "чем", "сколько"}
        self.how_many = "сколько"
        self.change_root_tokens = {"каким был", "какой была"}
        self.first_tokens = {"первый", "первая", "первое"}
        self.last_tokens = {"последний"}
        self.begin_tokens = {"начинать", "начать"}
        self.end_tokens = {"завершить", "завершать", "закончить"}
        self.ranking_tokens = {"самый"}
        self.date_tokens = {"год", "месяц"}
        self.nlp = spacy.load("ru_core_news_sm")
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.sparql_queries_filename = expand_path(sparql_queries_filename)
        template_queries = read_json(self.sparql_queries_filename)
        self.template_queries = preprocess_template_queries(template_queries, kb_prefixes)
        self.syntax_parser = syntax_parser
        self.adj_to_noun = adj_to_noun

    def __call__(self, questions_batch: List[str], substr_batch: List[List[str]], tags_batch: List[List[str]],
                 offsets_batch: List[List[List[int]]], positions_batch: List[List[List[int]]],
                 probas_batch: List[List[float]]) -> Tuple[
        List[Union[str, Any]], List[Union[List[str], List[Union[str, Any]]]], List[Union[List[str], Any]], List[
            Union[List[Union[str, Any]], Any]], List[Union[List[Union[float, Any]], Any]], List[List[int]], List[
            Union[List[str], List[Any]]]]:
        substr_batch, tags_batch, offsets_batch, positions_batch, probas_batch = \
            self.sort_substr(substr_batch, tags_batch, offsets_batch, positions_batch, probas_batch)
        log.debug(f"substr: {substr_batch} tags: {tags_batch} positions: {positions_batch}")
        query_nums_batch, s_substr_batch, s_tags_batch, s_probas_batch, types_batch = [], [], [], [], []
        entities_to_link_batch = []
        clean_questions_batch = []
        count = False
        for question, substr_list, tags_list, offsets_list, probas_list, positions in \
                zip(questions_batch, substr_batch, tags_batch, offsets_batch, probas_batch, positions_batch):
            entities_dict, probas_dict = {}, {}
            for substr, tag, proba in zip(substr_list, tags_list, probas_list):
                entities_dict[substr.lower()] = tag
                probas_dict[substr.lower()] = proba
            for i in range(len(substr_list)):
                substr = substr_list[i]
                if len(substr) > 2 and ("-" in substr or f"{substr}-" in question) and " - " not in substr:
                    if "-" in substr:
                        length = len(re.findall(self.re_tokenizer, substr))
                    else:
                        length = 3
                    substr_tokens = list(tokenize(substr))
                    positions[i] = [positions[i][j] for j in range(len(substr_tokens))]
                    if i < len(substr_list) - 1:
                        for j in range(i + 1, len(substr_list)):
                            pos_inds = positions[j]
                            pos_inds = [ind - length + 1 for ind in pos_inds]
                            positions[j] = pos_inds

            root, tree, tree_desc, unknown_node, unknown_branch = self.syntax_parse(question, offsets_list)
            query_nums = ["7"]
            s_substr_list = substr_list
            s_tags_list = tags_list
            s_probas_list = probas_list
            types_list = []
            if unknown_node:
                log.debug(f"syntax tree info 1, unknown node: {unknown_node.form}, unkn branch: {unknown_branch.form}")
                log.debug(f"wh_leaf: {self.wh_leaf}")
                clause_node, clause_branch = self.find_clause_node(root, unknown_branch)
                log.debug(f"clause node: {clause_node}")
                tok_and_ord = {node.ord: node for node in tree.descendants}
                appos_token_nums = sorted(self.find_appos_tokens(root, tok_and_ord, []))
                appos_tokens = [elem.form for elem in tree_desc if elem.ord in appos_token_nums]
                clause_token_nums = sorted(self.find_clause_tokens(root, tok_and_ord, clause_node))
                clause_tokens = [elem.form for elem in tree_desc if elem.ord in clause_token_nums]
                log.debug(f"appos tokens: {appos_tokens}")
                log.debug(f"clause_tokens: {clause_tokens}")
                question, ranking_tokens = self.sanitize_question(tree, root, appos_token_nums, clause_token_nums)
                if appos_token_nums or clause_token_nums:
                    root, tree, tree_desc, unknown_node, unknown_branch = self.syntax_parse(question, offsets_list)
                    log.debug(f"syntax tree info 2, unknown node: {unknown_node}, unkn branch: {unknown_branch}")

                if unknown_node:
                    modifiers, clause_modifiers = self.find_modifiers_of_unknown(unknown_node)
                    log.debug(f"modifiers: {modifiers} --- clause modifiers: {[nd.form for nd in clause_modifiers]}")
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
                                if self.find_entities(node, positions) or \
                                        (self.find_year_or_number(node) and node.deprel in ["obl", "nummod"]):
                                    root_desc[node.deprel].append(node)

                    if root.form.lower() == self.how_many or ("nsubj" in root_desc.keys() and
                                                              self.how_many in [nd.form.lower() for nd in
                                                                                root_desc["nsubj"]]):
                        count = True
                    log.debug(f"root_desc {root_desc.keys()}")
                    self.root_entity = False
                    if root.ord - 1 in positions:
                        self.root_entity = True

                    temporal_order = self.find_first_last(new_root)
                    new_root_nf = self.nlp(new_root.form)[0].lemma_
                    if new_root_nf in self.begin_tokens or new_root_nf in self.end_tokens:
                        temporal_order = new_root_nf
                    query_nums, s_substr_list, types_list = self.build_query(new_root, unknown_branch, root_desc,
                                                                             unknown_node, modifiers, clause_modifiers,
                                                                             clause_node, positions, entities_dict,
                                                                             count, temporal_order, ranking_tokens)
                    s_tags_list, s_probas_list = [], []
                    for substr in s_substr_list:
                        substr = substr.replace(" - ", "-")
                        s_tags_list.append(entities_dict.get(substr.lower(), "E"))
                        s_probas_list.append(probas_dict.get(substr.lower(), 1.0))
            clean_questions_batch.append(question)
            if query_nums and s_substr_list:
                entities_to_link = [1 for _ in s_substr_list]
                s_substr_list_lower = [s.lower() for s in s_substr_list]
                for substr, tag, proba in zip(substr_list, tags_list, probas_list):
                    if substr.lower() not in s_substr_list_lower:
                        s_substr_list.append(substr)
                        s_tags_list.append(tag)
                        s_probas_list.append(proba)
                        entities_to_link.append(0)
                s_substr_batch.append(s_substr_list)
                s_tags_batch.append(s_tags_list)
                s_probas_batch.append(s_probas_list)
                entities_to_link_batch.append(entities_to_link)
            else:
                mod_len = 0
                gr_len = 1
                if all([tags_list[i] == tags_list[0] for i in range(len(tags_list))]):
                    gr_len = len(substr_list)
                elif len(substr_list) > 1:
                    mod_len = 1
                for num, template in self.template_queries.items():
                    syntax_info = [gr_len, 0, mod_len, 0, False, False, False]
                    if syntax_info == list(template["syntax_structure"].values()):
                        query_nums.append(num)
                entities_to_link = [1 for _ in s_substr_list]
                s_substr_batch.append(substr_list)
                s_tags_batch.append(tags_list)
                s_probas_batch.append(probas_list)
                entities_to_link_batch.append(entities_to_link)
            query_nums_batch.append(query_nums)
            types_batch.append(types_list)
        log.debug(f"clean_questions: {clean_questions_batch} --- substr: {s_substr_batch} --- tags: {s_tags_batch} "
                  f"--- entities_to_link {entities_to_link_batch} --- types: {types_batch}")
        return clean_questions_batch, query_nums_batch, s_substr_batch, s_tags_batch, s_probas_batch, \
               entities_to_link_batch, types_batch

    def sort_substr(self, substr_batch: List[List[str]], tags_batch: List[List[str]],
                    offsets_batch: List[List[List[int]]], positions_batch: List[List[List[int]]],
                    probas_batch: List[List[float]]) -> Tuple[
        List[List[str]], List[List[str]], List[List[List[int]]], List[List[List[int]]], List[List[float]]]:
        s_substr_batch, s_tags_batch, s_offsets_batch, s_positions_batch, s_probas_batch = [], [], [], [], []
        for substr_list, tags_list, offsets_list, positions_list, probas_list \
                in zip(substr_batch, tags_batch, offsets_batch, positions_batch, probas_batch):
            substr_info = [(substr, tag, offsets, positions, proba) for substr, tag, offsets, positions, proba
                           in zip(substr_list, tags_list, offsets_list, positions_list, probas_list)]
            substr_info = sorted(substr_info, key=lambda x: x[3][0])
            s_substr_batch.append([elem[0] for elem in substr_info])
            s_tags_batch.append([elem[1] for elem in substr_info])
            s_offsets_batch.append([elem[2] for elem in substr_info])
            s_positions_batch.append([elem[3] for elem in substr_info])
            s_probas_batch.append([elem[4] for elem in substr_info])
        return s_substr_batch, s_tags_batch, s_offsets_batch, s_positions_batch, s_probas_batch

    def syntax_parse(self, question: str, entity_offsets_list: List[List[int]]) -> Tuple[
        Union[str, Any], Union[str, Any], Union[str, Any], str, str]:
        syntax_tree = self.syntax_parser([question], [entity_offsets_list])[0]
        log.debug(f"syntax tree: \n{syntax_tree}")
        root, tree, tree_desc, unknown_node, unknown_branch = "", "", "", "", ""
        try:
            tree = Conllu(filehandle=StringIO(syntax_tree)).read_tree()
            root = self.find_root(tree)
            tree_desc = tree.descendants
        except ValueError as e:
            log.warning(f"error in parsing syntax tree, {e}")
        if root:
            unknown_node, unknown_branch = self.find_branch_with_unknown(root)
            log.debug(f"syntax tree info, root: {root.form} unk_node: {unknown_node} unk_branch: {unknown_branch}")
        return root, tree, tree_desc, unknown_node, unknown_branch

    def sanitize_question(self, tree: Node, root: Node, appos_token_nums: List[int], clause_token_nums: List[int]) -> \
            Tuple[str, list]:
        ranking_tokens = self.find_ranking_tokens(root, appos_token_nums, clause_token_nums)
        question_tokens = []
        for node in tree.descendants:
            if node.ord not in appos_token_nums + clause_token_nums:
                if ranking_tokens and (node.ord in ranking_tokens or node.form.lower() in self.q_pronouns):
                    question_tokens.append(self.nlp(node.form)[0].lemma_)
                else:
                    question_tokens.append(node.form)
        question = " ".join(question_tokens)
        log.debug(f"sanitized question: {question}")
        return question, ranking_tokens

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

    def find_entities(self, node: Node, positions: List[List[int]]) -> List[str]:
        node_desc = [(node.form, node.ord, node.parent)] + \
                    [(elem.form, elem.ord, elem.parent) for elem in node.descendants]
        node_desc = sorted(node_desc, key=lambda x: x[1])
        entities_list, heads_list = [], []
        for pos_elem in positions:
            entity, parents = [], []
            for ind in pos_elem:
                for node_elem in node_desc:
                    if ind + 1 == node_elem[1]:
                        entity.append(node_elem[0])
                        parents.append(node_elem[2])
                        break
            if len(entity) == len(pos_elem):
                entity = " ".join(entity).replace(" .", ".")
                entities_list.append(entity)
                heads_list.append(parents[0])
        log.debug(f"node: {node.form} --- found_entities: {entities_list} --- node_desc: {node_desc} --- "
                  f"positions: {positions}")
        return entities_list

    def find_year_or_number(self, node: Node) -> bool:
        found = False
        for elem in node.descendants:
            if elem.deprel == "nummod" or re.findall(r"[\d]{4}", elem.form):
                return True
        return found

    def find_year_constraint(self, node: Node) -> list:
        node_desc = [(node.form, node.ord)] + [(elem.form, elem.ord) for elem in node.descendants]
        node_desc = sorted(node_desc, key=lambda x: x[1])
        desc_text = " ".join([elem[0] for elem in node_desc])
        for symb in ".,:;)":
            desc_text = desc_text.replace(f" {symb}", symb)
        for pattern in [r"в ([\d]{3,4}) году", r"с ([\d]{3,4}) по ([\d]{3,4})"]:
            fnd = re.findall(pattern, desc_text)
            if fnd:
                return fnd
        return []

    def find_appos_tokens(self, node: Node, tok_and_ord: List[Tuple[Node, int]],
                          appos_token_nums: List[int]) -> List[int]:
        for elem in node.children:
            e_desc = elem.descendants
            if elem.deprel == "appos" and elem.ord > 1 and tok_and_ord[elem.ord - 1].deprel == "punct" \
                    and not all([nd.deprel in {"appos", "flat:name"} for nd in e_desc]) \
                    and not ({"«", '"', '``', '('} & {nd.form for nd in e_desc}):
                appos_token_nums.append(elem.ord)
                for desc in elem.descendants:
                    appos_token_nums.append(desc.ord)
            else:
                appos_token_nums = self.find_appos_tokens(elem, tok_and_ord, appos_token_nums)
        return appos_token_nums

    def find_clause_tokens(self, node: Node, tok_and_ord: Dict[int, Node], clause_node: Node) -> List[int]:
        clause_token_nums = []
        for elem in node.children:
            if elem != clause_node and elem.deprel == "acl":
                clause_token_nums.append(elem.ord)
                for desc in elem.descendants:
                    clause_token_nums.append(desc.ord)
            else:
                clause_token_nums = self.find_appos_tokens(elem, tok_and_ord, clause_token_nums)
        return clause_token_nums

    def find_first_last(self, node: Node) -> str:
        first_or_last = ""
        nodes = [node]
        while nodes:
            for node in nodes:
                node_desc = defaultdict(set)
                for elem in node.children:
                    normal_form = self.nlp(elem.form.lower())[0].lemma_
                    node_desc[elem.deprel].add(normal_form)
                log.debug(f"find_first_last {node_desc}")
                if "amod" in node_desc.keys() and "nmod" in node_desc.keys() and \
                        node_desc["amod"].intersection(self.first_tokens | self.last_tokens):
                    first_or_last = ' '.join(node_desc["amod"].intersection(self.first_tokens | self.last_tokens))
                    return first_or_last
            nodes = [elem for node in nodes for elem in node.children]
        return first_or_last

    def find_ranking_tokens(self, node: Node, appos_token_nums: List[int], clause_token_nums: List[int]) -> list:
        ranking_tokens = []
        for elem in node.descendants:
            if self.nlp(elem.form)[0].lemma_ in self.ranking_tokens \
                    and elem.ord not in appos_token_nums + clause_token_nums:
                ranking_tokens.append(elem.ord)
                ranking_tokens.append(elem.parent.ord)
                return ranking_tokens
        return ranking_tokens

    @staticmethod
    def choose_grounded_entity(grounded_entities: List[str], entities_dict: Dict[str, str]):
        tags = [entities_dict.get(entity.lower(), "") for entity in grounded_entities]
        if len(grounded_entities) > 1:
            if not all([tags[i] == tags[0] for i in range(1, len(tags))]):
                for f_tag in ["WORK_OF_ART", "FAC", "PERSON", "GPE"]:
                    for entity, tag in zip(grounded_entities, tags):
                        if tag == f_tag:
                            return [entity]
            elif not all([entity[0].islower() for entity in grounded_entities]):
                for entity in grounded_entities:
                    if entity[0].isupper():
                        return [entity]
        return grounded_entities

    def build_query(self, root: Node, unknown_branch: Node, root_desc: Dict[str, List[Node]], unknown_node: Node,
                    unknown_modifiers: List[Node], clause_modifiers: List[Node], clause_node: Node,
                    positions: List[List[int]], entities_dict: Dict[str, str], count: bool = False,
                    temporal_order: str = "", ranking_tokens: List[str] = None) -> Tuple[
        List[str], List[str], List[str]]:
        query_nums = []
        grounded_entities_list, types_list, modifiers_list, qualifier_entities_list = [], [], [], []
        found_year_or_number = False
        order = False
        root_desc_deprels = []
        for key in root_desc.keys():
            for i in range(len(root_desc[key])):
                if key in {"nsubj", "obj", "obl", "iobj", "acl", "nmod", "xcomp", "cop"}:
                    root_desc_deprels.append(key)
        root_desc_deprels = sorted(root_desc_deprels)
        log.debug(f"build_query: root_desc.keys, {root_desc_deprels}, positions {positions}, wh_leaf {self.wh_leaf}, "
                  f"one_chain {self.one_chain}, temporal order {temporal_order}, ranking tokens {ranking_tokens}")
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
                    grounded_entities_list = self.find_entities(root_desc["nsubj"][0], positions)
                    if not grounded_entities_list:
                        grounded_entities_list = self.find_entities(root_desc["obl"][0], positions)
                else:
                    for nodes in root_desc.values():
                        if nodes[0].form not in self.q_pronouns:
                            grounded_entities_list = self.find_entities(nodes[0], positions)
                            if grounded_entities_list:
                                break
            else:
                if self.root_entity:
                    grounded_entities_list = [root.form]
                for nodes in root_desc.values():
                    if nodes[0] != unknown_branch:
                        grounded_entities_list = self.find_entities(nodes[0], positions)
                        if grounded_entities_list:
                            type_entity = unknown_node.form
                            types_list.append(type_entity)
                            break

                if unknown_modifiers:
                    for n, modifier in enumerate(unknown_modifiers):
                        if isinstance(modifier, str):
                            modifiers_list.append(modifier)
                        else:
                            modifier_entities = self.find_entities(modifier, positions)
                            if modifier_entities:
                                modifiers_list += modifier_entities
                if clause_modifiers:
                    found_year_or_number = self.find_year_or_number(clause_modifiers[0])
                    if found_year_or_number:
                        query_nums.append("0")
                    qualifier_entities_list = self.find_entities(clause_modifiers[0], positions)

        if root_desc_deprels == ["nsubj", "obl", "obl"]:
            grounded_entities_list = self.find_entities(root_desc["nsubj"][0], positions)
            for node in root_desc["obl"]:
                if node == unknown_branch:
                    types_list.append(node.form)
                else:
                    grounded_entities_list += self.find_entities(node, positions)

        if root_desc_deprels == ["nsubj", "obj", "obj"]:
            obj_desc = root_desc["obj"]
            qualifier_entities_list = self.find_entities(obj_desc[0], positions)
            grounded_entities_list = self.find_entities(obj_desc[1], positions)

        year_constraint = self.find_year_constraint(root)
        if root_desc_deprels == ["nmod", "nsubj"] and year_constraint:
            if len(year_constraint[0]) == 2:
                query_nums.append("24")
            elif len(year_constraint[0]) == 1:
                query_nums.append("0")

        if root_desc_deprels == ["obj", "xcomp"]:
            grounded_entities_list = self.find_entities(root_desc["xcomp"][0], positions)

        if (self.wh_leaf and root_desc_deprels in [["nsubj", "obj", "obl"], ["obj", "obl"]]) \
                or (root_desc_deprels in [["nsubj", "obj", "obl"], ["obl", "xcomp"]]
                    and self.find_year_or_number(root_desc["obl"][0])):
            found_year_or_number = self.find_year_or_number(root_desc["obl"][0])
            nsubj_ent_list, obj_ent_list = [], []
            if "nsubj" in root_desc_deprels:
                nsubj_ent_list = self.find_entities(root_desc["nsubj"][0], positions)
            if "obj" in root_desc:
                obj_ent_list = self.find_entities(root_desc["obj"][0], positions)
            obl_ent_list = self.find_entities(root_desc["obl"][0], positions)
            log.debug(f"nsubj_ent: {nsubj_ent_list} --- obj_ent: {obj_ent_list} obl_ent: {obl_ent_list}")
            if self.wh_leaf:
                grounded_entities_list = obl_ent_list
                qualifier_entities_list = obj_ent_list
            elif not found_year_or_number and nsubj_ent_list and obl_ent_list:
                grounded_entities_list = nsubj_ent_list
                modifiers_list = obl_ent_list
            else:
                grounded_entities_list = obj_ent_list
            if found_year_or_number:
                query_nums.append("0")
            if not grounded_entities_list:
                grounded_entities_list = self.find_entities(root, positions)
                grounded_entities_list = self.choose_grounded_entity(grounded_entities_list, entities_dict)

        if clause_node:
            for node in clause_node.children:
                if node.deprel == "obj":
                    grounded_entities_list = self.find_entities(node, positions)
                if self.find_year_or_number(node):
                    query_nums.append("0")

            if not self.wh_leaf:
                type_entity = unknown_node.form
                types_list.append(type_entity)

        if root_desc_deprels == ["nmod", "nmod"]:
            grounded_entities_list = self.find_entities(root_desc["nmod"][0], positions)
            modifiers_list = self.find_entities(root_desc["nmod"][1], positions)

        if root_desc_deprels == ["nmod", "nsubj", "nummod"]:
            if not self.wh_leaf:
                grounded_entities_list = self.find_entities(root_desc["nmod"][0], positions)
                found_year_or_number = self.find_year_or_number(root_desc["nummod"][0])

        if temporal_order and not query_nums:
            for deprel in root_desc:
                for node in root_desc[deprel]:
                    entities = self.find_entities(node, positions)
                    if entities:
                        grounded_entities_list = entities
                        break
                if grounded_entities_list:
                    break
            if temporal_order in self.first_tokens | self.begin_tokens:
                query_nums += ["22"]
            if temporal_order in self.last_tokens | self.end_tokens:
                query_nums += ["23"]
        log.debug(f"query_nums: {query_nums} --- year_constraint: {year_constraint}")

        if count:
            grounded_entities_list = self.find_entities(root, positions)

        grounded_entities_list = self.choose_grounded_entity(grounded_entities_list, entities_dict)
        entities_list = grounded_entities_list + qualifier_entities_list + modifiers_list
        types_list = [tp for tp in types_list
                      if not (len(tp.split()) == 1 and self.nlp(tp)[0].lemma_ in self.date_tokens)]

        gr_len = len(grounded_entities_list)
        types_len = len(types_list)
        mod_len = len(modifiers_list)
        qua_len = len(qualifier_entities_list)
        if qua_len or count:
            types_len = 0

        if not temporal_order and not query_nums:
            for num, template in self.template_queries.items():
                syntax_info = [gr_len, types_len, mod_len, qua_len, found_year_or_number, count, order]
                if syntax_info == list(template["syntax_structure"].values()):
                    query_nums.append(num)
                if mod_len:
                    syntax_info[1] = 0
                    if syntax_info == list(template["syntax_structure"].values()):
                        query_nums.append(num)

        log.debug(f"tree_to_sparql, grounded entities: {grounded_entities_list} --- types: {types_list} --- "
                  f"modifier entities: {modifiers_list} --- qualifier entities: {qualifier_entities_list} --- "
                  f"year_or_number {found_year_or_number} --- count: {count} --- order: {order} --- "
                  f"query nums: {query_nums}")

        return query_nums, entities_list, types_list
