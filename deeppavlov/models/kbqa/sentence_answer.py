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

import importlib
import re
from logging import getLogger

import pkg_resources
import spacy

log = getLogger(__name__)

# en_core_web_sm is installed and used by test_inferring_pretrained_model in the same interpreter session during tests.
# Spacy checks en_core_web_sm package presence with pkg_resources, but pkg_resources is initialized with interpreter,
# sot it doesn't see en_core_web_sm installed after interpreter initialization, so we use importlib.reload below.

if 'en-core-web-sm' not in pkg_resources.working_set.by_key.keys():
    importlib.reload(pkg_resources)

# TODO: move nlp to sentence_answer, sentence_answer to rel_ranking_infer and revise en_core_web_sm requirement,
# TODO: make proper downloading with spacy.cli.download
nlp = spacy.load('en_core_web_sm')

pronouns = ["who", "what", "when", "where", "how"]


def find_tokens(tokens, node, not_inc_node):
    if node != not_inc_node:
        tokens.append(node.text)
        for elem in node.children:
            tokens = find_tokens(tokens, elem, not_inc_node)
    return tokens


def find_inflect_dict(sent_nodes):
    inflect_dict = {}
    for node in sent_nodes:
        if node.dep_ == "aux" and node.tag_ == "VBD" and (node.head.tag_ == "VBP" or node.head.tag_ == "VB"):
            new_verb = node.head._.inflect("VBD")
            inflect_dict[node.head.text] = new_verb
            inflect_dict[node.text] = ""
        if node.dep_ == "aux" and node.tag_ == "VBZ" and node.head.tag_ == "VB":
            new_verb = node.head._.inflect("VBZ")
            inflect_dict[node.head.text] = new_verb
            inflect_dict[node.text] = ""
    return inflect_dict


def find_wh_node(sent_nodes):
    wh_node = ""
    main_head = ""
    wh_node_head = ""
    for node in sent_nodes:
        if node.text.lower() in pronouns:
            wh_node = node
            break

    if wh_node:
        wh_node_head = wh_node.head
        if wh_node_head.dep_ == "ccomp":
            main_head = wh_node_head.head

    return wh_node, wh_node_head, main_head


def find_tokens_to_replace(wh_node_head, main_head, question_tokens, question):
    redundant_tokens_to_replace = []
    question_tokens_to_replace = []

    if main_head:
        redundant_tokens_to_replace = find_tokens([], main_head, wh_node_head)
    what_tokens_fnd = re.findall("what (.*) (is|was|does|did) (.*)", question, re.IGNORECASE)
    if what_tokens_fnd:
        what_tokens = what_tokens_fnd[0][0].split()
        if len(what_tokens) <= 2:
            redundant_tokens_to_replace += what_tokens

    wh_node_head_desc = [node for node in wh_node_head.children if node.text != "?"]
    wh_node_head_dep = [node.dep_ for node in wh_node_head.children if
                        (node.text != "?" and node.dep_ not in ["aux", "prep"] and node.text.lower() not in pronouns)]
    for node in wh_node_head_desc:
        if node.dep_ == "nsubj" and len(wh_node_head_dep) > 1 or node.text.lower() in pronouns or node.dep_ == "aux":
            question_tokens_to_replace.append(node.text)
            for elem in node.subtree:
                question_tokens_to_replace.append(elem.text)

    question_tokens_to_replace = list(set(question_tokens_to_replace))

    redundant_replace_substr = []
    for token in question_tokens:
        if token in redundant_tokens_to_replace:
            redundant_replace_substr.append(token)
        else:
            if redundant_replace_substr:
                break

    redundant_replace_substr = ' '.join(redundant_replace_substr)

    question_replace_substr = []

    for token in question_tokens:
        if token in question_tokens_to_replace:
            question_replace_substr.append(token)
        else:
            if question_replace_substr:
                break

    question_replace_substr = ' '.join(question_replace_substr)

    return redundant_replace_substr, question_replace_substr


def sentence_answer(question, entity_title, entities=None, template_answer=None):
    log.debug(f"question {question} entity_title {entity_title} entities {entities} template_answer {template_answer}")
    sent_nodes = nlp(question)
    reverse = False
    if sent_nodes[-2].tag_ == "IN":
        reverse = True
    question_tokens = [elem.text for elem in sent_nodes]
    log.debug(f"spacy tags: {[(elem.text, elem.tag_, elem.dep_, elem.head.text) for elem in sent_nodes]}")

    inflect_dict = find_inflect_dict(sent_nodes)
    wh_node, wh_node_head, main_head = find_wh_node(sent_nodes)
    redundant_replace_substr, question_replace_substr = find_tokens_to_replace(wh_node_head, main_head,
                                                                               question_tokens, question)
    log.debug(f"redundant_replace_substr {redundant_replace_substr} question_replace_substr {question_replace_substr}")
    if redundant_replace_substr:
        answer = question.replace(redundant_replace_substr, '')
    else:
        answer = question

    if answer.endswith('?'):
        answer = answer.replace('?', '').strip()

    if question_replace_substr:
        if template_answer and entities:
            answer = template_answer.replace("[ent]", entities[0]).replace("[ans]", entity_title)
        elif wh_node.text.lower() in ["what", "who", "how"]:
            fnd_date = re.findall(f"what (day|year) (.*)\?", question, re.IGNORECASE)
            fnd_wh = re.findall("what (is|was) the name of (.*) (which|that) (.*)\?", question, re.IGNORECASE)
            fnd_name = re.findall("what (is|was) the name (.*)\?", question, re.IGNORECASE)
            if fnd_date:
                fnd_date_aux = re.findall(f"what (day|year) (is|was) ({entities[0]}) (.*)\?", question, re.IGNORECASE)
                if fnd_date_aux:
                    answer = f"{entities[0]} {fnd_date_aux[0][1]} {fnd_date_aux[0][3]} on {entity_title}"
                else:
                    answer = f"{fnd_date[0][1]} on {entity_title}"
            elif fnd_wh:
                answer = f"{entity_title} {fnd_wh[0][3]}"
            elif fnd_name:
                aux_verb, sent_cut = fnd_name[0]
                if sent_cut.startswith("of "):
                    sent_cut = sent_cut[3:]
                answer = f"{entity_title} {aux_verb} {sent_cut}"
            else:
                if reverse:
                    answer = answer.replace(question_replace_substr, '')
                    answer = f"{answer} {entity_title}"
                else:
                    answer = answer.replace(question_replace_substr, entity_title)
        elif wh_node.text.lower() in ["when", "where"] and entities:
            sent_cut = re.findall(f"(when|where) (was|is) {entities[0]} (.*)\?", question, re.IGNORECASE)
            if sent_cut:
                if sent_cut[0][0].lower() == "when":
                    answer = f"{entities[0]} {sent_cut[0][1]} {sent_cut[0][2]} on {entity_title}"
                else:
                    answer = f"{entities[0]} {sent_cut[0][1]} {sent_cut[0][2]} in {entity_title}"
            else:
                answer = answer.replace(question_replace_substr, '')
                answer = f"{answer} in {entity_title}"

    for old_tok, new_tok in inflect_dict.items():
        answer = answer.replace(old_tok, new_tok)
    answer = re.sub("\s+", " ", answer).strip()

    answer = answer + '.'

    return answer
