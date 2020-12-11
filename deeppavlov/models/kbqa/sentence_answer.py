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

import pyinflect
import re
import spacy

nlp = spacy.load('en_core_web_sm')

pronouns = ["who", "what", "when", "where"]

def find_tokens(tokens, node, not_inc_node):
    if node != not_inc_node:
        tokens.append(node.text)
        for elem in node.children:
            tokens = find_tokens(tokens, elem, not_inc_node)
    return tokens
    
def find_inflect_dict(sent_nodes):
    inflect_dict = {}
    for node in sent_nodes:
        if node.dep_ == "aux" and node.tag_ == "VBD" and node.head.tag_ == "VBP":
            new_verb = node.head._.inflect("VBD")
            inflect_dict[node.head.text] = new_verb
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
            
    return wh_node, wh_node_head,  main_head
    
def find_tokens_to_replace(wh_node_head, main_head, question_tokens):
    redundant_tokens_to_replace = []
    question_tokens_to_replace = []
    
    if main_head:
        redundant_tokens_to_replace = find_tokens([], main_head, wh_node_head)
    
    wh_node_head_desc = [node for node in wh_node_head.children if node.text != "?"]
    wh_node_head_dep = [node.dep_ for node in wh_node_head.children if (node.text != "?" and node.dep_ not in ["aux", "prep"] and node.text.lower() not in pronouns)]
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
    
def find_who_is_tokens(answer, noun_tokens):
    fnd = re.findall("who is (.*)", answer) or re.findall("Who is (.*)", answer)
    who_is_question = False
    entity_tokens = ""
    if fnd:
        tokens = fnd[0].split()
        if all([tok in noun_tokens for tok in tokens]):
            who_is_question = True
            entity_tokens = fnd[0]
    
    return who_is_question, entity_tokens

def sentence_answer(question, entity_title):
    sent_nodes = nlp(question)
        
    question_tokens = [elem.text for elem in sent_nodes]
    noun_tokens = [elem.text for elem in sent_nodes if elem.tag_ in ["NN", "NNP"]]
    
    inflect_dict = find_inflect_dict(sent_nodes)
    wh_node, wh_node_head, main_head = find_wh_node(sent_nodes)
    redundant_replace_substr, question_replace_substr = find_tokens_to_replace(wh_node_head, main_head, question_tokens)
    
    answer = ""
    if redundant_replace_substr:
        answer = question.replace(redundant_replace_substr, '')
    else:
        answer = question
        
    if answer.endswith('?'):
        answer = answer.replace('?', '').strip()
    
    who_is_question, entity_tokens = find_who_is_tokens(answer, noun_tokens)
        
    if question_replace_substr:
        if wh_node.text.lower() in ["what", "who"]:
            if who_is_question:
                answer = f"{entity_tokens} is {entity_title}"
            else:
                answer = answer.replace(question_replace_substr, entity_title)
        if wh_node.text.lower() in ["when", "where"]:
            answer = answer.replace(question_replace_substr, '')
            answer = f"{answer} in {entity_title}"
            
    for old_tok, new_tok in inflect_dict.items():
        answer = answer.replace(old_tok, new_tok)
    
    answer = answer + '.'
    
    return answer
