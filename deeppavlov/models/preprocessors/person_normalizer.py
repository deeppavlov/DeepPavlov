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

import itertools
import random
from typing import List, Dict, Tuple, Union, Callable
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = getLogger()


@register('person_normalizer')
class PersonNormalizer(Component):
    """
    Detects mentions of mate user's name and either
    (0) converts them to user's name taken from state
    (1) either removes them.

    Parameters:
        person_tag: tag name that corresponds to a person entity
    """
    def __init__(self,
                 person_tag: str = 'PER',
                 state_slot: str = 'user_name',
                 **kwargs):
        self.per_tag = person_tag
        self.state_slot = state_slot

    def __call__(self,
                 tokens: List[List[str]],
                 tags: List[List[str]],
                 states: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
        out_tokens, out_tags = [], []
        states = states if states else [{}] * len(tokens)
        for u_state, u_toks, u_tags in zip(states, tokens, tags):
            u_toks, u_tags = self.tag_mate_gooser_name(u_toks,
                                                       u_tags,
                                                       person_tag=self.per_tag)
            if 'B-MATE-GOOSER' in u_tags:
                print("Found MATE-GOOSER name")
            else:
                print("Didn't find any MATE-GOOSER name")
            if u_state and u_state.get(self.state_slot):
                print("Replacing all users name mentions.")
                u_toks, u_tags = self.replace_mate_gooser_name(u_toks,
                                                               u_tags,
                                                               u_state[self.state_slot])
                if random.random() > .8:
                    print("Adding calling user by name")
                    u_toks = [u_state[self.state_slot], ','] + u_toks
                    u_tags = ['B-MATE-GOOSER', 'O'] + u_tags

                    u_toks[0] = u_toks[0][0].upper() + u_toks[0][1:]
                    if u_tags[2] == 'O':
                        u_toks[2] = u_toks[2][0].lower() + u_toks[2][1:]
            else:
                print("Removing all users name mentions.")
                u_toks, u_tags = self.remove_mate_gooser_name(u_toks, u_tags)
            out_tokens.append(u_toks)
            out_tags.append(u_tags)
        print(f"out_tags = {out_tags}")
        return out_tokens, out_tags

    @staticmethod
    def tag_mate_gooser_name(tokens: List[str],
                             tags: List[str],
                             person_tag: str = 'PER',
                             mate_tag: str = 'MATE-GOOSER') -> \
            Tuple[List[str], List[str]]:
        if 'B-PER' not in tags:
            return tokens, tags
        out_tags = []
        i = 0
        while (i < len(tokens)):
            tok, tag = tokens[i], tags[i]
            if i + 1 < len(tokens):
                if (tok == ',') and (tags[i + 1] == 'B-' + person_tag):
                    # it might be mate gooser name
                    out_tags.append(tag)
                    j = 1
                    while (i + j < len(tokens)) and (tags[i + j][2:] == person_tag):
                        j += 1
                    if (i + j == len(tokens)) or (tokens[i + j] in ',.!?;)'):
                        # it is mate gooser
                        out_tags.extend([t[:2] + mate_tag for t in tags[i+1:i+j]])
                    else:
                        # it isn't
                        out_tags.extend(tags[i+1:i+j])
                    i += j
                    continue
            if i > 0:
                if (tok == ',') and (tags[i - 1][2:] == 'PER'):
                    # it might have been mate gooser name
                    j = 1
                    while (len(out_tags) >= j) and (out_tags[-j][2:] == person_tag):
                        j += 1
                    if (len(out_tags) < j) or (tokens[i-j] in ',.!?'):
                        # it was mate gooser
                        for k in range(j - 1):
                            out_tags[-k-1] = out_tags[-k-1][:2] + mate_tag
                    out_tags.append(tag)
                    i += 1
                    continue
            out_tags.append(tag)
            i += 1
        return tokens, out_tags

    @staticmethod
    def replace_mate_gooser_name(tokens: List[str],
                                 tags: List[str],
                                 replacement: str,
                                 mate_tag: str = 'MATE-GOOSER') ->\
            Tuple[List[str], List[str]]:
        assert len(tokens) == len(tags),\
            f"tokens({tokens}) and tags({tags}) should have the same length"
        if 'B-' + mate_tag not in tags:
            return tokens, tags

        repl_tokens = replacement.split()
        repl_tags = ['B-' + mate_tag] + ['I-' + mate_tag] * (len(repl_tokens) - 1)

        out_tokens, out_tags = [], []
        i = 0
        while (i < len(tokens)):
            tok, tag = tokens[i], tags[i]
            if tag == 'B-' + mate_tag:
                out_tokens.extend(repl_tokens)
                out_tags.extend(repl_tags)
                i += 1
                while (i < len(tokens)) and (tokens[i] == 'I-' + mate_tag):
                    i += 1
            else:
                out_tokens.append(tok)
                out_tags.append(tag)
                i += 1
        return out_tokens, out_tags

    @staticmethod
    def remove_mate_gooser_name(tokens: List[str],
                                tags: List[str],
                                mate_tag: str = 'MATE-GOOSER') ->\
            Tuple[List[str], List[str]]:
        assert len(tokens) == len(tags),\
            f"tokens({tokens}) and tags({tags}) should have the same length"
        # TODO: uppercase first letter if name was removed
        if 'B-' + mate_tag not in tags:
            return tokens, tags
        out_tokens, out_tags = [], []
        i = 0
        while (i < len(tokens)):
            tok, tag = tokens[i], tags[i]
            if i + 1 < len(tokens):
                if (tok == ',') and (tags[i + 1] == 'B-' + mate_tag):
                    # it will be mate gooser name next, skip comma
                    i += 1
                    continue
            if i > 0:
                if (tok == ',') and (tags[i - 1][2:] == mate_tag):
                    # that was mate gooser name, skip comma
                    i += 1
                    continue
            if tag[2:] != mate_tag:
                out_tokens.append(tok)
                out_tags.append(tag)
            i += 1
        return out_tokens, out_tags


LIST_LIST_STR_BATCH = List[List[List[str]]]


@register('history_person_normalizer')
class HistoryPersonNormalize(Component):
    """
    Takes batch of dialog histories and normalizes only bot responses.

    Detects mentions of mate user's name and either
    (0) converts them to user's name taken from state
    (1) either removes them.

    Parameters:
        per_tag: tag name that corresponds to a person entity
    """
    def __init__(self, per_tag: str = 'PER', **kwargs):
        self.per_normalizer = PersonNormalizer(per_tag=per_tag)

    def __call__(self,
                 history_tokens: LIST_LIST_STR_BATCH,
                 tags: LIST_LIST_STR_BATCH,
                 states: List[Dict]) -> Tuple[LIST_LIST_STR_BATCH, LIST_LIST_STR_BATCH]:
        out_tokens, out_tags = [], []
        states = states if states else [{}] * len(tags)
        for u_state, u_hist_tokens, u_hist_tags in zip(states, history_tokens, tags):
            # TODO: normalize bot response history
            pass
        return out_tokens, out_tags


@register('myself_detector')
class MyselfDetector(Component):
    """
    Finds first mention of a name and sets it as a user name.

    Parameters:
        person_tag: tag name that corresponds to a person entity
        state_slot: name of a state slot corresponding to a user's name

    """
    def __init__(self,
                 person_tag: str = 'PER',
                 state_slot: str = 'user_name',
                 **kwargs):
        self.per_tag = person_tag
        self.state_slot = state_slot

    def __call__(self,
                 tokens: List[List[str]],
                 tags: List[List[str]],
                 states: List[Dict]) -> List[Dict]:
        out_states = []
        states = states if states else [{}] * len(tokens)
        for u_state, u_toks, u_tags in zip(states, tokens, tags):
            if not u_state or not u_state.get(self.state_slot):
                name_found = self.find_my_name(u_toks, u_tags, person_tag=self.per_tag)
                if name_found is not None:
                    if not u_state:
                        u_state = {}
                    u_state[self.state_slot] = name_found
            out_states.append(u_state)
        return out_states

    @staticmethod
    def find_my_name(tokens: List[str], tags: List[str], person_tag: str) -> str:
        if 'B-' + person_tag not in tags:
            return None
        per_start = tags.index('B-' + person_tag)
        per_excl_end = per_start + 1
        while (per_excl_end < len(tokens)) and (tags[per_excl_end] == 'I-' + person_tag):
            per_excl_end += 1
        return ' '.join(tokens[per_start:per_excl_end])


@register('ner_with_context')
class NerWithContextWrapper(Component):
    """
    Tokenizers utterance and history of dialogue and gets entity tags for
    utterance's tokens.

    Parameters:
        ner_model: named entity recognition model
        tokenizer: tokenizer to use

    """
    def __init__(self,
                 ner_model: Union[Component, Callable],
                 tokenizer: Union[Component, Callable],
                 context_delimeter: str = None,
                 **kwargs):
        self.ner_model = ner_model
        self.tokenizer = tokenizer
        self.context_delimeter = context_delimeter

    def __call__(self,
                 utterances: List[str],
                 history: List[List[str]] = [[]],
                 prev_utterances: List[str] = []) ->\
            Tuple[List[List[str]], List[List[str]]]:
        if prev_utterances:
            history = history or itertools.repeat([])
            history = [hist + [prev]
                       for prev, hist in zip(prev_utterances, history)]
        history_toks = [[tok
                         for toks in self.tokenizer(hist or [''])
                         for tok in toks + [self.context_delimeter] if tok is not None]
                        for hist in history]
        utt_toks = self.tokenizer(utterances)
        texts, ranges = [], []
        for utt, hist in zip(utt_toks, history_toks):
            if self.context_delimeter is not None:
                txt = hist + utt + [self.context_delimeter]
            else:
                txt = hist + utt
            ranges.append((len(hist), len(hist) + len(utt)))
            texts.append(txt)

        _, tags = self.ner_model(texts)
        print(f"texts = {texts}, ranges = {ranges}, tags = {tags}")
        tags = [t[l:r] for t, (l, r) in zip(tags, ranges)]

        return utt_toks, tags


@register('name_asker')
class NameAskerPostprocessor(Component):

    def __init__(self,
                 state_slot: str = 'user_name',
                 flag_slot: str = 'asked_name',
                 **kwargs) -> None:
        self.state_slot = state_slot
        self.flag_slot = flag_slot

    def __call__(self,
                 utters: List[str],
                 histories: List[List[str]],
                 states: List[dict],
                 responses: List[str],
                 **kwargs) -> List[str]:
        new_responses, new_states = [], []
        print(f"states in NameAsker = {states}")
        states = states if states else [{}] * len(utters)
        for utter, hist, state, resp in zip(utters, histories, states, responses):
            state = state or {}
            if (self.state_slot not in state) and\
                    (self.flag_slot not in state):
                if (len(hist) == 0) and (random.random() < 0.2):
                    new_responses.append('Привет! Тебя как зовут?')
                    state[self.flag_slot] = True
                elif (len(hist) < 2) and (random.random() < 0.5):
                    new_responses.append('Как тебя зовут?')
                    state[self.flag_slot] = True
                elif (len(hist) >= 2) and (random.random() < 0.1):
                    new_responses.append('Тебя как зовут-то?')
                    state[self.flag_slot] = True
                else:
                    new_responses.append(resp)
            else:
                new_responses.append(resp)
            new_states.append(state)
        return new_responses, new_states

