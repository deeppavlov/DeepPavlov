# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
from typing import List, Any, Dict, Iterable, Optional, Tuple

import spacy
from spacy.matcher import Matcher

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.tokenizers.spacy_tokenizer import _try_load_spacy_model


@register('ecommerce_preprocess')
class EcommercePreprocess(Component):
    """Class to process strings for EcommerceBot skill

    Parameters:
        spacy_model: SpaCy model name
        disable: SpaCy pipeline to disable
    """

    def __init__(self, spacy_model: str = 'en_core_web_sm', disable: Optional[Iterable[str]] = None, **kwargs):
        if disable is None:
            disable = ['parser', 'ner']

        self.model = _try_load_spacy_model(spacy_model, disable=disable)

        below = lambda text: bool(re.compile(r'below|cheap').match(text))
        BELOW = self.model.vocab.add_flag(below)

        above = lambda text: bool(re.compile(r'above|start').match(text))
        ABOVE = self.model.vocab.add_flag(above)

        self.matcher = Matcher(self.model.vocab)

        self.matcher.add('below', None, [{BELOW: True}, {'LOWER': 'than', 'OP': '?'},
                                         {'LOWER': 'from', 'OP': '?'}, {'ORTH': '$', 'OP': '?'},
                                         {'ENT_TYPE': 'MONEY', 'LIKE_NUM': True}])

        self.matcher.add('above', None, [{ABOVE: True}, {'LOWER': 'than', 'OP': '?'},
                                         {'LOWER': 'from', 'OP': '?'}, {'ORTH': '$', 'OP': '?'},
                                         {'ENT_TYPE': 'MONEY', 'LIKE_NUM': True}])

    def __call__(self, **kwargs):
        pass

    def extract_money(self, doc: spacy.tokens.Doc) -> Tuple[List, Tuple[float, float]]:
        """Extract money entities and money related tokens from `doc`.

        Parameters:
            doc: a list of tokens with corresponding tags, lemmas, etc.

        Returns:
            doc_no_money: doc with no money related tokens.
            money_range: money range from `money_range[0]` to `money_range[1]` extracted from the doc.
        """

        matches = self.matcher(doc)
        money_range: Tuple = ()
        doc_no_money = list(doc)
        negated = False

        for match_id, start, end in matches:
            string_id = self.model.vocab.strings[match_id]
            span = doc[start:end]
            for child in doc[start].children:
                if child.dep_ == 'neg':
                    negated = True

            num_token = [token for token in span if token.like_num == True]
            if (string_id == 'below' and negated == False) or (string_id == 'above' and negated == True):
                money_range = (0, float(num_token[0].text))

            if (string_id == 'above' and negated == False) or (string_id == 'below' and negated == True):
                money_range = (float(num_token[0].text), float(math.inf))

            del doc_no_money[start:end + 1]
        return doc_no_money, money_range

    def analyze(self, text: str) -> Iterable:
        """SpaCy `text` preprocessing"""
        return self.model(text)

    def spacy2dict(self, doc: spacy.tokens.Doc, fields: List[str] = None) -> List[Dict[Any, Any]]:
        """Convert SpaCy doc into list of tokens with `fields` properties only"""
        if fields is None:
            fields = ['tag_', 'like_num', 'lemma_', 'text']
        return [{field: getattr(token, field) for field in fields} for token in doc]

    def filter_nlp(self, tokens: Iterable) -> List[Any]:
        """Filter tokens according to the POS tags"""
        res = []
        for word in tokens:
            if word.tag_ not in ['MD', 'SP', 'DT', 'TO']:
                res.append(word)
        return res

    def filter_nlp_title(self, doc: Iterable) -> List[Any]:
        """Filter item titles according to the POS tags"""
        return [w for w in doc if w.tag_ in ['NNP', 'NN', 'PROPN', 'JJ'] and not w.like_num]

    def lemmas(self, doc: Iterable) -> List[str]:
        """Return lemma of `doc`"""
        return [w.get('lemma_') if isinstance(w, dict) else w.lemma_ for w in doc]

    def price(self, item: Dict[Any, Any]) -> float:
        """Return price of item in a proper format"""
        if 'ListPrice' in item:
            return float(item['ListPrice'].split('$')[1].replace(",", ""))
        return 0

    def parse_input(self, inp: str) -> Dict[Any, Any]:
        """Convert space-delimited string into dialog state"""
        state: List = []
        for i in range(len(inp.split()) // 2, 0, -1):
            state.append([inp.split(None, 1)[0], inp.split(None, 1)[1].split()[0]])

            if i > 1:
                inp = inp.split(None, 2)[2]

        return dict(state)
