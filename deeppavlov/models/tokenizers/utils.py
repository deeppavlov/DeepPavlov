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
from typing import List, Generator, Any


def detokenize(tokens):
    """
    Detokenizing a text undoes the tokenizing operation, restores
    punctuation and spaces to the places that people expect them to be.
    Ideally, `detokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(tokens)
    step0 = text.replace('. . .', '...')
    step1 = step0.replace("`` ", '"').replace(" ''", '"')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't") \
        .replace(" nt", "nt").replace("can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


def ngramize(items: List[str], ngram_range=(1, 1), doc: str = None) -> Generator[List[str], Any, None]:
    """
    Make ngrams from a list of tokens/lemmas
    :param items: list of tokens, lemmas or other strings to form ngrams
    :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
    (1, 2), for bigrams only should be set to (2, 2)
    :return: ngrams (as strings) generator
    """

    ngrams = []
    ranges = [(0, i) for i in range(ngram_range[0], ngram_range[1] + 1)]
    for r in ranges:
        ngrams += list(zip(*[items[j:] for j in range(*r)]))

    formatted_ngrams = [' '.join(item) for item in ngrams]
    if doc is not None:
        doc_lower = doc.lower()
        formatted_ngrams = [ngram for ngram in formatted_ngrams if (ngram in doc or ngram in doc_lower)]

    yield formatted_ngrams
