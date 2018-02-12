"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import shutil
from collections import defaultdict

import requests
from lxml import html

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import is_done, mark_done
from deeppavlov.core.common.file import load_pickle, save_pickle
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('static_dictionary')
class StaticDictionary:
    dict_name = None

    def __init__(self, data_dir=None, *args, **kwargs):
        data_dir = expand_path(data_dir or '')
        if self.dict_name is None:
            self.dict_name = args[0] if args else kwargs.get('dictionary_name', 'dictionary')

        data_dir = data_dir / self.dict_name

        alphabet_path = data_dir / 'alphabet.pkl'
        words_path = data_dir / 'words.pkl'
        words_trie_path = data_dir / 'words_trie.pkl'

        if not is_done(data_dir):
            log.info('Trying to build a dictionary in {}'.format(data_dir))
            if data_dir.is_dir():
                shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True)

            words = self._get_source(data_dir, *args, **kwargs)
            words = {self._normalize(word) for word in words}

            alphabet = {c for w in words for c in w}
            alphabet.remove('⟬')
            alphabet.remove('⟭')

            save_pickle(alphabet, alphabet_path)
            save_pickle(words, words_path)

            words_trie = defaultdict(set)
            for word in words:
                for i in range(len(word)):
                    words_trie[word[:i]].add(word[:i+1])
                words_trie[word] = set()
            words_trie = {k: sorted(v) for k, v in words_trie.items()}

            save_pickle(words_trie, words_trie_path)

            mark_done(data_dir)
            log.info('built')
        else:
            log.info('Loading a dictionary from {}'.format(data_dir))

        self.alphabet = load_pickle(alphabet_path)
        self.words_set = load_pickle(words_path)
        self.words_trie = load_pickle(words_trie_path)

    @staticmethod
    def _get_source(*args, **kwargs):
        raw_path = args[2] if len(args) > 2 else kwargs.get('raw_dictionary_path', None)
        if not raw_path:
            raise RuntimeError('raw_path for StaticDictionary is not set')
        raw_path = expand_path(raw_path)
        with open(raw_path, newline='') as f:
            data = [line.strip().split('\t')[0] for line in f]
        return data

    @staticmethod
    def _normalize(word):
        return '⟬{}⟭'.format(word.strip().lower().replace('ё', 'е'))


@register('russian_words_vocab')
class RussianWordsVocab(StaticDictionary):
    dict_name = 'russian_words_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        log.info('Downloading russian vocab from https://github.com/danakt/russian-words/')
        url = 'https://github.com/danakt/russian-words/raw/master/russian.txt'
        page = requests.get(url)
        return [word.strip() for word in page.content.decode('cp1251').split('\n')]


@register('wikitionary_100K_vocab')
class Wiki100KDictionary(StaticDictionary):
    dict_name = 'wikipedia_100K_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        words = []
        log.info('Downloading english vocab from Wiktionary')
        for i in range(1, 100000, 10000):
            k = 10000 + i - 1
            url = 'https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/{}-{}'.format(i, k)
            page = requests.get(url)
            tree = html.fromstring(page.content)
            words += tree.xpath('//div[@class="mw-parser-output"]/p/a/text()')
        return words
