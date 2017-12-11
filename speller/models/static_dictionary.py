import os
import shutil

import pickle
from collections import defaultdict

from deeppavlov.common.registry import register_model
from deeppavlov.data.utils import is_done, mark_done


@register_model('static_dictionary')
class StaticDictionary:
    @staticmethod
    def _get_source(*args, **kwargs):
        raw_path = args[2] if len(args) > 2 else kwargs.get('raw_dictionary_path', None)
        if not raw_path:
            raise RuntimeError('raw_path for StaticDictionary is not set')
        with open(raw_path, newline='') as f:
            data = f.readlines()
        return data

    @staticmethod
    def _normalize(word):
        return '⟬{}⟭'.format(word.strip().lower().replace('ё', 'е'))

    def __init__(self, data_dir, *args, **kwargs):
        dict_name = args[0] if args else kwargs.get('name', 'dictionary')
        data_dir = os.path.join(data_dir, dict_name)
        if not is_done(data_dir):
            print('Trying to build a dictionary in {}'.format(data_dir))
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
            os.makedirs(data_dir, 0o755)

            words = self._get_source(data_dir, *args, **kwargs)
            words = {self._normalize(word) for word in words}

            alphabet = {c for w in words for c in w}
            alphabet.remove('⟬')
            alphabet.remove('⟭')

            with open(os.path.join(data_dir, 'alphabet.pkl'), 'wb') as f:
                pickle.dump(alphabet, f)

            with open(os.path.join(data_dir, 'words.pkl'), 'wb') as f:
                pickle.dump(words, f)

            words_trie = defaultdict(set)
            for word in words:
                for i in range(len(word)):
                    words_trie[word[:i]].add(word[:i+1])
                words_trie[word] = set()
            words_trie = {k: sorted(v) for k, v in words_trie.items()}

            with open(os.path.join(data_dir, 'words_trie.pkl'), 'wb') as f:
                pickle.dump(words_trie, f)

            mark_done(data_dir)
            print('built')

        with open(os.path.join(data_dir, 'alphabet.pkl'), 'rb') as f:
            self.alphabet = pickle.load(f)
        with open(os.path.join(data_dir, 'words.pkl'), 'rb') as f:
            self.words_set = pickle.load(f)
        with open(os.path.join(data_dir, 'words_trie.pkl'), 'rb') as f:
            self.words_trie = pickle.load(f)
