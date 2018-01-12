import requests
import sys

from deeppavlov.core.common.registry import register

from deeppavlov.models.spellers.error_model.error_model import StaticDictionary


@register('russian_words_vocab')
class RussianWordsVocab(StaticDictionary):
    dict_name = 'russian_words_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        print('Downloading russian vocab from https://github.com/danakt/russian-words/', file=sys.stderr)
        url = 'https://github.com/danakt/russian-words/raw/master/russian.txt'
        page = requests.get(url)
        return [word.strip() for word in page.content.decode('cp1251').split('\n')]
