import requests
from deeppavlov.common.registry import register_model

from deeppavlov.models.speller.models import StaticDictionary


@register_model('russian_words_vocab')
class RussianWordsVocab(StaticDictionary):
    dict_name = 'russian_words_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        print('Downloading russian vocab from https://github.com/danakt/russian-words/')
        url = 'https://github.com/danakt/russian-words/raw/master/russian.txt'
        page = requests.get(url)
        return [word.strip() for word in page.content.decode('cp1251').split('\n')]
