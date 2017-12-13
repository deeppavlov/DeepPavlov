from deeppavlov.common.registry import register_model
from speller.models.static_dictionary import StaticDictionary
from lxml import html
import requests


@register_model('russian_words_vocab')
class RussianWordsVocab(StaticDictionary):
    dict_name = 'russian_words_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        print('Downloading russian vocab from https://github.com/danakt/russian-words/')
        url = 'https://github.com/danakt/russian-words/raw/master/russian.txt'
        page = requests.get(url)
        return page.content.decode('cp1251').split()
