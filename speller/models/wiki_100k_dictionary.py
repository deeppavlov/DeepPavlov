from deeppavlov.common.registry import register_model
from speller.models.static_dictionary import StaticDictionary
from lxml import html
import requests


@register_model('wikipedia_100K_vocab')
class Wiki100KDictionary(StaticDictionary):
    dict_name = 'wikipedia_100K_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        words = []
        print('Downloading english vocab from Wikipedia')
        for i in range(1, 100000, 10000):
            k = 10000 + i - 1
            url = 'https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/{}-{}'.format(i, k)
            page = requests.get(url)
            tree = html.fromstring(page.content)
            words += tree.xpath('//div[@class="mw-parser-output"]/p/a/text()')
        return words
