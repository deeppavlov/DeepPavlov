import requests
import sys
from lxml import html

from deeppavlov.core.common.registry import register
from deeppavlov.vocabs.static_dictionary import StaticDictionary


@register('wikipedia_100K_vocab')
class Wiki100KDictionary(StaticDictionary):
    dict_name = 'wikipedia_100K_vocab'

    @staticmethod
    def _get_source(*args, **kwargs):
        words = []
        print('Downloading english vocab from Wikipedia', file=sys.stderr)
        for i in range(1, 100000, 10000):
            k = 10000 + i - 1
            url = 'https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/{}-{}'.format(i, k)
            page = requests.get(url)
            tree = html.fromstring(page.content)
            words += tree.xpath('//div[@class="mw-parser-output"]/p/a/text()')
        return words
