import requests
from lxml import html
from lxml.html import builder as E

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('proxy')
class Proxy(Component):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _add_base_url(text, base_url):
        tree = html.fromstring(text)
        head = tree.find('head')
        if not head:
            head = E.HEAD()
            tree.append(head)
        base = head.find('base')
        if not base:
            base = E.BASE()
            head.append(base)
        base.attrib['target'] = '_blank'
        if 'href' not in base.attrib:
            base.attrib['href'] = base_url
        return html.tostring(tree)

    def __call__(self, urls):
        res = []
        errors = []
        for url in urls:
            if url.find('://') == -1:
                url = 'http://' + url
            try:
                r = requests.head(url, allow_redirects=True)
                if 'text/html' not in r.headers['content-type']:
                    raise RuntimeError(f'`{url}` is not an html page')
                r = requests.get(url)
                if r.status_code == 200:
                    res.append(self._add_base_url(r.text, r.url))
                    errors.append(False)
                    continue

                res.append(f'Got status code {r.status_code}')
            except Exception as e:
                res.append(str(e))
            errors.append(True)
        return res, errors
