from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register

@register('ecommerce_preprocess')
class EcommercePreprocess(Component):

    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass

    def filter_nlp(self, tokens):
        res = []
        for word in tokens:
            if word.tag_ not in ['MD', 'SP', 'DT', 'TO']:
                res.append(word)
        return res

    def filter_nlp_title(self, doc):
        return [w for w in doc if w.tag_ in ['NNP', 'NN', 'PROPN', 'JJ'] and not w.like_num]

    def lemmas(self, doc):
        return [w.get('lemma_') if type(w) == dict else w.lemma_ for w in doc]

    def price(self, item):
        if 'ListPrice' in item:
            return float(item['ListPrice'].split('$')[1].replace(",", ""))
        else:
            return 0

    def parse_input(self, inp):
        state = []
        for i in range(len(inp.split())//2, 0, -1):
            state.append([inp.split(None, 1)[0], inp.split(None, 1)[1].split()[0]])

            if i > 1:
                inp = inp.split(None, 2)[2]

        return dict(state)