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
		return [w.lemma_ for w in doc]
