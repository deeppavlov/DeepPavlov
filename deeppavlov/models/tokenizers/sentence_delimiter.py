import pysbd

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register('sentence_delimiter')
def SentenceDelimiter(x_long):
    seg = pysbd.Segmenter(clean=False)
    xs = [a for a in seg.segment(x_long[0]) if len(a)>0]
    return tuple(xs)
    
@register('sentence_concatenator')
def SentenceConcatenator(x_long):
    x_short = []
    for sent in x_long:
        x_short.extend(sent)
    return x_short