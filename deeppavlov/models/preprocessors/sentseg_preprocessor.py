from nltk import word_tokenize
from overrides import overrides
from typing import List

from deeppavlov.core.common.registry import register


@register("sentseg_restore_sent")
def SentSegRestoreSent(batch_x: List[List[str]], batch_y: List[List[str]]) -> List[str]:
    ret = []
    for x, y in zip(batch_x, batch_y):
        assert len(x) == len(y)
        if len(y) == 0:
            ret.append("")
            continue
        sent = x[0]
        punct = "" if y[0] == "O" else y[0][-1]
        for word, tag in zip(x[1:], y[1:]):
            if tag != "O":
                sent += punct
                punct = tag[-1]
            sent += " " + word
        sent += punct
        ret.append(sent)

    return ret
