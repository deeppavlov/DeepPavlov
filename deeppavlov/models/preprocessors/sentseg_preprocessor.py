from typing import List

from deeppavlov.core.common.registry import register


@register("sentseg_restore_sent")
def SentSegRestoreSent(batch_words: List[List[str]], batch_tags: List[List[str]]) -> List[str]:
    ret = []
    for words, tags in zip(batch_words, batch_tags):
        if len(tags) == 0:
            ret.append("")
            continue
        sent = words[0]
        punct = "" if tags[0] == "O" else tags[0][-1]
        for word, tag in zip(words[1:], tags[1:]):
            if tag != "O":
                sent += punct
                punct = tag[-1]
            sent += " " + word
        sent += punct
        ret.append(sent)

    return ret
