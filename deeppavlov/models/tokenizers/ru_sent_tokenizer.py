import re
import logging
from typing import Set, Tuple, List

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


_SENT_RE = re.compile(r'[^\.?!…]+[\.?!…]*["»“ ]*')

_LAST_WORD = re.compile(r'(?:\b|\d)([a-zа-я]+)\.$', re.IGNORECASE)
_FIRST_WORD = re.compile(r'^\W*(\w+)')
_ENDS_WITH_ONE_LETTER_LAT_AND_DOT = re.compile(r'(\d|\W|\b)([a-zA-Z])\.$')
_HAS_DOT_INSIDE = re.compile(r'[\w]+\.[\w]+\.$', re.IGNORECASE)
_INITIALS = re.compile(r'(\W|\b)([A-ZА-Я]{1})\.$')
_ONLY_RUS_CONSONANTS = re.compile(r'^[бвгджзйклмнпрстфхцчшщ]{1,4}$', re.IGNORECASE)
_STARTS_WITH_EMPTYNESS = re.compile(r'^\s+')
_ENDS_WITH_EMOTION = re.compile(r'[!?…]|\.{2,}\s?[)"«»,“]?$')
_STARTS_WITH_LOWER = re.compile(r'^\s*[–-—-("«]?\s*[a-zа-я]')
_STARTS_WITH_DIGIT = re.compile(r'^\s*\d')
_NUMERATION = re.compile(r'^\W*[IVXMCL\d]+\.$')
_PAIRED_SHORTENING_IN_THE_END = re.compile(r'\b(\w+)\. (\w+)\.\W*$')

_JOIN = 0
_MAYBE = 1
_SPLIT = 2

JOINING_SHORTENINGS = {'mr', 'mrs', 'ms', 'dr', 'vs', 'англ', 'итал', 'греч', 'евр', 'араб', 'яп', 'слав', 'кит',
                       'тел', 'св', 'ул', 'устар', 'им', 'г', 'см', 'д', 'стр', 'корп', 'пл', 'пер', 'сокр', 'рис'}
SHORTENINGS = {'co', 'corp', 'inc', 'авт', 'адм', 'барр', 'внутр', 'га', 'дифф', 'дол', 'долл', 'зав', 'зам', 'искл',
               'коп', 'корп', 'куб', 'лат', 'мин', 'о', 'обл', 'обр', 'прим', 'проц', 'р', 'ред', 'руб', 'рус', 'русск',
               'сан', 'сек', 'тыс', 'эт', 'яз', 'гос', 'мн', 'жен', 'муж', 'накл', 'повел', 'букв', 'шутл', 'ед'}

PAIRED_SHORTENINGS = {('и', 'о'), ('т', 'е'), ('т', 'п'), ('у', 'е'), ('н', 'э')}


def _regex_split_separators(text: str) -> [str]:
    return [x.strip() for x in _SENT_RE.findall(text)]


def _is_sentence_end(left: str, right: str,
                     shortenings: Set[str],
                     joining_shortenings: Set[str],
                     paired_shortenings: Set[Tuple[str, str]]) -> int:
    if not _STARTS_WITH_EMPTYNESS.match(right):
        return _JOIN

    if _HAS_DOT_INSIDE.search(left):
        return _JOIN

    left_last_word = _LAST_WORD.search(left)
    lw = ' '
    if left_last_word:
        lw = left_last_word.group(1)

        if lw.lower() in joining_shortenings:
            return _JOIN

        if _ONLY_RUS_CONSONANTS.search(lw) and lw[-1].islower():
            return _MAYBE

    pse = _PAIRED_SHORTENING_IN_THE_END.search(left)
    if pse:
        s1, s2 = pse.groups()
        if (s1, s2) in paired_shortenings:
            return _MAYBE

    right_first_word = _FIRST_WORD.match(right)
    if right_first_word:
        rw = right_first_word.group(1)
        if (lw, rw) in paired_shortenings:
            return _MAYBE

    if _ENDS_WITH_EMOTION.search(left) and _STARTS_WITH_LOWER.match(right):
        return _JOIN

    initials = _INITIALS.search(left)
    if initials:
        border, _ = initials.groups()
        if (border or ' ') not in "°'":
            return _JOIN

    if lw.lower() in shortenings:
        return _MAYBE

    last_letter = _ENDS_WITH_ONE_LETTER_LAT_AND_DOT.search(left)
    if last_letter:
        border, _ = last_letter.groups()
        if (border or ' ') not in "°'":
            return _MAYBE
    if _NUMERATION.match(left):
        return _JOIN
    return _SPLIT


def ru_sent_tokenize(text: str,
                     shortenings: Set[str] = SHORTENINGS,
                     joining_shortenings: Set[str] = JOINING_SHORTENINGS,
                     paired_shortenings: Set[Tuple[str, str]] = PAIRED_SHORTENINGS) -> List[str]:
    sentences = []
    sents = _regex_split_separators(text)
    si = 0
    processed_index = 0
    sent_start = 0
    while si < len(sents):
        s = sents[si]
        span_start = text[processed_index:].index(s) + processed_index
        span_end = span_start + len(s)
        processed_index += len(s)

        si += 1

        send = _is_sentence_end(text[sent_start: span_end], text[span_end:],
                                shortenings, joining_shortenings, paired_shortenings)
        if send == _JOIN:
            continue

        if send == _MAYBE:
            if _STARTS_WITH_LOWER.match(text[span_end:]):
                continue
            if _STARTS_WITH_DIGIT.match(text[span_end:]):
                continue

        if not text[sent_start: span_end].strip():
            logging.warning("Something went wrong while tokenizing")
        sentences.append(text[sent_start: span_end].strip())
        sent_start = span_end
        processed_index = span_end

    if sent_start != len(text):
        if text[sent_start:].strip():
            sentences.append(text[sent_start:].strip())
    return sentences


@register("ru_sent_tokenizer")
class RuSentTokenizer(Component):
    """
    Rule-base sentence tokenizer for Russian language
    """
    def __init__(self, shortenings: Set[str] = SHORTENINGS,
                       joining_shortenings: Set[str] = JOINING_SHORTENINGS,
                       paired_shortenings: Set[Tuple[str, str]] = PAIRED_SHORTENINGS):
        """
        Args:
            shortenings: list of known shortenings. Use default value if working on news or fiction texts
            joining_shortenings: list of shortenings after that sentence split is not possible (i.e. "ул").
                                 Use default value if working on news or fiction texts
            paired_shortenings: list of known paired shotenings (i.e. "т. е.").
                                Use default value if working on news or fiction texts

        """
        self.shortenings = shortenings
        self.joining_shortenings = joining_shortenings
        self.paired_shortenings = paired_shortenings

    def __call__(self, batch: [str]) -> [[str]]:
        return [ru_sent_tokenize(x, self.shortenings, self.joining_shortenings, self.paired_shortenings) for x in batch]


if __name__ == '__main__':
    assert ru_sent_tokenize('купил за 5 руб. и остался доволен.') == ['купил за 5 руб. и остался доволен.']
    assert ru_sent_tokenize('Я ему сказал и т.к. он не послушался за 500р.') == ['Я ему сказал и т.к. он не послушался за 500р.']
    assert ru_sent_tokenize('Ура. Ура. 500р.') == ['Ура.', 'Ура.', '500р.']
    assert ru_sent_tokenize('Среди других её представителей — Л. Р. Зиндер, Л. В. Бондарко, М. И. Матусевич.') == \
           ['Среди других её представителей — Л. Р. Зиндер, Л. В. Бондарко, М. И. Матусевич.']
    assert ru_sent_tokenize('И. П. Павлов.') == ['И. П. Павлов.']
    assert ru_sent_tokenize('Павлов И. П., Сеченов И. М.') == ['Павлов И. П., Сеченов И. М.']
    assert ru_sent_tokenize('Основателем школы является Л. В. Щерба.') == ['Основателем школы является Л. В. Щерба.']
    assert ru_sent_tokenize('Я ему сказале: "Чтобы ничего не трогале." Но он не послушался.') == \
           ['Я ему сказале: "Чтобы ничего не трогале."', 'Но он не послушался.']
    assert ru_sent_tokenize('Нефть за $27/барр. не снится.') == ['Нефть за $27/барр. не снится.']
    assert ru_sent_tokenize('Сказала я. Он оглянулся.') == ['Сказала я.', 'Он оглянулся.']
    assert ru_sent_tokenize(
        'Летописец Нестор относит их возникновение к I столетию н.э., когда св. Андрей, проповедуя в Киеве '
        'евангельское слово, отправился потом в Новгород, где он увидел чудо – парившихся в бане.') == \
           ['Летописец Нестор относит их возникновение к I столетию н.э., когда св. Андрей, проповедуя в Киеве евангельское слово, отправился потом в Новгород, где он увидел чудо – парившихся в бане.']
    assert ru_sent_tokenize(
        '- Ну, хорошо, хочешь, я тебе тоннели покажу? - спрашивает наконец Мариам и сворачивает с одной ничем не примечательной улицы, застроенной обычными городскими многоэтажками, на другую точно такую же.') == ['- Ну, хорошо, хочешь, я тебе тоннели покажу? - спрашивает наконец Мариам и сворачивает с одной ничем не примечательной улицы, застроенной обычными городскими многоэтажками, на другую точно такую же.']
    assert ru_sent_tokenize('Где они были эти …адцать лет?') == ['Где они были эти …адцать лет?']
    assert ru_sent_tokenize('Православие... более всего подходит на роль такой идеи...') == ['Православие... более всего подходит на роль такой идеи...']
    assert ru_sent_tokenize('Yolka стоит 2400р. без трех копеек сто долларов, между прочим.') == ['Yolka стоит 2400р. без трех копеек сто долларов, между прочим.']
    assert ru_sent_tokenize(
        'А если лень читать всё - общее количество ответов: 8272! - то можно почитать книжку избранных мест.') == ['А если лень читать всё - общее количество ответов: 8272! - то можно почитать книжку избранных мест.']
    assert ru_sent_tokenize('Это стоило 50 т. к. вчера') == ['Это стоило 50 т. к. вчера']
    assert ru_sent_tokenize(
        "Официально закрытие фастфудов назначено на предстоящее воскресенье, причём менеджеры не планируют снова открывать в этой стране рестораны McDonald's. Причин закрытия исландских McDonald's несколько.") == \
           ["Официально закрытие фастфудов назначено на предстоящее воскресенье, причём менеджеры не планируют снова открывать в этой стране рестораны McDonald's.",
            "Причин закрытия исландских McDonald's несколько."]
    assert ru_sent_tokenize(
        '12 января ожидается понижение до минус 44 — минус 48°C. В школах региона отменены занятия в начальных классах.') == \
           ['12 января ожидается понижение до минус 44 — минус 48°C.',
            'В школах региона отменены занятия в начальных классах.']
    assert ru_sent_tokenize(
        'У государственных людей тоже есть дети, и если для них ночные заведения работать-таки будут… (а вы попробуйте им отказать) ну, в общем, не хотелось бы думать о волне народного возмущения.') == \
           ['У государственных людей тоже есть дети, и если для них ночные заведения работать-таки будут… (а вы попробуйте им отказать) ну, в общем, не хотелось бы думать о волне народного возмущения.']
    assert ru_sent_tokenize(
        'По сравнению с 2009 годом Россия опустилась в рейтинге на 9 позиций (т. е. ситуация в ней улучшилась).') == \
           ['По сравнению с 2009 годом Россия опустилась в рейтинге на 9 позиций (т. е. ситуация в ней улучшилась).']
    logging.info('all tests passed!')