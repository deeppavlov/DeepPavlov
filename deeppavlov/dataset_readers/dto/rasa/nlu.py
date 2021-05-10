import re
from collections import defaultdict
from typing import List, Tuple, Dict

slots_markup_pattern = r"\[" + \
                               r"(?P<slot_value>.*?)" + \
                               r"\]" + \
                               r"\(" + \
                               r"(?P<slot_name>.*?)" + \
                               r"\)"


class IntentLine:
    def __init__(self, text, cleaned_text_slots: List[Tuple] = None):
        if cleaned_text_slots is None:
            cleaned_text_slots = list()
        self.text = text
        self.slots_key = tuple(sorted((slot[0], slot[1])
                                      for slot in cleaned_text_slots))
        self.slots_di = cleaned_text_slots
        self.slot_name2text2value = None

    @classmethod
    def from_line(cls, line, ignore_slots=False):
        intent_text_w_markup = line.strip().strip('-').strip()
        line_slots_found = re.finditer(slots_markup_pattern,
                                       intent_text_w_markup)
        if ignore_slots:
            line_slots_found = []

        curr_char_ix = 0
        intent_text_without_markup = ''
        cleaned_text_slots = []  # intent text can contain slots highlighted

        slot_name2text2value = defaultdict(lambda: defaultdict(list))

        for line_slot in line_slots_found:
            line_slot_l_span, line_slot_r_span = line_slot.span()
            # intent w.o. markup for "some [entity](entity_example) text" is "some entity text"
            # so we should remove brackets and the parentheses content
            intent_text_without_markup += intent_text_w_markup[
                                          curr_char_ix:line_slot_l_span]

            slot_value_text = str(line_slot["slot_value"])
            slot_name = line_slot["slot_name"]
            slot_value = slot_value_text
            if ':' in slot_name:
                # e.g. [moderately](price:moderate)
                slot_name, slot_value = slot_name.split(':', 1)

            slot_value_new_l_span = len(
                intent_text_without_markup)  # l span in cleaned text
            slot_value_new_r_span = slot_value_new_l_span + len(
                slot_value_text)  # r span in cleaned text
            # intent w.o. markup for "some [entity](entity_example) text" is "some entity text"
            # so we should remove brackets and the parentheses content
            intent_text_without_markup += slot_value_text

            cleaned_text_slots.append((slot_name, slot_value))

            slot_name2text2value[slot_name][slot_value_text].append(slot_value)

            curr_char_ix = line_slot_r_span
        intent_text_without_markup += intent_text_w_markup[
                                      curr_char_ix: len(intent_text_w_markup)]

        intent_l = cls(intent_text_without_markup, cleaned_text_slots)
        intent_l.slot_name2text2value = slot_name2text2value

        return intent_l


class IntentDesc:
    def __init__(self, title):
        self.title = title
        self.lines = list()

    def add_line(self, intent_line:IntentLine):
        self.lines.append(intent_line)


class Intents:
    def __init__(self):
        self.intents: List[IntentDesc] = list()
        self.lines = None
        self._slot_name2text2value = None
        self._intent2slot2text = None

    @property
    def slot_name2text2value(self) -> Dict:
        if self._slot_name2text2value is not None:
            return self._slot_name2text2value
        sn2t2v = dict()
        for intent in self.intents:
            for intent_l in intent.lines:
                for slot_name, slot_text2value in intent_l.slot_name2text2value.keys():
                    if slot_name not in sn2t2v.keys():
                        sn2t2v[slot_name] = dict()
                    for slot_text, slot_values_li in slot_text2value.items():
                        if slot_text not in sn2t2v[slot_name].keys()
                            sn2t2v[slot_name][slot_text] = list()
                        sn2t2v[slot_name][slot_text].extend(slot_values_li)
        self._slot_name2text2value = sn2t2v
        return sn2t2v

    @property
    def intent2slot2text(self) -> Dict:
        if self._intent2slot2text is not None:
            return self._intent2slot2text

        intent2slots2text = dict()
        for intent in self.intents:
            slots2text = dict()
            intent_title = intent.title
            for intent_l in intent.lines:
                slots2text[intent_l.slots_key] =  {"text": intent_l.text,
                     "slots_di": intent_l.slots_di,
                     "slots": intent_l.slots_key}
            intent2slots2text[intent_title] = slots2text
        self._intent2slot2text = intent2slots2text
        return intent2slots2text


    @classmethod
    def from_nlu_md(cls, lines):
        intents = cls()
        ignore_slots = False
        for line in lines:
            if line.startswith("##"):
                # lines starting with ## are starting section describing new intent type
                curr_intent_name = line.strip("##").strip().split("intent:", 1)[-1]
                curr_intent = IntentDesc(curr_intent_name)
                intents.intents.append(curr_intent)
            if line.strip().startswith('-'):
                # lines starting with - are listing the examples of intent texts of the current intent type
                intent_l = IntentLine.from_line(line, ignore_slots)
                # noinspection PyUnboundLocalVariable
                curr_intent.add_line(intent_l)