
import collections
import pprint
import numpy as np
import random
import copy
import re

from deeppavlov.models.transformer_chit_chat.hacks_utils import spec_utters, yandex_api, stop_words

punct = re.compile('[!.,?]')


def hello_bye(input_utter):
    # correct
    answers = None
    if input_utter =='/start':
        answers = random.sample(spec_utters.sample_inttros, 1)[0]
    if [None for mtch in spec_utters.match_ends if mtch in input_utter]:
        answers = random.sample(spec_utters.sample_ends, 1)[0]
    return answers


def split_text(text):
    text = text.split('<t>')
    return [t.strip() for t in text]


def clean_text(text):
    text = text.strip().lower()
    char_text = punct.sub('', text)
    char_text = ' '.join([spec_utters.MORPH.parse(word)[0].normal_form for word in char_text.split()])
    return split_text(text), split_text(char_text)


def clean_hypt(hypt):
    hypt = hypt.strip().lower()
    char_hypt = punct.sub('', hypt)
    char_hypt = ' '.join([spec_utters.MORPH.parse(word)[0].normal_form for word in char_hypt.split()])
    return char_hypt


def clean_pers(pers):
    text = ' <t> '.join(pers)
    return clean_text(text)


def clean_his(his):
    text = ' <t> '.join(his)
    text = yandex_api.send(text)
    return clean_text(text)


def len_filter(hypt, min_len=2):
    return len(hypt.split()) > min_len


def rm_sw_utter(utter, min_len=2):
    utter = punct.sub('', utter)
    words = utter.strip().lower().split()
    words = collections.OrderedDict(zip(words, [0]*len(words)))
    [words.pop(i, None)for i in stop_words.stop_words]
    utter = ' '.join(words.keys())
    return utter


def his_repeat_filter(cntx, hypt, his_len=4, max_repeat=2):
    his = cntx.his[1::2][-his_len:]
    his = set([rm_sw_utter(utter) for utter in his])
    # res = len([None for utter in char_his[-his_len:] if hypt.strip().lower() == utter.strip().lower()]) < max_repeat
    # print(f'{his} & {set([rm_sw_utter(hypt)])}')
    res = his & set([rm_sw_utter(hypt)])
    # print(f'{res} - {hypt}')
    return not bool(res)


def hypt_repeat_filter(cntx, hypt):
    if not hasattr(cntx, 'hypt_set'):
        cntx.hypt_set = set()
    hypt = punct.sub('', hypt)
    hypt = hypt.lower().strip()
    if hypt in cntx.hypt_set:
        return False
    else:
        cntx.hypt_set.add(hypt)
        return True


def intersection_persona_filter(cntx, hypt):
    # print(f'hyp - {set(clean_hypt(hypt).split())}')
    # print(f'persona_word_set - {cntx.persona_word_set}')
    # print(f'res - {set(clean_hypt(hypt).split()) & set(cntx.persona_word_set)}')
    return set(clean_hypt(hypt).split()) & set(cntx.persona_word_set)


def intersection_last_utter_filter(cntx, hypt):
    # print(f'hyp - {set(clean_hypt(hypt).split())}')
    # print(f'last_uttr_word_set - {cntx.last_uttr_word_set}')
    # print(f'res - {set(clean_hypt(hypt).split()) & cntx.last_uttr_word_set}')
    return set(clean_hypt(hypt).split()) & cntx.last_uttr_word_set


def cntx_analysis(cntx):
    cntx.persona_word_set = set(' '.join(cntx.char_persona).split())
    cntx.persona_word_set = cntx.persona_word_set - (cntx.persona_word_set & stop_words.stop_words)
    cntx.his_word_set = set(' '.join(cntx.char_his).split())
    cntx.his_word_set = cntx.his_word_set - (cntx.his_word_set & stop_words.stop_words)
    cntx.last_uttr_word_set = set(cntx.char_his[-1].split())
    cntx.last_uttr_word_set = cntx.last_uttr_word_set - (cntx.last_uttr_word_set & stop_words.stop_words)


def renew_hypt_conf(hypts, change_conf):
    hypts = copy.deepcopy(hypts)
    hypts = [(change_conf(conf, hypt), hypt) for conf, hypt in hypts]
    return hypts


def switch_hypt(cntx):
    hypts = cntx.hypts
    hypts = [(conf, hypt) for conf, hypt in hypts if len_filter(hypt, min_len=2)]
    # print(f'cntx.his[1::2] = {cntx.his[1::2]}')
    hypts = [(conf, hypt) for conf, hypt in hypts if his_repeat_filter(cntx, hypt, his_len=60, max_repeat=1)]
    hypts = [(conf, hypt) for conf, hypt in hypts if hypt_repeat_filter(cntx, hypt)]
    persona_correlation_hypts = [(conf, hypt) for conf, hypt in hypts if intersection_persona_filter(cntx, hypt)]
    # pprint.pprint(persona_correlation_hypts)
    last_correlation_hypts = [(conf, hypt) for conf, hypt in hypts if intersection_last_utter_filter(cntx, hypt)]
    # pprint.pprint(last_correlation_hypts)
    hypts.sort(key=lambda x: x[0], reverse=True)
    hight_prob_hypts = hypts[: 3]

    hight_prob_hypts = renew_hypt_conf(hight_prob_hypts, lambda c, h: 1)
    last_correlation_hypts = renew_hypt_conf(last_correlation_hypts, lambda c, h: 4)
    persona_correlation_hypts = renew_hypt_conf(persona_correlation_hypts, lambda c, h: 2)
    # hight_prob_hypts = renew_hypt_conf(hight_prob_hypts, lambda c, h: c*1)
    # last_correlation_hypts = renew_hypt_conf(last_correlation_hypts, lambda c, h: c*2)
    # persona_correlation_hypts = renew_hypt_conf(persona_correlation_hypts, lambda c, h: c*2)
    res_hypts = hight_prob_hypts + last_correlation_hypts + persona_correlation_hypts

    def drop_conf_of_question(conf, hypt):
        if len(cntx.his) < 20 and '?' in hypt:
            # if last utter is question

            if '?' in cntx.his[-1]:
                return conf/(2**2)
            bot_utters = cntx.his[1::2]
            decrease_pow = len([None for utter in bot_utters[-2:] if '?' in utter])*2 - 1
            return conf/(2**decrease_pow)
        else:
            return conf

    res_hypts = renew_hypt_conf(res_hypts, drop_conf_of_question)
    # pprint.pprint(res_hypts)
    if res_hypts:
        confs, answers = list(zip(*res_hypts))
        # pprint.pprint(confs)
        # pprint.pprint(answers)

        confs = np.array(confs)
        confs = confs/confs.sum()
        return np.random.choice(answers, p=confs)
    else:
        return random.sample(["Не знаю, что сказать... Как дела?",
                              "Как все сложно. Извини, не понимаю.",
                              "Не понимаю."], 1)[0]


def hacking(persona, his, hyp_answers, confs):
    def cntx(x): return x
    cntx.hypts = [(conf, ans) for conf, ans in zip(confs, hyp_answers)]
    cntx.persona, cntx.char_persona = clean_pers(persona)
    cntx.his, cntx.char_his = clean_his(his)
    hb_utter = hello_bye(cntx.char_his[-1])
    if hb_utter:
        return hb_utter
    cntx_analysis(cntx)
    return switch_hypt(cntx)
