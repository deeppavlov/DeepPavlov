#!/bin/env python3
# -*- coding: utf-8 -*-

import json
import re


PHRASES_TO_NORMALIZE = [
    (re.compile("there are restaurants that area would you like"),
     "what part of town do you have in mind"),
    (re.compile("in the moderate price range"),  "and the prices are moderate"),
    (re.compile("in the expensive price range"), "and the prices are expensive"),
    (re.compile("in the cheap price range"),     "and the prices are cheap"),
    (re.compile("is a nice restaurant"),         "is a nice place"),
    (re.compile("is a great restaurant"),        "is a nice place"),
    (re.compile("is a nice place"),              "is"),
    (re.compile("the west part"),                "the west"),
    (re.compile("the centre part"),              "the centre"),
    (re.compile("the south part"),               "the south"),
    (re.compile("serving tasty"),                "serving"),
    (re.compile("noise "),                       ""),
    (re.compile("unintelligible "),              ""),
    (re.compile(" unintelligible"),              ""),
    (re.compile(" noise"),                       ""),
    (re.compile("is a great [a-z]+ serving"
                " ([a-z ]+) food and it is"
                " in the ([a-z]+) price range"),
     "serves \\1 food in the \\2 price range"),
    (re.compile("([a-z ]+ is on)[a-z ]+"),  "\\1"),
    (re.compile("(sure [a-z ]+ is on)[a-z ]+"),  "\\1"),
    (re.compile("(the phone number of[a-z ]* is).*"), "\\1"),
    (re.compile("(the post code of[a-z ]* is).*"), "\\1"),
    (re.compile("([a-z ]+) is and the prices are ([a-z ]+)"),
     "the price range at \\1 is \\2"),
    (re.compile("the price range at ([a-z ]+) is serving ([a-z ]+) food and it is ([a-z ]+)"),
     "\\1 serves \\2 food and the prices are \\3"),
    (re.compile("and it is"),                    ""),
    (re.compile("i m sorry but there is no restaurant serving ([a-z ]+) food"),
     "i am sorry but there is no \\1 restaurant that matches your request"),
    (re.compile(".*there is no ([a-z ]+) restaurant.*"),
     "i am sorry but there is no \\1 restaurant that matches your request")


]
#    (re.compile("(the phone number of .* is).*( and.*)"), "\\1\\2"),
#    (re.compile("(the phone number of .* is).*(?! and.*)"), "\\1"),
#    (re.compile("(.*) and it is (.*)"), "\\1\\2")
#    (re.compile(), )


def normalize(texts):
    normalized = []
    for t in texts:
        words = re.findall("[^\W\d_]+", t.lower())
        norm_text = " ".join(words)
        for phr, norm_phr in PHRASES_TO_NORMALIZE:
            if phr.search(norm_text):
                norm_text = phr.sub(norm_phr, norm_text)
        normalized.append(norm_text)
    return ",".join(normalized)


def check_if_dialog_exists(key, dialog, data, duplicates):
    if key in data:
        if dialog == data[key]:
            duplicates += 1
        else:
            raise Exception(f"dialog key {key} repeated in data"
                            f", dialogs=\n{data[key]}\n{dialog}")


def load_dstc2(data_type, data):

    path = '/home/vimary/.deeppavlov/downloads/dstc2_new/dstc2-{}.jsonlist'.format(data_type)
    duplicates = 0

    def check_if_dstc2_dialog_exists(key, dialog):
        check_if_dialog_exists(key, dialog, data=data, duplicates=duplicates)

    dialog = []
    for ln in open(path, 'rt'):
        if ln.strip():
            turn = json.loads(ln)
            # TODO: handle sil user responses
            if turn['text'] not in ('sil', 'unintelligible', 'noise'):
                dialog.append(turn)
        else:
            if dialog:
                dialog_key = normalize((turn['text'] for turn in dialog
                                        if 'api_call' not in turn['text']))
                check_if_dstc2_dialog_exists(dialog_key, dialog)
                data[dialog_key] = dialog
                dialog = []
    if dialog:
        check_if_dstc2_dialog_exists(dialog_key, dialog)
        data.append(dialog)

    print(f"Found {duplicates} duplicates for dstc2-{data_type}.")
    print(f"dstc2 has now size equal to {len(data)}.\n")
    return data


def read_babi(data_type):
    path = '/home/vimary/ipavlov/ParlAI/data/dialog-bAbI/dialog-bAbI-tasks/'\
        'dialog-babi-task6-dstc2-{}.txt'.format(data_type)
    data = {}
    duplicates = 0

    def check_if_babi_dialog_exists(key, dialog):
        check_if_dialog_exists(key, dialog, data=data, duplicates=duplicates)

    dialog = []
    in_api_call = False
    for ln in open(path, 'rt'):
        if ln.strip():
            _, text = ln.split(' ', 1)
            if not in_api_call:
                user_text, system_text = text.split('\t', 1)
                if '<SILENCE>' not in user_text:
                    dialog.append(user_text)
                if 'api_call' not in system_text:
                    dialog.append(system_text)
                else:
                    in_api_call = True
            else:
                if '<SILENCE>' in text:
                    in_api_call = False
                    user_text, system_text = text.split('\t', 1)
                    dialog.append(system_text)
        else:
            if dialog:
                if in_api_call:
                    raise Exception(f"Still in api_call after end of dialog"
                                    f", dialog={dialog}")
                dialog_key = normalize(dialog)
                check_if_babi_dialog_exists(dialog_key, dialog)
                data[dialog_key] = dialog
                dialog = []
                in_api_call = False
    if dialog:
        if in_api_call:
            raise Exception(f"Still in api_call after end of dialog, dialog={dialog}")
        dialog_key = normalize(dialog)
        check_if_babi_dialog_exists(dialog_key, dialog)
        data[dialog_key] = dialog

    print(f"Found {duplicates} duplicates for babi6-{data_type}.")
    print(f"Loaded {len(data)} dialogues from babi6-{data_type}.\n")
    return data


def fuzzy_find(key, keys, threshold=0.9):
    turns = key.split(',')
    import fuzzyset
    fuzzyset = fuzzyset.FuzzySet()
    fuzzyset.add(key)
    best_k, best_sc = None, threshold
    for k in keys:
        k_turns = k.split(',')
        if (len(k_turns) == len(turns)) and\
                (turns[:2] == k_turns[:2]) and\
                (('unintelligible' in key) == ('unintelligible' in k)):
            score = fuzzyset.get(k)[0][0]
            if score > best_sc:
                best_k = k
                best_sc = score
        #if k_turns[:2] == turns[:2]:
        #    if (k_turns[:3] == turns[:3]) and (k_turns[-3:] == turns[-3:]):
        #        return k
        #    else:
        #        print(turns)
        #        print(k_turns)
        #        break
    if (best_sc > threshold) and (best_sc < threshold + 0.005):
        print("BABI =", best_k)
        print("DSTC2 =", k)
    return best_k


def fuzzy_find2(key, keys, threshold=0.7):

    from collections import Counter

    def calc_counter_distance(A, B):
        A, B = set(A.most_common()), set(B.most_common())
        return calc_set_distance(A, B)

    def calc_set_distance(A, B):
        return float(len(A.intersection(B)))/max(len(A), len(B))

    turns = key.split(',')
    best_k, best_sc = None, threshold
    words = Counter((w for turn in turns for w in turn.split()))
    for k in keys:
        k_turns = k.split(',')
        #if (len(k_turns) == len(turns)) and\
        #        (turns[:2] == k_turns[:2]):
        if turns[-2:] == k_turns[-2:]:
            k_words = Counter((w for turn in k_turns for w in turn.split()))
            score = calc_counter_distance(words, k_words)
            if set(turns).issubset(k_turns):
                score = 0.95
            elif len(set(turns) - set(k_turns)) < 0.4 * len(set(k_turns)):
                score = 0.95 - 0.001 * (len(set(turns) - set(k_turns)))
            if abs(len(k_words) - len(words)) > 10:
                score = 0.

            if score > best_sc:
                best_k = k
                best_sc = score

    if (best_sc > threshold) and (best_sc < threshold + 0.001):
        print("BABI =", best_k)
        print("DSTC2 =", k)

    return best_k



BABI_NORM_TO_SKIP = [
]


def filter_turns(babi_dialog, dstc2_dialog, window_size=30, threshold=0.4):
    import fuzzyset
    babi_norm = [normalize([t]) for t in babi_dialog]
    if babi_norm in BABI_NORM_TO_SKIP:
        return None
    dstc2_norm = [normalize([t['text']]) for t in dstc2_dialog]

    j = 0
    filtered = []
    skipped = []
    for i in range(len(babi_dialog)):
        try:
            if babi_norm[i] == dstc2_norm[j]:
                filtered.append(dstc2_dialog[j])
                j += 1
            elif babi_norm[i] == dstc2_norm[j+1]:
                if 'api_call' in dstc2_dialog[j]['text']:
                    filtered.append(dstc2_dialog[j])
                filtered.append(dstc2_dialog[j+1])
                j += 2
            else:
                fuzzykey = fuzzyset.FuzzySet()
                fuzzykey.add(babi_norm[i])
                best_j, best_sc = None, threshold
                for j_ in range(j, min(j + window_size, len(dstc2_norm))):
                    res = fuzzykey.get(dstc2_norm[j_])
                    if not res:
                        score = 0
                    else:
                        score = res[0][0]
                    if score > best_sc:
                        best_j = j_
                        best_sc = score
                if best_j is None:
                    raise Exception(f"Couldn't match '{babi_norm[i]}' with"
                                    f" one of dstc2 turns "
                                    f"'{dstc2_norm[j:j + window_size]}'")
                    print(f"Skipping dialog with babi_norm = {babi_norm}\n"
                          f"dstc2_norm = {dstc2_norm}")
                    skipped.append(babi_dialog)
                    return None
                if 'api_call' in dstc2_dialog[best_j-1]['text']:
                    filtered.append(dstc2_dialog[best_j-1])
                filtered.append(dstc2_dialog[best_j])
                print(f"Matched '{babi_norm[i]}' and '{dstc2_norm[best_j]}'.")
                j = best_j + 1
        except Exception as msg:
            raise Exception(f"i = {i}, j = {j}\nbabi_norm_i = {babi_norm[i]}"
                            f"\nbabi_norm  = {babi_norm}"
                            f"\ndstc2_norm = {dstc2_norm}"
                            f"\nfiltered   = {[t['text'] for t in filtered]}"
                            f"\nCaught exception with '{msg}'")

    return filtered


def get_new_data(babi_data, dstc2_data):
    new_data = []
    not_found_data = []
    fuzzy_found = 0
    match_found = 0
    not_found = 0
    for i, dialog_key in enumerate(babi_data.keys()):
        if (i - 1) % 100 == 0:
            print(f"Processed {i} samples")
        if dialog_key in dstc2_data:
            #new_data.append(dstc2_data[dialog_key])
            match_found += 1
        else:
            dstc2_fuzzy_key = fuzzy_find2(dialog_key, dstc2_data.keys())
            if dstc2_fuzzy_key is not None:
                # new_data.append(dstc2_data[dstc2_fuzzy_key])
                dialog = filter_turns(babi_data[dialog_key], dstc2_data[dstc2_fuzzy_key])
                if dialog:
                    new_data.append(dialog)
                    fuzzy_found += 1
                not_found += 1
            else:
                not_found += 1
                not_found_data.append(babi_data[dialog_key])
                # raise Exception(f"Couldn't fuzzy find dialog with key: {dialog_key}.")
    print(f"Fully match: {match_found}/{len(babi_data)}\n"
          f"Fuzzy match: {fuzzy_found}/{len(babi_data)}\n"
          f"  Not found: {not_found}/{len(babi_data)}")
    return new_data, not_found_data


def main():
    dstc2_data = {}
    load_dstc2('trn', dstc2_data)
    load_dstc2('val', dstc2_data)
    load_dstc2('tst', dstc2_data)

    babi_trn = read_babi('trn')
    babi_val = read_babi('dev')
    babi_tst = read_babi('tst')

    new_dstc2_trn, notfound = get_new_data(babi_trn, dstc2_data)
    with open('new_data.trn.fuzzy.jsonlist', 'wt') as f:
        for turn in new_dstc2_trn:
            f.write(json.dumps(turn) + '\n')
    with open('new_data.trn.notfound.jsonlist', 'wt') as f:
        for turn in notfound:
            f.write(json.dumps(turn) + '\n')

    """
    new_dstc2_val, notfound = get_new_data(babi_val, dstc2_data)

    new_dstc2_tst, notfound = get_new_data(babi_tst, dstc2_data)
    with open('new_data.tst.fuzzy.jsonlist', 'wt') as f:
        for turn in new_dstc2_trn:
            f.write(json.dumps(turn) + '\n')
    with open('new_data.tst.notfound.jsonlist', 'wt') as f:
        for turn in notfound:
            f.write(json.dumps(turn) + '\n')
    """


if __name__ == "__main__":
    main()
