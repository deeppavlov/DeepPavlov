#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from nltk import word_tokenize

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download
# from deeppavlov.dataset_iterator.dialod_iterator import DialogStateDatasetIterator


USER_ACTION_MAP = {
    'inform': 'INFORM',
    'affirm': 'AFFIRM',
    'ack': 'AFFIRM',
    'reqalts': 'REQUEST_ALTS',
    'reqmore': 'REQUEST_ALTS',
    'thankyou': 'THANK_YOU',
    'bye': 'GOOD_BYE',
    'negate': 'NEGATE',
    'deny': 'NEGATE',
    'restart': 'CANT_UNDERSTAND',
    'repeat': 'CANT_UNDERSTAND',
    'hello': 'GREETING',
    'request': 'REQUEST',
    'confirm': 'CONFIRM'
}

SYSTEM_ACTION_MAP = {
    'welcomemsg': 'GREETING',
    'bye': 'GOODBYE',
    'canthear': 'CANTHEAR',
    'canthelp': 'NEGATE',
    'confirm-domain': 'CONFIRM_DOMAIN',
    'inform': 'INFORM',
    'expl-conf': 'CONFIRM',
    'expl_conf': 'CONFIRM',
    'impl-conf': 'CONFIRM',
    'offer': 'OFFER',
    'repeat': 'REPEAT',
    'reqmore': 'REQMORE',
    'request': 'REQUEST',
    'select': 'SELECT'
}


class SlotConverter:

    def __init__(self, data_path: str):
        dataset_path = expand_path(data_path) / 'slot_vals.json'
        self._build_slot_vals(dataset_path)
        with open(dataset_path, encoding='utf-8') as f:
            self._slot_vals = json.load(f)

    def get_slot_mentions(self, tokens, slots):
        n_toks = len(tokens)
        mentions = {}
        for n in range(n_toks):
            for slot in slots:
                entity_pull = set([slot['value']])
                if slot['slot'] in self._slot_vals:
                    if slot['value'] in self._slot_vals[slot['slot']]:
                        entity_pull |= set(self._slot_vals[slot['slot']][slot['value']])
                for entity in entity_pull:
                    slot_tokens = entity.split()
                    slot_len = len(slot_tokens)
                    if (n + slot_len <= n_toks) and \
                            self._cmp_seqs(tokens[n: n + slot_len], slot_tokens):
                        if slot['slot'] in mentions:
                            s0, s1 = mentions[slot['slot']]
                            if (((n - s0) * (n + slot_len - 1 - s0) <= 0) or
                                    ((n - s1) * (n + slot_len - 1 - s1) <= 0))\
                                    and (s1 - s0 >= slot_len):
                                break
                        mentions[slot['slot']] = (n, n + slot_len)
                        break
        return [{'slot': k, 'start': v[0], 'exclusive_end': v[1]}
                for k, v in mentions.items()]

    @staticmethod
    def _cmp_seqs(seq1, seq2):
        equality_list = [tok1 == tok2 for tok1, tok2 in zip(seq1, seq2)]
        return all(equality_list)

    @staticmethod
    def _build_slot_vals(slot_vals_json_path='data/'):
        url = 'http://files.deeppavlov.ai/datasets/dstc_slot_vals.json'
        download(slot_vals_json_path, url)


slot_converter = SlotConverter('dstc2')


def get_acts(a, action_map):
    act_type = action_map[a['act']]
    if a['slots']:
        for slot_type, slot_val in a['slots']:
            if slot_type == 'slot':
                yield {'type': act_type, 'slot': slot_val}
            else:
                yield {'type': act_type, 'slot': slot_type}
    else:
        yield {'type': act_type}


def dstc2google(new_turn, turn, slotval2mention={}, old_dialog_state={}):
    # TODO: convert slot values to slot mentions in `goals`
    if 'goals' in turn:

        u_tokens = turn['text'].split()

        slots = [{'slot': s, 'value': v}
                 for a in turn['dialog_acts'] for s, v in a['slots'] if s not in ('slot')]
        state_diff = dict(set(turn['goals'].items()) - set(old_dialog_state.items()))
        for i in range(len(slots)):
            s = slots[i]
            if s['slot'] == 'this':
                if s['value'] in state_diff.values():
                    renames = [k for k, v in state_diff.items() if v == s['value']]
                    slots[i]['slot'] = renames[0]
                elif s['value'] not in turn['goals'].keys():
                    slots = slots[:i] + slots[i + 1:]
                break

        slot_mentions = slot_converter.get_slot_mentions(u_tokens, slots)

        u_acts = [a0 for a in turn['dialog_acts'] for a0 in get_acts(a, USER_ACTION_MAP)]
        nonmentioned_slots = [s['slot'] for s in slots
                              if s['slot'] not in (a.get('slot') for a in u_acts)]
        for i in range(len(u_acts)):
            a = u_acts[i]
            if (a.get('slot') == 'this') and (a['type'] == 'INFORM'):
                if len(nonmentioned_slots) == 1:
                    u_acts[i]['slot'] = nonmentioned_slots[0]
                elif not len(slots):
                    u_acts = u_acts[:i] + u_acts[i+1:]
                break

        # slot_mentions = DialogStateDatasetIterator._biomarkup2list(u_tokens, u_tags)
        state = [{'slot': s, 'value': v} for s, v in turn['goals'].items()]

        user = {
            'dialogue_state': state,
            'user_acts': u_acts,
            'user_intents': ["RESERVE_RESTAURANT"],
            'user_utterance': {
                'slots': slot_mentions,
                'slot_values': slots,
                'text': turn['text'],
                'tokens': u_tokens
            }
        }
        new_turn.update(user)
    else:
        u_tokens = word_tokenize(turn['text'].lower())
        slots = [{'slot': s, 'value': v}
                 for a in turn['dialog_acts'] for s, v in a['slots'] if s not in ('slot')]
        slot_mentions = slot_converter.get_slot_mentions(u_tokens, slots)
        u_parsed_acts = [
            {'act': a_split[0],
             'slots': filter(lambda s_v: s_v[0] in a_split[1:], act['slots'])}
            if len(a_split) > 1 else {'act': a, 'slots': act['slots']}
            for act in turn['dialog_acts']
            for a in act['act'].split('+') for a_split in [a.split('_')]
        ]
        u_acts = [a0 for a in u_parsed_acts for a0 in get_acts(a, SYSTEM_ACTION_MAP)]
        system = {
            'system_acts': u_acts,
            'system_utterance': {
                'slots': slot_mentions,
                'slot_values': slots,
                'text': ' '.join(u_tokens),
                'tokens': u_tokens
            }
        }
        new_turn.update(system)
    return new_turn


def main(data):

    def get_id(idx):
        return f"restaurant_{idx:010}"

    res = [{'dialogue_id': get_id(0), 'turns': []}]
    new_turn, last_state = {}, []
    data_it = iter(data)
    for turn in data_it:
        if not turn:
            res.append({'dialogue_id': get_id(len(res)), 'turns': []})
            new_turn, last_state = {}, {}
        else:
            # if system action is an api call, then skip the turn and the next user's turn
            if turn['text'].lower().startswith('api_call'):
                data_it.__next__()
                continue
            new_turn = dstc2google(new_turn, turn)
            last_state = turn['goals'] if 'goals' in turn else last_state
            if 'dialogue_state' in new_turn:
                res[-1]['turns'].append(new_turn)
                new_turn, last_state = {}, {}
    if turn:
        return res
    return res[:-1]


if __name__ == "__main__":
    data = main((json.loads(ln) if ln.strip() else None
                 for ln in sys.stdin.readlines()))
    sys.stdout.write(json.dumps(data, indent=2))
