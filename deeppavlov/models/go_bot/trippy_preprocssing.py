# Copyright 2021 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



### Implements Preprocessing functions to turn DSTC2 into TripPy readable format ###

### Funcs ###

import re
from logging import getLogger
from typing import Dict, Any, List, Optional, Union, Tuple

import six
import numpy as np
import torch

logger = getLogger(__name__)


## EXP ##
from deeppavlov.models.spelling_correction.levenshtein.searcher_component import LevenshteinSearcherComponent

class DSTExample(object):
    """
    Taken from TripPy except for the __repr__ function,
    as it contains data type mistakes (e.g. lbl is a dict not number)
    A single training/test example for the DST dataset.
    """
    def __init__(self,
                 guid,
                 text_a,
                 text_b,
                 history,
                 text_a_label=None,
                 text_b_label=None,
                 history_label=None,
                 values=None,
                 inform_label=None,
                 inform_slot_label=None,
                 refer_label=None,
                 diag_state=None,
                 class_label=None,
                 action_label=None,
                 prev_action_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.history = history
        self.text_a_label = text_a_label
        self.text_b_label = text_b_label
        self.history_label = history_label
        self.values = values
        self.inform_label = inform_label
        self.inform_slot_label = inform_slot_label
        self.refer_label = refer_label
        self.diag_state = diag_state
        self.class_label = class_label
        self.action_label = action_label
        self.prev_action_label = prev_action_label

# From bert.tokenization (TF code) # From TripPy
def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


# From TripPy MultiWoz
def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text) # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text) # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text) # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text) # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text) # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text) # Correct times that use 24 as hour
    return text

# From TripPy MultiWoz, adjusted to add centre & pricerange
def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text) # Systematic typo
    text = re.sub("guesthouse", "guest house", text) # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text) # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text) # Normalization

    # Add inconsistent slot values here - Incorrect one first, then correct
    text = re.sub("price range", "pricerange", text) # These are very necessary, because there are discrepencies btw the TripPy slot values & the actual text
    text = re.sub("center", "centre", text)
    text = re.sub("moderately", "moderate", text)
    text = re.sub("medium", "moderate", text)
    text = re.sub("central", "centre", text)
    text = re.sub("portugese", "portuguese", text)
    text = re.sub("steak house", "steakhouse", text)
    text = re.sub("turkiesh", " turkish", text)
    text = re.sub("asian ori$", "asian oriental", text) # Cannot remove the space cuz else it will mess with normal asian ori; probably need some more regex
    text = re.sub("bristish", " british", text)
    text = re.sub("asian$", "asian oriental", text)
    text = re.sub("portugeuse", "portuguese", text)
    text = re.sub("sea food", "seafood", text)
    text = re.sub("australian asian", "australasian", text)
    text = re.sub("^derately", "moderately", text)


    return text


def tokenize(utt):
    """
    Returns tokenized utterance according to TripPy's MultiWoz code
    """
    utt_lower = convert_to_unicode(utt).lower()
    utt_lower = normalize_text(utt_lower)
    utt_tok = [tok for tok in map(str.strip, re.split("(\W+)", utt_lower)) if len(tok) > 0]
    return utt_tok


def get_sys_inform(response, slot_type):
    """
    DSTC2-specific function to check if the system informs in current round about specific slot. 

    Args:
        response: Response data
        slot_type: What slot type to check for
    Returns:
        boolean - if the system informs in round
    """

    return "inform_" + slot_type in response["act"] if (response is not None) and (response["act"] is not None) else False


def get_dialogue_state_sv_dict(context):
    """
    Creates dialogue state Slot-Value dict for current context
    In Trippy, DS is generally given and then converted to an SV dict - We create the DS & convert it in one
    In order to get an overall diag state we then merge the diag states with more recent ones taking preference
    """
    sv_dict = {}
    # There may be no slots & slot values, though it doesnt really make sense to use TripPy then
    if "intents" in context:
        for intent in context["intents"]:
            if (intent["slots"] is not None) and (len(intent["slots"]) > 0):
                # Only inform since we only want slots where we have the value (in request we dont)
                if intent["act"] == "inform":
                    slot = intent["slots"][0][0]
                    value = intent["slots"][0][1]
                    sv_dict[slot] = value
    return sv_dict


def get_tok_label(prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok,
                  sys_slot_label, usr_utt_tok, usr_slot_label, dial_id,
                  turn_id, slot_last_occurrence=True):
    """
    Creates labels of 11111111 for where an slot value is uttered by user/system

    Adapted from TripPy get_tok_label.
    The position of the last occurrence of the slot value will be used.
    """
    sys_utt_tok_label = [0 for _ in sys_utt_tok]
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    if slot_type not in cur_ds_dict:
        class_type = 'none'
    else:
        value = cur_ds_dict[slot_type]
        if value == 'dontcare' and (slot_type not in prev_ds_dict or prev_ds_dict[slot_type] != 'dontcare'):
            # Only label dontcare at its first occurrence in the dialog
            class_type = 'dontcare'
        else: # If not none or dontcare, we have to identify whether
            # there is a span, or if the value is informed
            in_usr = False
            in_sys = False
            for label_d in usr_slot_label:
                if (label_d['slot'] == slot_type) and (label_d['start'] >= 0):
                    # Allow start & end logits referring to utterances that do not exactly match the slot value; Not in original TripPy
                    # Start should still be >0 not to allow ones with -1, i.e. there is not value
                    #value == ' '.join(usr_utt_tok[label_d['start']:label_d['exclusive_end']]):
                    for idx in range(label_d['start'], label_d['exclusive_end']):
                        usr_utt_tok_label[idx] = 1
                    in_usr = True
                    class_type = 'copy_value'
                    if slot_last_occurrence:
                        break

            # This is never used for DSTC2, as there are no sys_slot_labels, however, we leave it for possible future use & TripPy conformity
            for label_d in sys_slot_label:
                if label_d['slot'] == slot_type and value == ' '.join(
                        sys_utt_tok[label_d['start']:label_d['exclusive_end']]):
                    for idx in range(label_d['start'], label_d['exclusive_end']):
                        sys_utt_tok_label[idx] = 1
                    in_sys = True
                    if not in_usr or not slot_last_occurrence:
                        class_type = 'inform'
                    if slot_last_occurrence:
                        break

            if not in_usr and not in_sys:
                assert sum(usr_utt_tok_label + sys_utt_tok_label) == 0
                if (slot_type not in prev_ds_dict or value != prev_ds_dict[slot_type]):
                    # Added clarifications
                    print("Value: {} for slot type {} does not exist in utterance: {}".format(value, slot_type, usr_utt_tok))
                    print("Other Information: ", prev_ds_dict, usr_slot_label, sys_utt_tok)
                    print("Most likely have to add the incorrect utterance token to be replaced with the value see normalizations in the code.")
                    print("This error is likely because of a mismatch in the slot value & the actual text!\n \
                          E.g. the slot value is pricerange: moderate but the text contains only moderately\n \
                          Possibly add a replacement to the normalization.")
                    raise ValueError('Copy value cannot found in Dial %s Turn %s' % (str(dial_id), str(turn_id)))
                else:
                    class_type = 'none'
            else:
                assert sum(usr_utt_tok_label + sys_utt_tok_label) > 0
    return sys_utt_tok_label, usr_utt_tok_label, class_type



def get_token_and_slot_label(context, response=None):
    """
    Creates tokenized version of text & labels with start, end & value of slots
    Adapted from TripPy "get_token_and_slot_label(turn):"

    Args:
        context: User-utterance related data
        response: System-utterance related data

    Returns:
        sys_utt_tok: Tokenized system utterances
        sys_slot_label: 

        usr_utt_tok: May be empty list
        usr_slot_label: May be empty list
    """
    # Note that TripPy Sim-M data already includes tokens; We tokenize it here using a func provided by TripPy for MultiWoz
    # This is need because of the label ids
    if response is not None:
        sys_utt_tok = tokenize(response['text'])
        sys_slot_label = [] # Not present in DP
    else:
        # Inference
        sys_utt_tok = []
        sys_slot_label = []


    usr_utt_tok = tokenize(context['text'])

    # Possibly simplify
    usr_slot_label = []
    if 'intents' in context:
        for intent in context['intents']:
            if (intent["slots"] is not None) and (len(intent["slots"]) > 0):
                if intent["act"] == "request":
                
                    slot = intent["slots"][0][1]
                    value = None # request slots have no value yet

                    # End & start queried lateron, so set to -1
                    slot_dict = {
                    "exclusive_end": -1,
                    "slot": slot,
                    "start": -1
                    }
                    usr_slot_label.append(slot_dict)

                elif intent["act"] == "inform":

                    slot = intent["slots"][0][0]
                    value = intent["slots"][0][1]

                    value_tok = tokenize(value)

                    # if e.g. "dontcare" skip it; It will be in the diag_state (like in TripPy)
                    # This unfortunately is not robust to slot values that are different from the text & they have to be manually added to replace
                    # Generally TripPy's text index predicting is brittle and we should move to generating slot values not copying them in the future
                    if set(value_tok) <= set(usr_utt_tok):
                        # Will have to be adpated for slot values of len > 1
                        slot_dict = {
                        "exclusive_end": usr_utt_tok.index(value_tok[-1]) + 1,
                        "slot": slot,
                        "start": usr_utt_tok.index(value_tok[0])
                        }
                        usr_slot_label.append(slot_dict)

                    elif value not in ["dontcare", "itdoesntmatter"]:
                        # Not in original TripPy - Search for most similar values with Levenshtein in case
                        # Slot value label is not in the user tokens
                        searcher = LevenshteinSearcherComponent(usr_utt_tok, max_distance=10)
                        candidates = searcher([[value]])
                        top_candidate = candidates[0][0][0][1]

                        # The LevenshteinSearcher seems not to work for removal edits
                        if top_candidate not in usr_utt_tok:
                            top_candidate = usr_utt_tok[0] # Just randomly take the first token

                        slot_dict = {
                        "exclusive_end": usr_utt_tok.index(top_candidate) + 1,
                        "slot": slot,
                        "start": usr_utt_tok.index(top_candidate),
                        "candidate": top_candidate
                        }
                        usr_slot_label.append(slot_dict)

    

    return sys_utt_tok, sys_slot_label, usr_utt_tok, usr_slot_label


def get_turn_label(context, response, cur_ds_dict, prev_ds_dict, slot_list, dial_id, turn_id,
                   delexicalize_sys_utts=False, unk_token="[UNK]", slot_last_occurrence=True):
    """
    Make turn_label a dictionary of slot with value positions or being dontcare / none:
    Turn label contains:
      (1) the updates from previous to current dialogue state,
      (2) values in current dialogue state explicitly mentioned in system or user utterance.
    """

    (sys_utt_tok, sys_slot_label, usr_utt_tok, usr_slot_label) = get_token_and_slot_label(context, response)

    sys_utt_tok_label_dict = {}
    usr_utt_tok_label_dict = {}
    inform_label_dict = {}
    inform_slot_label_dict = {}
    referral_label_dict = {}
    class_type_dict = {}

    for slot_type in slot_list:
        inform_label_dict[slot_type] = 'none'
        inform_slot_label_dict[slot_type] = 0
        referral_label_dict[slot_type] = 'none' # Referral is not present in data

        sys_utt_tok_label, usr_utt_tok_label, class_type = get_tok_label(
            prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok, sys_slot_label,
            usr_utt_tok, usr_slot_label, dial_id, turn_id,
            slot_last_occurrence=slot_last_occurrence)

        if sum(sys_utt_tok_label) > 0:
            inform_label_dict[slot_type] = cur_ds_dict[slot_type]

        # Check on the fly if the system informs in current text instead of prior as in TripPy
        if get_sys_inform(response, slot_type):
            inform_slot_label_dict[slot_type] = 1

        sys_utt_tok_label = [0 for _ in sys_utt_tok_label] # Don't use token labels for sys utt
        sys_utt_tok_label_dict[slot_type] = sys_utt_tok_label
        usr_utt_tok_label_dict[slot_type] = usr_utt_tok_label
        class_type_dict[slot_type] = class_type

    return (sys_utt_tok, sys_utt_tok_label_dict,
            usr_utt_tok, usr_utt_tok_label_dict,
            inform_label_dict, inform_slot_label_dict,
            referral_label_dict, cur_ds_dict, class_type_dict)
    


def create_examples(batch_dialogues_utterances_contexts_info, 
                    batch_dialogues_utterances_responses_info,
                    slot_list,
                    nlg_manager=None,
                    append_history=True,
                    use_history_labels=True,
                    swap_utterances=True,
                    debug=False):
    """
    Create TripPy input examples.

    Args:
        batch_dialogues_utterances_contexts_info: Utterance information
        batch_dialogues_utterances_responses_info: Response information
        slot_list: List of all possible slots; Defined in the config
        nlg_manager: NLG Manager for retrieving action labels
        use_history_labels: Whether to use history labels; True for TripPy advanced
        append_history: Whether to append the dialogue history; True for TripPy advanced
        swap_utterances: true in TripPy advanced, but for dstc2 the system starts not the user; Use this if the user starts

        delexicalize_sys_utts[REMOVED]: Whether to mask slot value utterances from the system - Used in the advanced TripPy, however, 
          we have no slot value information in DSTC2, so cannot be used - Refer to TripPy to readd
    Returns:
        examples: List of DSTExample instances
    """

    examples = []
    for dial_id, (contexts, responses) in enumerate(zip(batch_dialogues_utterances_contexts_info, batch_dialogues_utterances_responses_info)):
        # Presets
        prev_ds = {} # dict instead of list since we use the sv_dict formated ver
        hst = []
        prev_hst_lbl_dict = {slot: [] for slot in slot_list}
        prev_ds_lbl_dict = {slot: 'none' for slot in slot_list}
        response_saved = None

        for turn_id, (context, response) in enumerate(zip(contexts, responses)):
            guid = '%s-%s' % (dial_id, str(turn_id))

            # Not in original TripPy; Get the action label if training time, i.e. we have response data
            action_label = nlg_manager.get_action_id(response["act"]) if (response is not None) and (response["act"] is not None) else 0
            prev_action_label = nlg_manager.get_action_id(context["prev_resp_act"]) if ("prev_resp_act" in context) and (context["prev_resp_act"] is not None) else 0

            # Move the responses one backwards, because the first input should be user only with the response being None
            # The final utterance by the system is not needed in the text, except for the action label (action labels are not moved back)
            response_saved, response = response, response_saved


            ds_lbl_dict = prev_ds_lbl_dict.copy()
            hst_lbl_dict = prev_hst_lbl_dict.copy()

            cur_ds = get_dialogue_state_sv_dict(context) # Create DS here instead of in get_turn_label
            cur_ds = {**prev_ds, **cur_ds} # Merge with prev_ds, giving preference to cur_ds

            (text_a,
            text_a_label,
            text_b,
            text_b_label,
            inform_label,
            inform_slot_label,
            referral_label,
            cur_ds_dict,
            class_label) = get_turn_label(context, # context & response instd of turn
                                          response,
                                          cur_ds, # Add cur_ds, which is normally in turn["dialogue_state"]
                                          prev_ds,
                                          slot_list,
                                          dial_id,
                                          turn_id,
                                          slot_last_occurrence=True)
            

            # Set to true by default, since in DP the system starts 
            if swap_utterances:
                txt_a = text_b
                txt_b = text_a
                txt_a_lbl = text_b_label
                txt_b_lbl = text_a_label
            else:
                txt_a = text_a
                txt_b = text_b
                txt_a_lbl = text_a_label
                txt_b_lbl = text_b_label


            value_dict = {}
            for slot in slot_list:
                if slot in cur_ds_dict:
                    value_dict[slot] = cur_ds_dict[slot]
                else:
                    value_dict[slot] = 'none'
                if class_label[slot] != 'none':
                    ds_lbl_dict[slot] = class_label[slot]
                if append_history:
                    if use_history_labels:
                        hst_lbl_dict[slot] = txt_a_lbl[slot] + txt_b_lbl[slot] + hst_lbl_dict[slot]
                    else:
                        hst_lbl_dict[slot] = [0 for _ in txt_a_lbl[slot] + txt_b_lbl[slot] + hst_lbl_dict[slot]]

            examples.append(DSTExample(
                guid=guid,
                text_a=txt_a,
                text_b=txt_b,
                history=hst,
                text_a_label=txt_a_lbl,
                text_b_label=txt_b_lbl,
                history_label=prev_hst_lbl_dict,
                values=value_dict,
                inform_label=inform_label,
                inform_slot_label=inform_slot_label,
                refer_label=referral_label,
                diag_state=prev_ds_lbl_dict,
                class_label=class_label,
                action_label=action_label,
                prev_action_label=prev_action_label)) # Not in original TripPy; The action idx the model is supposed to predict

            prev_ds = cur_ds # use already transformed cur_ds instead of turn['dialogue_state']
            prev_ds_lbl_dict = ds_lbl_dict.copy()
            prev_hst_lbl_dict = hst_lbl_dict.copy()

            if append_history:
                hst = txt_a + txt_b + hst

            if (debug) and (dial_id == 0) and (turn_id < 2):
                logger.info(f"Example - Turn {turn_id}:")
                logger.info(f"Text A: {txt_a}")
                logger.info(f"Text B: {txt_b}")
                logger.info(f"Action Label: {action_label}")

    return examples

### Transform into final model input ###


### Because we have managed to turn DSTC2 into a valid DSTExample (a class used in TripPy) above, we can mostly copy the remaining transformations, i.e. below & possibly simplify
### Will simplify the below to 1/4 the length ###
### A lot of extra lines because of RoBERTa compatibility (I think we won't need that for DP)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_ids_unmasked,
                 input_mask,
                 segment_ids,
                 start_pos=None,
                 end_pos=None,
                 values=None,
                 inform=None,
                 inform_slot=None,
                 refer_id=None,
                 diag_state=None,
                 class_label_id=None,
                 guid="NONE"):
        self.guid = guid
        self.input_ids = input_ids
        self.input_ids_unmasked = input_ids_unmasked
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.values = values
        self.inform = inform
        self.inform_slot = inform_slot
        self.refer_id = refer_id
        self.diag_state = diag_state
        self.class_label_id = class_label_id


def convert_examples_to_features(examples, slot_list, class_types, tokenizer, max_seq_length, slot_value_dropout=0.0, debug=False):
    """Loads a data file into a list of `InputBatch`s."""

    # BERT Model Specs
    model_specs = {'MODEL_TYPE': 'bert',
                'CLS_TOKEN': '[CLS]',
                'UNK_TOKEN': '[UNK]',
                'SEP_TOKEN': '[SEP]',
                'TOKEN_CORRECTION': 4}

    def _tokenize_text_and_label(text, text_label_dict, slot, tokenizer, model_specs, slot_value_dropout):
        joint_text_label = [0 for _ in text_label_dict[slot]] # joint all slots' label
        for slot_text_label in text_label_dict.values():
            for idx, label in enumerate(slot_text_label):
                if label == 1:
                    joint_text_label[idx] = 1

        text_label = text_label_dict[slot]
        tokens = []
        tokens_unmasked = []
        token_labels = []
        for token, token_label, joint_label in zip(text, text_label, joint_text_label):
            token = convert_to_unicode(token)
            sub_tokens = tokenizer.tokenize(token) # Most time intensive step
            tokens_unmasked.extend(sub_tokens)
            if slot_value_dropout == 0.0 or joint_label == 0:
                tokens.extend(sub_tokens)
            else:
                rn_list = np.random.random_sample((len(sub_tokens),))
                for rn, sub_token in zip(rn_list, sub_tokens):
                    if rn > slot_value_dropout:
                        tokens.append(sub_token)
                    else:
                        tokens.append(model_specs['UNK_TOKEN'])
            token_labels.extend([token_label for _ in sub_tokens])
        assert len(tokens) == len(token_labels)
        assert len(tokens_unmasked) == len(token_labels)
        return tokens, tokens_unmasked, token_labels

    def _truncate_seq_pair(tokens_a, tokens_b, history, max_length):
        """Truncates a sequence pair in place to the maximum length.
        Copied from bert/run_classifier.py
        """
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(history)
            if total_length <= max_length:
                break
            if len(history) > 0:
                history.pop()
            elif len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    
    def _truncate_length_and_warn(tokens_a, tokens_b, history, max_seq_length, model_specs, guid):
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP], [SEP] with "- 4" (BERT)
        # Account for <s>, </s></s>, </s></s>, </s> with "- 6" (RoBERTa)
        if len(tokens_a) + len(tokens_b) + len(history) > max_seq_length - model_specs['TOKEN_CORRECTION']:
            if debug:
                logger.info("Truncate Example %s. Total len=%d." % (guid, len(tokens_a) + len(tokens_b) + len(history)))
                logger.info("Truncated Example History: %s" % history)
            input_text_too_long = True
        else:
            input_text_too_long = False
        _truncate_seq_pair(tokens_a, tokens_b, history, max_seq_length - model_specs['TOKEN_CORRECTION'])
        return input_text_too_long

    def _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs):
        token_label_ids = []
        token_label_ids.append(0) # [CLS]/<s>
        for token_label in token_labels_a:
            token_label_ids.append(token_label)
        token_label_ids.append(0) # [SEP]/</s></s>
        for token_label in token_labels_b:
            token_label_ids.append(token_label)
        token_label_ids.append(0) # [SEP]/</s></s>
        for token_label in token_labels_history:
            token_label_ids.append(token_label)
        token_label_ids.append(0) # [SEP]/</s>
        while len(token_label_ids) < max_seq_length:
            token_label_ids.append(0) # padding
        assert len(token_label_ids) == max_seq_length
        return token_label_ids

    def _get_start_end_pos(class_type, token_label_ids, max_seq_length):
        if class_type == 'copy_value' and 1 not in token_label_ids:
            #logger.warn("copy_value label, but token_label not detected. Setting label to 'none'.")
            class_type = 'none'
        start_pos = 0
        end_pos = 0
        if 1 in token_label_ids:
            start_pos = token_label_ids.index(1)
            # Parsing is supposed to find only first location of wanted value
            if 0 not in token_label_ids[start_pos:]:
                end_pos = len(token_label_ids[start_pos:]) + start_pos - 1
            else:
                end_pos = token_label_ids[start_pos:].index(0) + start_pos - 1
            for i in range(max_seq_length):
                if i >= start_pos and i <= end_pos:
                    assert token_label_ids[i] == 1
        return class_type, start_pos, end_pos

    def _get_transformer_input(tokens_a, tokens_b, history, max_seq_length, tokenizer, model_specs):
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append(model_specs['CLS_TOKEN'])
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(1)
        for token in history:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return tokens, input_ids, input_mask, segment_ids
    
    total_cnt = 0
    too_long_cnt = 0

    refer_list = ['none'] + slot_list

    features = []
    # Convert single example
    for (example_index, example) in enumerate(examples):
        if (example_index % 10 == 0) and (debug == True):
            logger.info("Writing example %d of %d" % (example_index, len(examples)))

        total_cnt += 1
        
        value_dict = {}
        inform_dict = {}
        inform_slot_dict = {}
        refer_id_dict = {}
        diag_state_dict = {}
        class_label_id_dict = {}
        start_pos_dict = {}
        end_pos_dict = {}
        for slot in slot_list:
            tokens_a, tokens_a_unmasked, token_labels_a = _tokenize_text_and_label(
                example.text_a, example.text_a_label, slot, tokenizer, model_specs, slot_value_dropout)
            tokens_b, tokens_b_unmasked, token_labels_b = _tokenize_text_and_label(
                example.text_b, example.text_b_label, slot, tokenizer, model_specs, slot_value_dropout)
            tokens_history, tokens_history_unmasked, token_labels_history = _tokenize_text_and_label(
                example.history, example.history_label, slot, tokenizer, model_specs, slot_value_dropout)

            input_text_too_long = _truncate_length_and_warn(
                tokens_a, tokens_b, tokens_history, max_seq_length, model_specs, example.guid)

            if input_text_too_long:
                if debug == True:
                    if len(token_labels_a) > len(tokens_a):
                        logger.info('    tokens_a truncated labels: %s' % str(token_labels_a[len(tokens_a):]))
                    if len(token_labels_b) > len(tokens_b):
                        logger.info('    tokens_b truncated labels: %s' % str(token_labels_b[len(tokens_b):]))
                    if len(token_labels_history) > len(tokens_history):
                        logger.info('    tokens_history truncated labels: %s' % str(token_labels_history[len(tokens_history):]))

                token_labels_a = token_labels_a[:len(tokens_a)]
                token_labels_b = token_labels_b[:len(tokens_b)]
                token_labels_history = token_labels_history[:len(tokens_history)]
                tokens_a_unmasked = tokens_a_unmasked[:len(tokens_a)]
                tokens_b_unmasked = tokens_b_unmasked[:len(tokens_b)]
                tokens_history_unmasked = tokens_history_unmasked[:len(tokens_history)]

            assert len(token_labels_a) == len(tokens_a)
            assert len(token_labels_b) == len(tokens_b)
            assert len(token_labels_history) == len(tokens_history)
            assert len(token_labels_a) == len(tokens_a_unmasked)
            assert len(token_labels_b) == len(tokens_b_unmasked)
            assert len(token_labels_history) == len(tokens_history_unmasked)
            token_label_ids = _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs)

            value_dict[slot] = example.values[slot]
            inform_dict[slot] = example.inform_label[slot]

            class_label_mod, start_pos_dict[slot], end_pos_dict[slot] = _get_start_end_pos(
                example.class_label[slot], token_label_ids, max_seq_length)
            if class_label_mod != example.class_label[slot]:
                example.class_label[slot] = class_label_mod
            inform_slot_dict[slot] = example.inform_slot_label[slot]
            refer_id_dict[slot] = refer_list.index(example.refer_label[slot])
            diag_state_dict[slot] = class_types.index(example.diag_state[slot])
            class_label_id_dict[slot] = class_types.index(example.class_label[slot])

        if input_text_too_long:
            too_long_cnt += 1
            
        tokens, input_ids, input_mask, segment_ids = _get_transformer_input(tokens_a,
                                                                            tokens_b,
                                                                            tokens_history,
                                                                            max_seq_length,
                                                                            tokenizer,
                                                                            model_specs)
        if slot_value_dropout > 0.0:
            _, input_ids_unmasked, _, _ = _get_transformer_input(tokens_a_unmasked,
                                                                 tokens_b_unmasked,
                                                                 tokens_history_unmasked,
                                                                 max_seq_length,
                                                                 tokenizer,
                                                                 model_specs)
        else:
            input_ids_unmasked = input_ids

        assert(len(input_ids) == len(input_ids_unmasked))

        if debug == True:
            logger.info("*** TripPy Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("start_pos: %s" % str(start_pos_dict))
            logger.info("end_pos: %s" % str(end_pos_dict))
            logger.info("values: %s" % str(value_dict))
            logger.info("inform: %s" % str(inform_dict))
            logger.info("inform_slot: %s" % str(inform_slot_dict))
            logger.info("refer_id: %s" % str(refer_id_dict))
            logger.info("diag_state: %s" % str(diag_state_dict))
            logger.info("class_label_id: %s" % str(class_label_id_dict))

        features.append(
            InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                input_ids_unmasked=input_ids_unmasked,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_pos=start_pos_dict,
                end_pos=end_pos_dict,
                values=value_dict,
                inform=inform_dict,
                inform_slot=inform_slot_dict,
                refer_id=refer_id_dict,
                diag_state=diag_state_dict,
                class_label_id=class_label_id_dict))

    if debug == True:
        logger.info("========== %d out of %d examples have text too long" % (too_long_cnt, total_cnt))

    return features



def get_turn(batch, index=-1):
    """
    Seeks turn from trippy input batch; None is used to keep [1, seq_len] shape
    If index = -1, gets the last turn
    """
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = {k: v[None, index] for k, v in value.items()}
        elif isinstance(value, list):
            result[key] = [v[None, index] for v in value]
        else:
            result[key] = value[None, index]
    return result

def batch_to_device(batch, device):
    """
    Moves items in batch to correct device
    """
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = {k: v.to(device) for k, v in value.items()}
        elif isinstance(value, list):
            result[key] = [v.to(device) for v in value]
        else:
            result[key] = value.to(device)
    return result


def prepare_trippy_data(batch_dialogues_utterances_contexts_info: List[List[dict]],
                        batch_dialogues_utterances_responses_info: List[List[dict]],
                        tokenizer,
                        slot_list,
                        class_types,
                        nlg_manager=None,
                        max_seq_length=180,
                        debug=False) -> dict:
    """
    Parse the passed DSTC2 dialogue information to BertForDST input. 

    Args:
        batch_dialogues_utterances_contexts_info: the dictionary containing
                                                    the dialogue utterances training information
        batch_dialogues_utterances_responses_info: the dictionary containing
                                                    the dialogue utterances responses training information
        tokenizer: BertTokenizer used to tokenize inputs
        slot_list: Slot Names to be filled
        class_types: Generally [copy_value, inform, none, dontcare] for TripPy
        nlg_manager: NLGManager necessary for getting action labels during example creation
        max_seq_length: Maximum length of TripPy input (incl history) - 180 is default for TripPy Advanced

    Returns:
        inputs: Dict of BertForDST Inputs
        features: ???

    """
    examples = create_examples(batch_dialogues_utterances_contexts_info,
                               batch_dialogues_utterances_responses_info,
                               slot_list=slot_list,
                               nlg_manager=nlg_manager,
                               debug=debug)


    

    features = convert_examples_to_features(examples, 
                                            slot_list, 
                                            class_types=class_types, 
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            debug=debug)

    # Convert to Tensors and return data
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    f_start_pos = [f.start_pos for f in features]
    f_end_pos = [f.end_pos for f in features]
    f_inform_slot_ids = [f.inform_slot for f in features]
    f_refer_ids = [f.refer_id for f in features]
    f_diag_state = [f.diag_state for f in features]
    f_class_label_ids = [f.class_label_id for f in features]
    all_start_positions = {}
    all_end_positions = {}
    all_inform_slot_ids = {}
    all_refer_ids = {}
    all_diag_state = {}
    all_class_label_ids = {}
    for s in slot_list:
        all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
        all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos], dtype=torch.long)
        all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
        all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
        all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
        all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)

    # Not in original TripPy; Add Action labels
    all_action_labels = torch.tensor([e.action_label for e in examples], dtype=torch.long)
    all_prev_action_labels =  torch.nn.functional.one_hot(torch.tensor([e.action_label for e in examples], dtype=torch.long), num_classes=nlg_manager.num_of_known_actions())


    # Possibly have this in main trippy cuz diag state needs to be updated for eval runs
    inputs = {'input_ids':       all_input_ids,
              'input_mask':      all_input_mask, 
              'segment_ids':     all_segment_ids,
              'start_pos':       all_start_positions,
              'end_pos':         all_end_positions,
              'inform_slot_id':  all_inform_slot_ids,
              'refer_id':        all_refer_ids,
              'diag_state':      all_diag_state,
              'class_label_id':  all_class_label_ids, 
              "action_label":    all_action_labels,
              "prev_action_label":    all_prev_action_labels}


    return inputs, features



