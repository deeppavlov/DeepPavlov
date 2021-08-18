### Implements a Dialogue State Tracker for TripPy ###

import re
from logging import getLogger
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

import torch

from deeppavlov.core.models.component import Component

logger = getLogger(__name__)

class TripPyDST:
    """
    TripPy Dialogue State Tracker.

    Args:
        slot_names: Names of all slots present in the data
        class_types: TripPy Class types - Predefined to most commonly used; Add True&False if slots which can take on those values
        database: Optional database which will be queried by make_api_call by default
        make_api_call: Optional function to replace default api calling
        fill_current_state_with_db_results: Optional function t replace default db result filling
    """
    def __init__(self,
                slot_names: List = [],
                class_types: List = ["none", "dontcare", "copy_value", "inform"],
                database: Component = None,
                make_api_call: Callable = None,
                fill_current_state_with_db_results: Callable = None,) -> None:

        self.slot_names = slot_names
        self.class_types = class_types

        # Parameters for user interaction
        self.batch_dialogues_utterances_contexts_info = [[]]
        # We always have one more user response than system response at inference
        self.batch_dialogues_utterances_responses_info = [[None]]

        self.ds = None
        self.ds_logits = None

        self.database = database

        # If the user as provided a make_api_call function
        # and a fill_current_state_with_db_results function use them, see trippy_extended_tutorial demo
        if make_api_call:
            # Override the functions for TripPy
            TripPyDST.make_api_call = make_api_call
            TripPyDST.fill_current_state_with_db_results = fill_current_state_with_db_results

    def update_ds(self,
                  per_slot_class_logits,
                  per_slot_start_logits,
                  per_slot_end_logits,
                  per_slot_refer_logits,
                  input_ids_unmasked,
                  inform,
                  tokenizer):
        """
        Updates slot-filled dialogue state based on model predictions.
        This function roughly corresponds to "predict_and_format" in the original TripPy code.

        Args:
            per_slot_class_logits: dict of class logits
            per_slot_start_logits: dict of start logits
            per_slot_end_logits: dict of end logits
            per_slot_refer_logits: dict of refer logits
            input_ids_unmasked: The unmasked input_ids from features to extract the preds
            inform: dict of inform logits
            tokenizer: BertTokenizer
        """
        # We set the index to 0, since we only look at the last turn
        # This function can be modified to look at multiple turns by iterating over them
        i = 0

        if self.ds is None:
            self.ds = {slot: 'none' for slot in self.slot_names}

        for slot in self.slot_names:
            class_logits = per_slot_class_logits[slot][i].cpu()
            start_logits = per_slot_start_logits[slot][i].cpu()
            end_logits = per_slot_end_logits[slot][i].cpu()
            refer_logits = per_slot_refer_logits[slot][i].cpu()

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            # DP / DSTC2 uses dontcare instead of none so we also replace none's wth dontcare
            # Just remove the 2nd part of the or statement to revert to TripPy standard
            if (class_prediction == self.class_types.index('dontcare')) or (class_prediction == self.class_types.index('none')):
                self.ds[slot] = 'dontcare'
            elif class_prediction == self.class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(
                    input_ids_unmasked[i])
                self.ds[slot] = ' '.join(
                    input_tokens[start_prediction:end_prediction + 1])
                self.ds[slot] = re.sub("(^| )##", "", self.ds[slot])
            elif 'true' in self.class_types and class_prediction == self.class_types.index('true'):
                self.ds[slot] = 'true'
            elif 'false' in self.class_types and class_prediction == self.class_types.index('false'):
                self.ds[slot] = 'false'
            elif class_prediction == self.class_types.index('inform'):
                self.ds[slot] = inform[i][slot]

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in self.slot_names:
            class_logits = per_slot_class_logits[slot][i].cpu()
            refer_logits = per_slot_refer_logits[slot][i].cpu()

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in self.class_types and class_prediction == self.class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                self.ds[slot] = self.ds[self.slot_names[refer_prediction - 1]]

    def make_api_call(self) -> None:
        db_results = []
        if self.database is not None:

            # filter slot keys with value equal to 'dontcare' as
            # there is no such value in database records
            # and remove unknown slot keys (for example, 'this' in dstc2 tracker)
            db_slots = {
                s: v for s, v in self.ds.items() if v != 'dontcare' and s in self.database.keys
            }

            db_results = self.database([db_slots])[0]

            # filter api results if there are more than one
            # TODO: add sufficient criteria for database results ranking
            if len(db_results) > 1:
                db_results = [r for r in db_results if r != self.db_result]
            else:
                print("Failed to get any results for: ", db_slots)
        else:
            logger.warning("No database specified.")

        logger.info(f"Made api_call with {self.ds.keys()}, got {len(db_results)} results.")
        self.current_db_result = {} if not db_results else db_results[0]
        self._update_db_result()

    def _update_db_result(self):
        if self.current_db_result is not None:
            self.db_result = self.current_db_result

    def update_ground_truth_db_result_from_context(self, context: Dict[str, Any]) -> None:
        self.current_db_result = context.get('db_result', None)
        self._update_db_result()

    def fill_current_state_with_db_results(self) -> None:
        if self.db_result:
            for k, v in self.db_result.items():
                self.ds[k] = str(v)

    def reset(self, user_id: Union[None, str, int] = None) -> None:
        """
        Reset dialogue state trackers.
        """
        self.ds_logits = {slot: torch.tensor([0]) for slot in self.slot_names}
        self.ds = None

        self.batch_dialogues_utterances_contexts_info = [[]]
        self.batch_dialogues_utterances_responses_info = [[None]]

        self.db_result = None
        self.current_db_result = None