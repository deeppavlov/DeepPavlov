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

import re
from logging import getLogger
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from numpy.lib.twodim_base import diag

import torch
from overrides import overrides
from transformers.modeling_bert import BertConfig
from transformers import BertTokenizerFast

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.go_bot.nlg.nlg_manager import NLGManagerInterface
from deeppavlov.models.go_bot.policy.dto.policy_prediction import PolicyPrediction
from deeppavlov.models.go_bot.trippy_bert_for_dst import BertForDST
from deeppavlov.models.go_bot.trippy_preprocssing import prepare_trippy_data, get_turn, batch_to_device


# EXP
from transformers import (AdamW, get_linear_schedule_with_warmup)

logger = getLogger(__name__)


@register('trippy')
class TripPy(TorchModel):
    """
    Go-bot architecture based on https://arxiv.org/abs/2005.02877.

    Parameters:
        save_path: Where to save the model
        class_types: TripPy Class types - Predefined to most commonly used; Add True&False if slots which can take on those values
        pretrained_bert: bert-base-uncased or full path to pretrained model
        bert_config: Can be path to a file in case different from bert-base-uncased config
        optimizer_parameters: dictionary with optimizer's parameters, e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        clip_norm: Clip gradients by norm
        max_seq_length: Max sequence length of an entire dialog. Defaults to TripPy 180 default. Too long examples will be logged.
        class_loss_ratio: The ratio applied on class loss in total loss calculation.
            Should be a value in [0.0, 1.0].
            The ratio applied on token loss is (1-class_loss_ratio)/2.
            The ratio applied on refer loss is (1-class_loss_ratio)/2.
        token_loss_for_nonpointable: Whether the token loss for classes other than copy_value contribute towards total loss.
        refer_loss_for_nonpointable: Whether the refer loss for classes other than refer contribute towards total loss.
        class_aux_feats_inform: Whether or not to use the identity of informed slots as auxiliary features for class prediction.
        class_aux_feats_ds: Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.
        debug: Turn on debug mode to get logging information on input examples & co
    """
    def __init__(self,
                 nlg_manager: NLGManagerInterface,
                 save_path: str,
                 slot_names: List,
                 class_types: List = ["none", "dontcare", "copy_value", "inform"],
                 pretrained_bert: str = "bert-base-uncased",
                 bert_config: str = "bert-base-uncased",
                 optimizer_parameters: dict = {"lr": 1e-4, "eps": 1e-6},
                 clip_norm: float = 1.0,
                 max_seq_length: int = 180,
                 dropout_rate: float = 0.3,
                 heads_dropout: float = 0.0,
                 class_loss_ratio: float = 0.8,
                 token_loss_for_nonpointable: bool = False,
                 refer_loss_for_nonpointable: bool = False,
                 class_aux_feats_inform: bool = True,
                 class_aux_feats_ds: bool = True,
                 database: Component = None,
                 debug: bool = False,
                 **kwargs) -> None:

        self.nlg_manager = nlg_manager
        self.save_path = save_path
        self.max_seq_length = max_seq_length
        self.slot_names = slot_names
        self.class_types = class_types
        self.debug = debug

        # BertForDST Configuration
        self.pretrained_bert = pretrained_bert
        self.config = BertConfig.from_pretrained(bert_config)
        self.config.dst_dropout_rate = dropout_rate
        self.config.dst_heads_dropout_rate = heads_dropout
        self.config.dst_class_loss_ratio = class_loss_ratio
        self.config.dst_token_loss_for_nonpointable = token_loss_for_nonpointable
        self.config.dst_refer_loss_for_nonpointable = refer_loss_for_nonpointable
        self.config.dst_class_aux_feats_inform = class_aux_feats_inform
        self.config.dst_class_aux_feats_ds = class_aux_feats_ds
        self.config.dst_slot_list = slot_names
        self.config.dst_class_types = class_types
        self.config.dst_class_labels = len(class_types)

        self.config.num_actions = nlg_manager.num_of_known_actions()

        # Parameters for user interaction
        self.batch_dialogues_utterances_contexts_info = [[]]
        # We always have one more user response than system response at inference
        self.batch_dialogues_utterances_responses_info = [[None]]

        self.ds = None
        self.ds_logits = None

        self.database = database
        self.clip_norm = clip_norm
        super().__init__(save_path=save_path,  
                        optimizer_parameters=optimizer_parameters,
                        **kwargs)

    @overrides
    def load(self, fname=None):
        """
        Loads BERTForDST. Note that it only supports bert-X huggingface weights. (RoBERTa & co are not supported.)
        """
        # Load model from huggingface // from path
        self.model = BertForDST.from_pretrained(
            self.pretrained_bert, config=self.config)
        # Tokenizer is always the same for bert-base / bert-large
        # We also always use uncased, as TripPy always lowercases all data in its input
        # If you think cases are super important for your data, feel free to change the below & remove the .lower() ops in preprocessing
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        self.model.to(self.device)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
               self.model.parameters(), **self.optimizer_parameters)

        self.optimizer = AdamW(self.model.parameters(), **self.optimizer_parameters)
        #t_total: batches / batch_size  *  epochs
        t_total = 550#900 // 4 * 2 # Rough estimate
        num_warmup_steps = int(t_total * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                        num_warmup_steps=num_warmup_steps, 
                                                        num_training_steps=t_total)

        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

    def __call__(self,
                 batch: Union[List[List[dict]], List[str]],
                 user_ids: Optional[List] = None) -> List:
        """
        Model invocation.

        Args:
            batch: batch of dialogue data or list of strings
            user_ids: Id that identifies the user # Check bocks

        Returns:
            results: list of model answers
        """
        # Turns off dropout
        self.model.eval()

        if not(isinstance(batch[0], list)):
            # User inference - Just one dialogue
            batch = [
                [{"text": text, "intents": [{"act": None, "slots": None}]} for text in batch]]
        else:
            self.reset()

        dialogue_results = []
        for diag_id, dialogue in enumerate(batch):

            turn_results = []
            for turn_id, turn in enumerate(dialogue):
                # Reset dialogue state if no dialogue state yet or the dialogue is empty (i.e. its a new dialogue)
                if (self.ds_logits is None) or (diag_id >= len(self.batch_dialogues_utterances_contexts_info)):
                    self.reset()
                    diag_id = 0

                # Append context to the dialogue
                self.batch_dialogues_utterances_contexts_info[diag_id].append(turn)

                # Update Database
                self.update_ground_truth_db_result_from_context(turn)

                # Preprocess inputs
                batch, features = prepare_trippy_data(self.batch_dialogues_utterances_contexts_info,
                                                      self.batch_dialogues_utterances_responses_info,
                                                      self.tokenizer,
                                                      self.slot_names,
                                                      self.class_types,
                                                      self.nlg_manager,
                                                      max_seq_length=self.max_seq_length,
                                                      debug=self.debug)

                # Take only the last turn - as we already know the previous ones; We need to feed them one by one to update the ds
                last_turn = get_turn(batch, index=-1)

                # Only take them from the last turn
                input_ids_unmasked = [features[-1].input_ids_unmasked]
                inform = [features[-1].inform]

                # Update data-held dialogue state based on new logits
                last_turn["diag_state"] = self.ds_logits

                # Move to correct device
                last_turn = batch_to_device(last_turn, self.device)

                # Run the turn through the model
                with torch.no_grad():
                    outputs = self.model(**last_turn)

                # Update dialogue state logits
                for slot in self.model.slot_list:
                    updates = outputs[2][slot].max(1)[1]
                    for i, u in enumerate(updates):
                        if u != 0:
                            self.ds_logits[slot][i] = u

                # Update self.ds (dialogue state) slotfilled values based on logits
                self.update_ds(outputs[2],
                               outputs[3],
                               outputs[4],
                               outputs[5],
                               input_ids_unmasked,
                               inform)

                # Wrap predicted action (outputs[6]) into a PolicyPrediction

                #print("POLICY")
                #print(outputs[6].cpu())
                #print(torch.nn.functional.one_hot(last_turn["action_label"], num_classes=46))

                policy_prediction = PolicyPrediction(
                    outputs[6].cpu(), None, None, None)

                # Ground truth Policy for testing
                #policy_prediction = PolicyPrediction(
                #    torch.nn.functional.one_hot(last_turn["action_label"], num_classes=46).cpu(), None, None, None
                #)

                # Fill DS with Database results if there are any
                self.fill_current_state_with_db_results()

                # NLG based on predicted action & dialogue state
                #print("DS FOR NLG", self.ds)
                response = self.nlg_manager.decode_response(None,
                                                            policy_prediction,
                                                            self.ds)


                #print("DS RESULTS:", self.ds)
                #print("RESPONSE", response)
                #print(last_turn["action_label"])
                #print("RESPONSE:", response)

                # Add system response to responses for possible next round
                self.batch_dialogues_utterances_responses_info[diag_id].insert(
                    -1, {"text": response, "act": None})

                turn_results.append(response)

            dialogue_results.append(turn_results)
        
        #print("LAST DS", self.ds)

        # Return NLG generated responses
        return dialogue_results

    def update_ds(self,
                  per_slot_class_logits,
                  per_slot_start_logits,
                  per_slot_end_logits,
                  per_slot_refer_logits,
                  input_ids_unmasked,
                  inform):
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
        """
        # We set the index to 0, since we only look at the last turn
        # This function can be modified to look at multiple turns by iterating over them
        i = 0

        if self.ds is None:
            self.ds = {slot: 'none' for slot in self.model.slot_list}

        for slot in self.model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            end_logits = per_slot_end_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if class_prediction == self.model.class_types.index('dontcare'):
                self.ds[slot] = 'dontcare'
            elif class_prediction == self.model.class_types.index('copy_value'):
                input_tokens = self.tokenizer.convert_ids_to_tokens(
                    input_ids_unmasked[i])
                self.ds[slot] = ' '.join(
                    input_tokens[start_prediction:end_prediction + 1])
                self.ds[slot] = re.sub("(^| )##", "", self.ds[slot])
            elif 'true' in self.model.class_types and class_prediction == self.model.class_types.index('true'):
                self.ds[slot] = 'true'
            elif 'false' in self.model.class_types and class_prediction == self.model.class_types.index('false'):
                self.ds[slot] = 'false'
            elif class_prediction == self.model.class_types.index('inform'):
                self.ds[slot] = inform[i][slot]

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in self.model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in self.model.class_types and class_prediction == self.model.class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                self.ds[slot] = self.ds[self.model.slot_list[refer_prediction - 1]]

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

    def update_ground_truth_db_result_from_context(self, context: Dict[str, Any]):
        self.current_db_result = context.get('db_result', None)
        self._update_db_result()

    def fill_current_state_with_db_results(self) -> dict:
        if self.db_result:
            for k, v in self.db_result.items():
                self.ds[k] = str(v)
            #print("FILLED DS", self.ds)

    def train_on_batch(self,
                       batch_dialogues_utterances_features: List[List[dict]],
                       batch_dialogues_utterances_targets: List[List[dict]]) -> dict:
        """
        Train model on given batch.

        Args:
            batch_dialogues_utterances_features:
            batch_dialogues_utterances_targets: 

        Returns:
            dict with loss value
        """
        # Turns on dropout
        self.model.train()
        # Zeroes grads
        self.model.zero_grad()
        batch, features = prepare_trippy_data(batch_dialogues_utterances_features,
                                              batch_dialogues_utterances_targets,
                                              self.tokenizer,
                                              self.slot_names,
                                              self.class_types,
                                              self.nlg_manager,
                                              self.max_seq_length,
                                              debug=self.debug)
        # Move to correct device
        batch = batch_to_device(batch, self.device)

        # Feed through model
        outputs = self.model(**batch)

        # Backpropagation
        loss = outputs[0]
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss.cpu().item()}

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

    def save(self, *args, **kwargs) -> None:
        """
        Save the model.
        """
        self.model.save_pretrained(self.save_path)
