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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_bert import BertModel, BertPreTrainedModel

class BertForDST(BertPreTrainedModel):
    """
    BERT model used by TripPy. 

    This extends the basic bert model for dialogue state tracking.

    Args:
        config: Model-specific attributes & settings
    """
    def __init__(self, config):
        super(BertForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
        self.class_aux_feats_inform = config.dst_class_aux_feats_inform
        self.class_aux_feats_ds = config.dst_class_aux_feats_ds
        self.class_loss_ratio = config.dst_class_loss_ratio

        # Not in original TripPy model; Added for action prediction 
        self.num_actions = config.num_actions

        # Only use refer loss if refer class is present in dataset.
        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

        if self.class_aux_feats_inform:
            self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
        if self.class_aux_feats_ds:
            self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

        aux_dims = len(self.slot_list) * (self.class_aux_feats_inform + self.class_aux_feats_ds) # second term is 0, 1 or 2

        for slot in self.slot_list:
            self.add_module("class_" + slot, nn.Linear(config.hidden_size + aux_dims, self.class_labels))
            self.add_module("token_" + slot, nn.Linear(config.hidden_size, 2))
            self.add_module("refer_" + slot, nn.Linear(config.hidden_size + aux_dims, len(self.slot_list) + 1))

        # Head for aux task
        if hasattr(config, "aux_task_def"):
            self.add_module("aux_out_projection", nn.Linear(config.hidden_size, int(config.aux_task_def['n_class'])))

        # Not in original TripPy model; Add action prediction CLF Head
        self.add_module("action_prediction", nn.Linear(config.hidden_size + aux_dims, self.num_actions))
        self.add_module("action_softmax", nn.Softmax(dim=1))

        # Loss functions
        self.aux_loss_fct = CrossEntropyLoss()
        self.refer_loss_fct = CrossEntropyLoss(reduction='none')
        self.class_loss_fct = CrossEntropyLoss(reduction='none')
        self.action_loss_fct = CrossEntropyLoss(reduction="sum")

        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                start_pos=None,
                end_pos=None,
                inform_slot_id=None,
                refer_id=None,
                class_label_id=None,
                diag_state=None,
                aux_task_def=None,
                action_label=None):
        """
        Runs the model and outputs predictions and loss. 

        Args:
            input_ids: BERT tokenized input_ids
            input_mask: Attention mask of input_ids
            segment_ids: 1 / 0s to differentiate sentence parts
            position_ids: Token positions in input_ids
            head_mask: Mask to hide attention heads
            start_pos: Labels for slot starting positions
            end_pos: Labels for slot ending positions
            inform_slot_id: Labels for whether the system informs
            refer_id: Labels for whether a slot value is referred from another slot
            class_label_id: Label for the class type of the slot value, e.g. dontcare
            diag_state: Current dialogue state of all slots
            aux_task_def: If there is an auxiliary task, such as classification
            action_label: Action to predict
        Returns:
            outputs: Tuple of logits & losses  
        """
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        if aux_task_def is not None:
            if aux_task_def['task_type'] == "classification":
                aux_logits = getattr(self, 'aux_out_projection')(pooled_output)
                aux_logits = self.dropout_heads(aux_logits)
                aux_loss = self.aux_loss_fct(aux_logits, class_label_id)
                # add hidden states and attention if they are here
                return (aux_loss,) + outputs[2:]
            elif aux_task_def['task_type'] == "span":
                aux_logits = getattr(self, 'aux_out_projection')(sequence_output)
                aux_start_logits, aux_end_logits = aux_logits.split(1, dim=-1)
                aux_start_logits = self.dropout_heads(aux_start_logits)
                aux_end_logits = self.dropout_heads(aux_end_logits)
                aux_start_logits = aux_start_logits.squeeze(-1)
                aux_end_logits = aux_end_logits.squeeze(-1)

                # If we are on multi-GPU, split add a dimension
                if len(start_pos.size()) > 1:
                    start_pos = start_pos.squeeze(-1)
                if len(end_pos.size()) > 1:
                    end_pos = end_pos.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = aux_start_logits.size(1) # This is a single index
                start_pos.clamp_(0, ignored_index)
                end_pos.clamp_(0, ignored_index)
            
                aux_token_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                aux_start_loss = aux_token_loss_fct(torch.cat((aux_start_logits, aux_end_logits), 1), start_pos)
                aux_end_loss = aux_token_loss_fct(torch.cat((aux_end_logits, aux_start_logits), 1), end_pos)
                aux_loss = (aux_start_loss + aux_end_loss) / 2.0
                return (aux_loss,) + outputs[2:]
            else:
                raise Exception("Unknown task_type")

        if inform_slot_id is not None:
            inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()
        if diag_state is not None:
            diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)
        
        total_loss = 0
        per_slot_per_example_loss = {}
        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_refer_logits = {}
        for slot in self.slot_list:
            if self.class_aux_feats_inform and self.class_aux_feats_ds:
                pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
            elif self.class_aux_feats_inform:
                pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
            elif self.class_aux_feats_ds:
                pooled_output_aux = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
            else:
                pooled_output_aux = pooled_output
            class_logits = self.dropout_heads(getattr(self, 'class_' + slot)(pooled_output_aux))

            token_logits = self.dropout_heads(getattr(self, 'token_' + slot)(sequence_output))
            start_logits, end_logits = token_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            refer_logits = self.dropout_heads(getattr(self, 'refer_' + slot)(pooled_output_aux))

            per_slot_class_logits[slot] = class_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits
            per_slot_refer_logits[slot] = refer_logits

            # If there are no labels, don't compute loss
            if class_label_id is not None and start_pos is not None and end_pos is not None and refer_id is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_pos[slot].size()) > 1:
                    start_pos[slot] = start_pos[slot].squeeze(-1)
                if len(end_pos[slot].size()) > 1:
                    end_pos[slot] = end_pos[slot].squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1) # This is a single index
                start_pos[slot].clamp_(0, ignored_index)
                end_pos[slot].clamp_(0, ignored_index)

                token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)

                start_loss = token_loss_fct(start_logits, start_pos[slot])
                end_loss = token_loss_fct(end_logits, end_pos[slot])
                token_loss = (start_loss + end_loss) / 2.0

                token_is_pointable = (start_pos[slot] > 0).float()
                if not self.token_loss_for_nonpointable:
                    token_loss *= token_is_pointable

                refer_loss = self.refer_loss_fct(refer_logits, refer_id[slot])
                token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
                if not self.refer_loss_for_nonpointable:
                    refer_loss *= token_is_referrable

                class_loss = self.class_loss_fct(class_logits, class_label_id[slot])

                if self.refer_index > -1:
                    per_example_loss = (self.class_loss_ratio) * class_loss + ((1 - self.class_loss_ratio) / 2) * token_loss + ((1 - self.class_loss_ratio) / 2) * refer_loss
                else:
                    per_example_loss = self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss

                total_loss += per_example_loss.sum()
                per_slot_per_example_loss[slot] = per_example_loss


        # Not in original TripPy; Predict action & add loss if training; At evaluation acton_label is set to 0
        if not self.slot_list:
            pooled_output_aux = pooled_output
        action_logits = getattr(self, 'action_prediction')(pooled_output_aux)

        if action_label is not None:
            action_loss = self.action_loss_fct(action_logits, action_label)

            # Increase the loss proportional to the amount of slots if present
            if self.slot_list:
                multiplier = len(self.slot_list)
            else:
                multiplier = 1

            total_loss += action_loss * multiplier
        else:
            action_loss = None

        # add hidden states and attention if they are here
        outputs = (total_loss,) + (per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits, action_logits, action_loss,) + outputs[2:]

        return outputs
