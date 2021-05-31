# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from logging import getLogger
from typing import Dict, Any, List, Optional, Union, Tuple

from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel


### Temp Note: Planning to copy this largely from TripPy code ###
class BertForDST(BertPreTrainedModel):
    """
    The Dialogue State-Tracking BERT Model used by TripPy
    """
    def __init__(self, config):
        super(BertForDST, self).__init__(config)
 

        ### Temp Note: Some more param changes ###

        self.bert = BertModel(config)
    
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
                aux_task_def=None):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        ### Temp Note: Some post-processing ###

        return outputs

    

@register('trippy')
class TripPy(TorchModel):
    """
    Go-bot architecture based on https://arxiv.org/abs/2005.02877.
    """
    def __init__(self,
                pretrained_bert: str,) -> None:
        pass

    @overrides
    def load(self, fname=None):
        """
        Load BERTForDST here
        """
        pass

    def __call__(self, batch: Union[List[List[dict]], List[str]],
                 user_ids: Optional[List] = None) -> List:
        pass

        ### Temp Note: What data comes in here? & Is there a defined output ###

    def train_on_batch(self,
                       batch_dialogues_utterances_features: List[List[dict]],
                       batch_dialogues_utterances_targets: List[List[dict]]) -> dict:
        """
        Train model on given batch.

        Args:
            batch_dialogues_utterances_features:
            batch_dialogues_utterances_targets: 

        Returns:
            WIP
        """
        ### Temp Note: Preprocess data ###
        pass
