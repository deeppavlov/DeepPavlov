# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoConfig

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('torch_transformers_hallucination_detector')
class TorchTransformersHallucinationDetector(TorchModel):
    """Transformer-based model on PyTorch for text tagging. It predicts a label for every token (not subtoken)
    """

    def __init__(self,
                 pretrained_bert: str,
                 bert_config_file: Optional[str] = None,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 **kwargs) -> None:
        if pretrained_bert:
            model = AutoModelForTokenClassification.from_pretrained(pretrained_bert, num_labels=2)
        elif bert_config_file and Path(bert_config_file).is_file():
            bert_config = AutoConfig.from_json_file(str(expand_path(bert_config_file)))
            if attention_probs_keep_prob is not None:
                bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
            if hidden_keep_prob is not None:
                bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob
            model = AutoModelForTokenClassification(config=bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        super().__init__(model, **kwargs)

    def train_on_batch(self,
                       input_ids: Union[List[List[int]], np.ndarray],
                       input_masks: Union[List[List[int]], np.ndarray],
                       y: List[List[int]],
                       *args, **kwargs) -> Dict[str, float]:
        """

        Args:
            input_ids: batch of indices of subwords
            input_masks: batch of masks which determine what should be attended
            args: arguments passed  to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.
            kwargs: keyword arguments passed to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.

        Returns:
            dict with fields 'loss', 'head_learning_rate', and 'bert_learning_rate'
        """
        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_masks = torch.from_numpy(input_masks).to(self.device)
        b_labels = torch.from_numpy(np.array(y)).to(torch.int64).to(self.device)
        self.optimizer.zero_grad()

        loss = self.model(input_ids=b_input_ids,
                          attention_mask=b_input_masks,
                          labels=b_labels).loss
        self._make_step(loss)

        return {'loss': loss.item()}

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray]) -> Tuple[List[List[int]], List[np.ndarray]]:
        """ Predicts tag indices for a given subword tokens batch

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Label indices or class probabilities for each token (not subtoken)

        """
        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_masks = torch.from_numpy(input_masks).to(self.device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = self.model(b_input_ids, attention_mask=b_input_masks)
            logits = outputs.logits
        probas = torch.nn.functional.softmax(logits, dim=-1)
        probas = probas.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        pred = np.argmax(logits, axis=-1)
        return pred, probas

    def load(self, fname=None):
        super().load(fname)
        # TODO: test
        if self.load_path:
            weights_path = Path(f"{self.load_path}").resolve()
            weights_path = weights_path.with_suffix(".pth.tar")
            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=self.device)
            else:
                log.warning(f"Init from scratch. Load path {weights_path} does not exist.")

    def save(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        super().save(fname, *args, **kwargs)
        # TODO: test
        if fname is None:
            fname = self.save_path
        weights_path = Path(f"{fname}").resolve()
        weights_path = weights_path.with_suffix(".pth.tar")