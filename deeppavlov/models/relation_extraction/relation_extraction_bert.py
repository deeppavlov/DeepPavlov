from logging import getLogger
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.classifiers.re_bert import BertWithAdaThresholdLocContextPooling

log = getLogger(__name__)


@register('re_classifier')
class REBertModel(TorchModel):

    def __init__(
            self,
            n_classes: int,
            model_name: str,
            num_ner_tags: int,
            pretrained_bert: str = None,
            criterion: str = "CrossEntropyLoss",
            optimizer: str = "AdamW",
            optimizer_parameters: Dict = None,
            return_probas: bool = False,
            attention_probs_keep_prob: Optional[float] = None,
            hidden_keep_prob: Optional[float] = None,
            clip_norm: Optional[float] = None,
            threshold: Optional[float] = None,
            device: str = "cpu",
            **kwargs
    ) -> None:
        """
        Transformer-based model on PyTorch for relation extraction. It predicts a relation hold between entities in a
        text sample (one or several sentences).
        Args:
            n_classes: number of output classes
            model_name: the model which will be used for extracting the relations
            num_ner_tags: number of NER tags
            pretrained_bert: key title of pretrained Bert model (e.g. "bert-base-uncased")
            criterion: criterion name from `torch.nn`
            optimizer: optimizer name from `torch.optim`
            optimizer_parameters: dictionary with optimizer's parameters
            return_probas: set this to `True` if you need the probabilities instead of raw answers
            attention_probs_keep_prob: keep_prob for Bert self-attention layers
            hidden_keep_prob: keep_prob for Bert hidden layers
            clip_norm: clip gradients by norm
            threshold: manually set value for defining the positively predicted classes (instead of adaptive one)
            device: cpu/gpu device to use for training the model
        """
        self.n_classes = n_classes
        self.num_ner_tags = num_ner_tags
        self.pretrained_bert = pretrained_bert
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm
        self.threshold = threshold
        self.device = device

        if self.n_classes == 0:
            raise ConfigError("Please provide a valid number of classes.")

        if optimizer_parameters is None:
            optimizer_parameters = {"lr": 5e-5, "weight_decay": 0.01, "eps": 1e-6}

        super().__init__(
            n_classes=n_classes,
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            device=self.device,
            **kwargs)

    def train_on_batch(
            self, input_ids: List, attention_mask: List, entity_pos: List, entity_tags: List, labels: List
    ) -> float:
        """
        Trains the relation extraction BERT model on the given batch.
        Returns:
            dict with loss and learning rate values.
        """

        _input = {
            'input_ids': torch.LongTensor(input_ids).to(self.device),
            'attention_mask': torch.LongTensor(attention_mask).to(self.device),
            'entity_pos': entity_pos,
            'ner_tags': entity_tags,
            'labels': labels
        }

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()      # zero the parameter gradients

        hidden_states = self.model(**_input)
        loss = hidden_states[0]
        loss.backward()
        self.optimizer.step()

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def __call__(
            self, input_ids: List, attention_mask: List, entity_pos: List, entity_tags: List
    ) -> Union[List[int], List[np.ndarray]]:
        """ Get model predictions using features as input """

        self.model.eval()

        _input = {
            'input_ids': torch.LongTensor(input_ids).to(self.device),
            'attention_mask': torch.LongTensor(attention_mask).to(self.device),
            'entity_pos': entity_pos,
            'ner_tags': entity_tags
        }

        with torch.no_grad():
            indices, probas = self.model(**_input)

        if self.return_probas:
            pred = probas.cpu().numpy()
            pred[np.isnan(pred)] = 0
            pred_without_no_rel = []        # eliminate no_relation predictions
            for elem in pred:
                elem[0] = 0.0
                pred_without_no_rel.append(elem)
            new_pred = np.argmax(pred_without_no_rel, axis=1)
            one_hot = [[0.0] * self.n_classes] * len(new_pred)
            for i in range(len(new_pred)):
                one_hot[i][new_pred[i]] = 1.0
            pred = np.array(one_hot)
        else:
            pred = indices.cpu().numpy()
            pred[np.isnan(pred)] = 0
        return pred

    def re_model(self, **kwargs) -> nn.Module:
        """
        BERT tokenizer -> Input features -> BERT (self.model) -> hidden states -> taking the mean of entities; bilinear
        formula -> return the whole model.
        model <= BERT + additional processing
        """
        return BertWithAdaThresholdLocContextPooling(
            n_classes=self.n_classes,
            pretrained_bert=self.pretrained_bert,
            bert_tokenizer_config_file=self.pretrained_bert,
            num_ner_tags=self.num_ner_tags,
            threshold=self.threshold,
            device=self.device
        )

    def collate_fn(self, batch: List[Dict]) -> Tuple[Tensor, Tensor, List, List, List]:
        input_ids = torch.tensor([f["input_ids"] for f in batch], dtype=torch.long)
        label = [f["label"] for f in batch]
        entity_pos = [f["entity_pos"] for f in batch]
        ner_tags = [f["ner_tags"] for f in batch]
        attention_mask = torch.tensor([f["attention_mask"] for f in batch], dtype=torch.float)
        out = (input_ids, attention_mask, entity_pos, ner_tags, label)
        return out
