from logging import getLogger
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from transformers import InputFeatures
from apex import amp

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.classifiers.torch_re_bert import BertWithAdaThresholdLocContextPooling

log = getLogger(__name__)


@register('re_torch_transformers_classifier')
class REBertModel(TorchModel):

    def __init__(
            self,
            n_classes: int,
            model_name: str,
            pretrained_bert: str = None,
            bert_config_file: Optional[str] = None,
            criterion: str = "CrossEntropyLoss",
            optimizer: str = "AdamW",
            optimizer_parameters: Dict = {"lr": 5e-5, "weight_decay": 0.01, "eps": 1e-6},
            return_probas: bool = False,
            attention_probs_keep_prob: Optional[float] = None,
            hidden_keep_prob: Optional[float] = None,
            clip_norm: Optional[float] = None,
            **kwargs
    ):
        self.n_classes = n_classes
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        if self.n_classes == 0:
            raise ConfigError("Please provide a valid number of classes.")

        super().__init__(
            n_classes=n_classes,
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            **kwargs)

    def train_on_batch(self, features: List[Dict], labels: List) -> float:
        """
        Trains the relation extraction BERT model on the given batch.
        Args:

        Returns:
            dict with loss and learning rate values.
        """

        _input = {'label': torch.from_numpy(np.array(labels)).to(self.device)}
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [f[elem] for f in features]
            _input[elem] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['entity_pos', 'ner_tags']:
            inp_elem = [f[elem] for f in features]
            _input[elem] = inp_elem

        print(self.epochs_done)
        self.model.train()
        self.optimizer.zero_grad()      # zero the parameter gradients

        hidden_states = self.model(**_input)
        loss = hidden_states[0]
        loss.backward()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def __call__(self, features: List[InputFeatures]):

        _input = {}
        for elem in ['input_ids', 'attention_mask']:
            _input[elem] = torch.cat([getattr(f, elem) for f in features], dim=0).to(self.device)

        if not self.return_probas:
            # todo: class id -> Batch[int]
            pred = self.sess.run(self.y_predictions, feed_dict=_input)
        else:
            # todo: softmax over classes -> Batch[np.ndarray]
            pred = self.sess.run(self.y_probas, feed_dict=_input)


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


if __name__ == "__main__":
    from joblib import load
    features = load("/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/dev_small")
    labels = load("/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/dev_labels_small")
    n_classes = 97

    # from DeepPavlov.deeppavlov.core.data.simple_vocab import SimpleVocabulary
    # smplvoc = SimpleVocabulary(
    #     save_path="/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/"
    #               "dev_small_labels_enc",
    # )
    # smplvoc.fit(data["labels"])
    # labels_enc = smplvoc.__call__(data["labels"])

    REBertModel(
        n_classes=n_classes,
        save_path="/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_model/model",
        load_path="/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_model/model",
        pretrained_bert="bert-base-uncased",
        model_name="re_model",
    ).train_on_batch(features, labels)
