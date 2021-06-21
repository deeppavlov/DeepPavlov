from logging import getLogger
from pathlib import Path
from typing import List, Dict, Union, Optional

import torch
import torch.nn as nn
import numpy as np
from overrides import overrides
from transformers import BertTokenizer, BertModel, AutoConfig, InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.classifiers.torch_re_bert import REBertWith

log = getLogger(__name__)


@register('re_torch_transformers_classifier')
class REBertModel(TorchModel):

    def __init__(
            self,
            n_classes: int,
            model_name: str,
            len_tokenizer: int = None,
            cls_token_id: int = 101,
            sep_token_id: int = 201,
            pretrained_bert: str = None,
            bert_config_file: Optional[str] = None,
            criterion: str = "CrossEntropyLoss",
            optimizer: str = "AdamW",
            optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999), "eps": 1e-6},
            return_probas: bool = False,
            attention_probs_keep_prob: Optional[float] = None,
            hidden_keep_prob: Optional[float] = None,
            clip_norm: Optional[float] = None,
            **kwargs
    ):
        self.n_classes = n_classes
        self.len_tokenizer = len_tokenizer
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
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

        # mb needed?
        # self.optimizer = getattr(torch.optim, self.optimizer_name)(
        #     self.model.parameters(), **self.optimizer_parameters)

    def train_on_batch(self, features: List[InputFeatures], y: List[int]):
        """
        Trains the relation extraction BERT model on the given batch.

        Args:
            features: batch of InputFeatures.
            y: batch of class labels.

        Returns:
            dict with loss and learning rate values.
        """

        _input = {'labels': torch.from_numpy(np.array(y)).to(self.device)}
        for elem in ['input_ids', 'attention_mask', 'entity_pos', 'token_type_ids']:
            _input[elem] = torch.cat([getattr(f, elem) for f in features], dim=0).to(self.device)

        self.optimizer.zero_grad()

        hidden_states = self.model(**_input)

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        return

    def __call__(self, features: List[InputFeatures]):

        _input = {}
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[elem] = torch.cat([getattr(f, elem) for f in features], dim=0).to(self.device)

        if not self.return_probas:
            pred = self.sess.run(self.y_predictions, feed_dict=_input)
        else:
            pred = self.sess.run(self.y_probas, feed_dict=_input)
        return pred

    def re_model(self, **kwargs) -> nn.Module:
        """
        BERT tokenizer -> Input features -> BERT (self.model) -> hidden states -> taking the mean of entities; bilinear formula -> return the whole model.
        model <= BERT + additional processing
        """
        return REBertWith(
            n_classes=self.n_classes,
            cls_token_id=self.cls_token_id,
            sep_token_id=self.sep_token_id,
            len_tokenizer=self.len_tokenizer,
            pretrained_bert=self.pretrained_bert,
            device=self.device
        )


if __name__ == "__main__":
    from joblib import load
    data = load("/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/dev_small")
    entity_pos = load("/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/"
                      "dev_small_entity_pos")
    ner_tags = load("/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/"
                    "dev_small_ner_tags")
    labels = load("/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/"
                  "dev_small_labels")
    n_classes = len(set(labels))

    from DeepPavlov.deeppavlov.core.data.simple_vocab import SimpleVocabulary
    smplvoc = SimpleVocabulary(
        save_path="/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/"
                  "dev_small_labels_enc",
    )
    smplvoc.fit(labels)
    labels_enc = smplvoc.__call__(labels)

    REBertModel(
        n_classes=n_classes,
        save_path="/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_model/model",
        load_path="/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_model/model",
        pretrained_bert="bert-base-uncased",
        len_tokenizer=30523,
        model_name="re_model",
        cls_token_id=101,
        sep_token_id=201,
    ).train_on_batch(data, labels_enc)





'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
special_tokens_dict = {'additional_special_tokens': ['<ENT>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

positions = [1, 3]
text = ["what", "is", "the", "capital", "of", "russia", "?"]

wordpiece_tokens = []

special_tokens_pos = []

count = 0
for n, token in enumerate(text):
    if n in positions:
        wordpiece_tokens.append("<ENT>")
        special_tokens_pos.append(count)
        count += 1

    word_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
    wordpiece_tokens += word_tokens
    count += len(word_tokens)

print(wordpiece_tokens)
print(special_tokens_pos)

encoding = tokenizer.encode_plus(wordpiece_tokens, add_special_tokens=True, truncation=True, padding="max_length",
                                 return_attention_mask=True, return_tensors="pt")
input_ids = encoding["input_ids"]
token_type_ids = encoding["token_type_ids"]
attention_mask = encoding["attention_mask"]

model_input = {"input_ids": [input_ids], "token_type_ids": [token_type_ids], "attention_mask": [attention_mask]}

print(encoding)

hidden_states = model(**encoding)

last_hidden_states = hidden_states.last_hidden_state
print(len(last_hidden_states), len(last_hidden_states[0]), len(last_hidden_states[0][0]))  # 1x512x768
'''
