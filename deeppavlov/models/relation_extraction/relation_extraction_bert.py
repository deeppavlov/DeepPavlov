from logging import getLogger
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import InputFeatures

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
            tokenizer_vocab_file: str = "bert-base-cased",
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
        self.tokenizer_vocab_file = tokenizer_vocab_file
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

    def train_on_batch(self, features: List[Dict], train_batch_size: int, num_epochs: int, device: str = "cpu"):
        """
        Trains the relation extraction BERT model on the given batch.

        Returns:
            dict with loss and learning rate values.
        """

        self.model.train()

        train_dataloader = DataLoader(
            features, batch_size=train_batch_size, shuffle=False, collate_fn=self.collate_fn, drop_last=True
        )

        # _input = {'labels': torch.from_numpy(np.array(y)).to(self.device)}
        # for elem in ['input_ids', 'attention_mask', 'entity_pos', 'token_type_ids']:
        #     _input[elem] = torch.cat([getattr(f, elem) for f in features], dim=0).to(self.device)

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            self.model.zero_grad()
            for _, batch in enumerate(train_dataloader):
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device),
                    'entity_pos': batch[2],
                    'ner_tags': batch[3],
                    'label': batch[4]
                }
                hidden_states = self.model(**inputs)
                loss = hidden_states[0]
                loss.backward()
                self.optimizer.step()

                # Clip the norm of the gradients to prevent the "exploding gradients" problem
                if self.clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

    def __call__(self, features: List[InputFeatures]):

        _input = {}
        for elem in ['input_ids', 'attention_mask']:
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
        return BertWithAdaThresholdLocContextPooling(
            n_classes=self.n_classes,
            tokenizer_vocab_file=self.tokenizer_vocab_file,
            pretrained_bert=self.pretrained_bert,
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
    data = load("/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/dev_small")

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
        tokenizer_vocab_file="bert-base-cased"
    ).train_on_batch(data, train_batch_size=4, num_epochs=2)





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