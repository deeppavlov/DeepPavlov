import itertools
from pathlib import Path
from logging import getLogger
from typing import List, Optional, Dict, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from overrides import overrides
from torch import Tensor
# from apex import amp

from deeppavlov.core.commands.utils import expand_path
from transformers import AutoConfig, AutoTokenizer, AutoModel
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


class BilinearClassification(nn.Module):

    def __init__(
            self,
            n_tags: int,
            pretrained_bert: str = None,
            device: str = "gpu"
    ):
        super().__init__()
        self.n_tags = n_tags
        self.pretrained_bert = pretrained_bert
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.config = AutoConfig.from_pretrained(self.pretrained_bert, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(self.pretrained_bert, config=self.config)
        self.encoder.to(self.device)
        self.zero_emb = torch.Tensor([0.0 for _ in range(self.config.hidden_size)]).to(self.device)
        self.fc = nn.Linear(self.config.hidden_size * 8, self.n_tags).to(self.device)
    
    def get_logits(self, input_ids, attention_mask, entity_subw_indices_batch, triplet_entity_nums_batch):
        outputs = self.encoder(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state
        embs_batch = []
        for n, entity_subw_indices_list in enumerate(entity_subw_indices_batch):
            embs_list = []
            for entity_subw_indices in entity_subw_indices_list:
                embs = []
                for ind in entity_subw_indices:
                    embs.append(hidden_states[n][ind])
                embs_list.append(torch.mean(torch.stack(embs), axis=0))
            embs_batch.append(embs_list)
        
        seq_lengths = []
        subj_batch, obj_batch = [], []
        for triplet_entity_nums_list, embs_list in zip(triplet_entity_nums_batch, embs_batch):
            subj_list, obj_list = [], []
            for subj_num, obj_num in triplet_entity_nums_list:
                subj_list.append(embs_list[subj_num])
                obj_list.append(embs_list[obj_num])
            seq_lengths.append(len(subj_list))
            subj_batch.append(subj_list)
            obj_batch.append(obj_list)
        
        max_seq_len = max(seq_lengths)
        for i in range(len(subj_batch)):
            for j in range(max_seq_len - len(subj_batch[i])):
                subj_batch[i].append(self.zero_emb)
                obj_batch[i].append(self.zero_emb)
        
        subj_tensors = []
        for subj_list in subj_batch:
            subj_tensors.append(torch.stack(subj_list))
        subj_tensor = torch.stack(subj_tensors).to(self.device)
        
        obj_tensors = []
        for obj_list in obj_batch:
            obj_tensors.append(torch.stack(obj_list))
        obj_tensor = torch.stack(obj_tensors).to(self.device)
        
        bs, seq_len, emb_size = subj_tensor.size()
        subj_tensor = subj_tensor.view(bs, seq_len, 96, 8, 1)
        obj_tensor = obj_tensor.view(bs, seq_len, 96, 1, 8)
        bl = (subj_tensor * obj_tensor).view(bs, seq_len, self.config.hidden_size * 8)
        logits = self.fc(bl)
        
        return logits

    def forward(self, input_ids, attention_mask, entity_subw_indices, triplet_entity_nums, labels = None):
        logits = self.get_logits(input_ids, attention_mask, entity_subw_indices, triplet_entity_nums)
        if labels is not None:
            labels_len = [len(elem) for elem in labels]
            max_len = max(labels_len)
            token_attention_mask = [[0 for _ in range(max_len)] for _ in labels]
            for i in range(len(triplet_entity_nums)):
                for j in range(len(triplet_entity_nums[i])):
                    token_attention_mask[i][j] = 1
            for i in range(len(labels)):
                for j in range(len(triplet_entity_nums[i]), len(labels[i])):
                    labels[i][j] = -1
            
            token_attention_mask = torch.LongTensor(token_attention_mask).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            
            if token_attention_mask is not None:
                active_loss = token_attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_tags)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.n_tags), labels.view(-1))
            return loss, logits
        else:
            return logits


@register('torch_transformers_bilinear_classification')
class TorchTransformersBilinearClassification(TorchModel):

    def __init__(
            self,
            n_tags: int,
            pretrained_bert: str = None,
            criterion: str = "CrossEntropyLoss",
            optimizer: str = "AdamW",
            optimizer_parameters: Dict = {"lr": 5e-5, "weight_decay": 0.01, "eps": 1e-6},
            return_probas: bool = False,
            attention_probs_keep_prob: Optional[float] = None,
            hidden_keep_prob: Optional[float] = None,
            clip_norm: Optional[float] = None,
            threshold: Optional[float] = None,
            **kwargs
    ):
        self.n_tags = n_tags
        self.pretrained_bert = pretrained_bert
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        super().__init__(
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            **kwargs)

    def train_on_batch(self, input_ids_batch, attention_mask_batch, entity_subw_indices_batch,
                             triplet_entity_nums_batch, labels):
        _input = {'entity_subw_indices': entity_subw_indices_batch, "triplet_entity_nums": triplet_entity_nums_batch}
        _input["input_ids"] = torch.LongTensor(input_ids_batch).to(self.device)
        _input["attention_mask"] = torch.LongTensor(attention_mask_batch).to(self.device)
        _input["labels"] = labels

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()      # zero the parameter gradients

        loss, softmax_scores = self.model(**_input)
        loss.backward()
        self.optimizer.step()

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def __call__(self, input_ids_batch, attention_mask_batch, entity_subw_indices_batch, triplet_entity_nums_batch):

        self.model.eval()
        _input = {'entity_subw_indices': entity_subw_indices_batch, "triplet_entity_nums": triplet_entity_nums_batch}
        _input["input_ids"] = torch.LongTensor(input_ids_batch).to(self.device)
        _input["attention_mask"] = torch.LongTensor(attention_mask_batch).to(self.device)

        with torch.no_grad():
            logits = self.model(**_input)
        
        probas = torch.nn.functional.softmax(logits, dim=-1)
        probas = probas.detach().cpu().numpy()

        logits = logits.detach().cpu().numpy()
        pred = np.argmax(logits, axis=-1)
        seq_lengths = [len(elem) for elem in entity_subw_indices_batch]
        pred = [p[:l] for l, p in zip(seq_lengths, pred)]

        if self.return_probas:
            return pred, probas
        else:
            return pred
        
    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        self.model = BilinearClassification(self.n_tags, self.pretrained_bert)
        # TODO that should probably be parametrized in config
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)
        super().load()
