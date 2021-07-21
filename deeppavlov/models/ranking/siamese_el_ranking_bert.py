import itertools
from pathlib import Path
from logging import getLogger
from typing import List, Optional, Dict, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
# from apex import amp

from deeppavlov.core.commands.utils import expand_path
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertModel, BertTokenizer
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('siamese_bert_el_ranker')
class SiameseBertElRanker(TorchModel):

    def __init__(
            self,
            model_name: str,
            text_encoder_save_path: str,
            descr_encoder_save_path: str,
            bilinear_save_path: str,
            pretrained_bert: str = None,
            bert_config_file: Optional[str] = None,
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
        self.text_encoder_save_path = text_encoder_save_path
        self.descr_encoder_save_path = descr_encoder_save_path
        self.bilinear_save_path = bilinear_save_path
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            **kwargs)

    def train_on_batch(self, q_features: List[Dict],
                             c_features: List[Dict],
                             entity_tokens_pos: List[int],
                             labels: List[int]) -> float:

        _input = {'labels': labels}
        _input['entity_tokens_pos'] = entity_tokens_pos
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in c_features]
            _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)

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

    def __call__(self, q_features: List[Dict],
                       c_features: List[Dict],
                       entity_tokens_pos: List[int]) -> Union[List[int], List[np.ndarray]]:

        self.model.eval()

        _input = {'entity_tokens_pos': entity_tokens_pos}
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in c_features]
            _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)

        with torch.no_grad():
            softmax_scores = self.model(**_input)
            if self.return_probas:
                pred = softmax_scores
            else:
                pred = torch.argmax(softmax_scores, dim=1).cpu().numpy()
            
        return pred

    def siamese_ranking_el_model(self, **kwargs) -> nn.Module:
        return SiameseBertElModel(
            pretrained_bert=self.pretrained_bert,
            text_encoder_save_path=self.text_encoder_save_path,
            descr_encoder_save_path=self.descr_encoder_save_path,
            bilinear_save_path=self.bilinear_save_path,
            bert_tokenizer_config_file=self.pretrained_bert,
            device=self.device
        )
        
    def save(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        if fname is None:
            fname = self.save_path
        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        weights_path = Path(fname).with_suffix(f".pth.tar")
        log.info(f"Saving model to {weights_path}.")
        torch.save({
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_done": self.epochs_done
        }, weights_path)
        self.model.to(self.device)
        self.model.save()


class BilinearRanking(nn.Module):
    def __init__(self, n_classes: int = 2, emb_size: int = 768, block_size: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.block_size = block_size
        self.bilinear = nn.Linear(self.emb_size * self.block_size, self.n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text1: Tensor, text2: Tensor):
        b1 = text1.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = text2.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        softmax_logits = self.softmax(logits)
        log_softmax = F.log_softmax(logits, dim=-1)
        return softmax_logits, log_softmax


class SiameseBertElModel(nn.Module):

    def __init__(
            self,
            text_encoder_save_path: str,
            descr_encoder_save_path: str,
            bilinear_save_path: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            device: str = "gpu"
    ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.text_encoder_save_path = text_encoder_save_path
        self.descr_encoder_save_path = descr_encoder_save_path
        self.bilinear_save_path = bilinear_save_path
        self.bert_config_file = bert_config_file
        self.device = device

        # initialize parameters that would be filled later
        self.q_encoder, self.c_encoder, self.config, self.bert_config = None, None, None, None
        self.load()

        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_bert)
        self.q_encoder.resize_token_embeddings(len(self.tokenizer) + 1)
        self.bilinear_ranker = BilinearRanking()

    def forward(
            self,
            q_input_ids: Tensor,
            q_attention_mask: Tensor,
            c_input_ids: Tensor,
            c_attention_mask: Tensor,
            entity_tokens_pos: List,
            labels: List[int] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        q_hidden_states, q_cls_emb, _ = self.q_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
        entity_emb = []
        for i in range(len(entity_tokens_pos)):
            pos = entity_tokens_pos[i]
            entity_emb.append(q_hidden_states[i, pos])
        
        entity_emb = torch.stack(entity_emb, dim=0)
        c_hidden_states, c_cls_emb, _ = self.c_encoder(input_ids=c_input_ids, attention_mask=c_attention_mask)
        softmax_scores, log_softmax = self.bilinear_ranker(entity_emb, c_cls_emb)
        
        if labels is not None:
            labels_one_hot = [[0.0, 0.0] for _ in labels]
            for i in range(len(labels)):
                labels_one_hot[i][labels[i]] = 1.0
            labels_one_hot = torch.Tensor(labels_one_hot).to(self.device)
            
            bs, dim = labels_one_hot.shape
            per_sample_loss = -torch.bmm(labels_one_hot.view(bs, 1, dim), log_softmax.view(bs, dim, 1)).squeeze(2).squeeze(1)
            loss = torch.mean(per_sample_loss)
            return loss, softmax_scores
        else:
            return softmax_scores

    def load(self) -> None:
        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            self.config = AutoConfig.from_pretrained(
                self.pretrained_bert, output_hidden_states=True
            )
            self.q_encoder = BertModel.from_pretrained(self.pretrained_bert, config=self.config)
            self.c_encoder = BertModel.from_pretrained(self.pretrained_bert, config=self.config)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.q_encoder = BertModel.from_config(config=self.bert_config)
            self.c_encoder = BertModel.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.q_encoder.to(self.device)
        self.c_encoder.to(self.device)
        
    def save(self) -> None:
        text_encoder_weights_path = expand_path(self.text_encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving text encoder to {text_encoder_weights_path}.")
        torch.save({"model_state_dict": self.q_encoder.cpu().state_dict()}, text_encoder_weights_path)
        descr_encoder_weights_path = expand_path(self.descr_encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving descr encoder to {descr_encoder_weights_path}.")
        torch.save({"model_state_dict": self.c_encoder.cpu().state_dict()}, descr_encoder_weights_path)
        bilinear_weights_path = expand_path(self.bilinear_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving bilinear weights to {bilinear_weights_path}.")
        torch.save({"model_state_dict": self.bilinear_ranker.cpu().state_dict()}, bilinear_weights_path)
        self.q_encoder.to(self.device)
        self.c_encoder.to(self.device)
        self.bilinear_ranker.to(self.device)
