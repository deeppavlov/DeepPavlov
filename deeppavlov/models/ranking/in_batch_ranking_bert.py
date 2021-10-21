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
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.ranking.siamese_el_ranking_bert import BilinearRanking

log = getLogger(__name__)


@register('torch_transformers_ranker')
class BertRanker(TorchModel):

    def __init__(
            self,
            model_name: str,
            text_encoder_save_path: str,
            descr_encoder_save_path: str,
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
                             c_features_list: List[List[Dict]],
                             positive_idx: List[List[int]]) -> float:

        _input = {'positive_idx': positive_idx}
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [f[elem] for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [f[elem] for c_features in c_features_list for f in c_features]
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
                       c_features_list: List[List[Dict]]) -> Union[List[int], List[np.ndarray]]:

        self.model.eval()

        _input = {}
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [f[elem] for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [f[elem] for c_features in c_features_list for f in c_features]
            _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)

        with torch.no_grad():
            softmax_scores = self.model(**_input)
            pred = torch.argmax(softmax_scores, dim=1).cpu().numpy()
            
        return pred

    def in_batch_ranking_model(self, **kwargs) -> nn.Module:
        return BertRanking(
            pretrained_bert=self.pretrained_bert,
            text_encoder_save_path=self.text_encoder_save_path,
            descr_encoder_save_path=self.descr_encoder_save_path,
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


class BertRanking(nn.Module):

    def __init__(
            self,
            text_encoder_save_path: str,
            descr_encoder_save_path: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            device: str = "gpu"
    ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.text_encoder_save_path = text_encoder_save_path
        self.descr_encoder_save_path = descr_encoder_save_path
        self.bert_config_file = bert_config_file
        self.device = device

        # initialize parameters that would be filled later
        self.q_encoder, self.c_encoder, self.config, self.bert_config = None, None, None, None
        self.load()

        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        else:
            tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        self.q_encoder.resize_token_embeddings(len(tokenizer) + 1)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

    def forward(
            self,
            q_input_ids: Tensor,
            q_attention_mask: Tensor,
            c_input_ids: Tensor,
            c_attention_mask: Tensor,
            positive_idx: List[List[int]] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        q_hidden_states, q_cls_emb, _ = self.q_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
        c_hidden_states, c_cls_emb, _ = self.c_encoder(input_ids=c_input_ids, attention_mask=c_attention_mask)
        dot_products = torch.matmul(q_cls_emb, torch.transpose(c_cls_emb, 0, 1))
        softmax_scores = F.log_softmax(dot_products, dim=1)
        if positive_idx is not None:
            loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx).to(softmax_scores.device), reduction="mean")
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
        self.q_encoder.to(self.device)
        self.c_encoder.to(self.device)


@register('torch_bert_cls_encoder')
class TorchBertCLSEncoder:
    def __init__(self, pretrained_bert, weights_path, add_special_tokens=None,
                       do_lower_case: bool = False, device: str = "gpu", **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pretrained_bert = pretrained_bert
        self.config = AutoConfig.from_pretrained(self.pretrained_bert, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert, do_lower_case=do_lower_case)
        if add_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': add_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.text_encoder = AutoModel.from_config(config=self.config)
        #self.text_encoder.resize_token_embeddings(len(self.tokenizer) + 1)
        self.weights_path = expand_path(weights_path)
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        self.text_encoder.load_state_dict(checkpoint["model_state_dict"])
        self.text_encoder.to(self.device)

    def __call__(self, texts_batch: List[str]):
        tokenizer_input = [[text, None] for text in texts_batch]
        encoding = self.tokenizer.batch_encode_plus(
            tokenizer_input, add_special_tokens = True, pad_to_max_length=True,
            return_attention_mask = True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)

        _, text_cls_emb, _ = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls_emb = text_cls_emb.detach().cpu().numpy()
        return text_cls_emb


@register('torch_bert_entity_ranker')
class TorchBertEntityRanker:
    def __init__(self, pretrained_bert,
                       context_weights_path,
                       bilinear_weights_path,
                       q_to_descr_emb_path,
                       special_token_id: int,
                       add_special_tokens=["[ENT]"],
                       do_lower_case: bool = False,
                       batch_size: int = 5,
                       device: str = "gpu", **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pretrained_bert = pretrained_bert
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_bert, do_lower_case=do_lower_case)
        if add_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': add_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.text_encoder, self.config = None, None
        self.config = AutoConfig.from_pretrained(self.pretrained_bert, output_hidden_states=True)
        self.text_encoder = BertModel.from_pretrained(self.pretrained_bert, config=self.config)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.context_weights_path = expand_path(context_weights_path)
        self.bilinear_weights_path = bilinear_weights_path
        self.q_to_descr_emb_path = q_to_descr_emb_path
        context_checkpoint = torch.load(self.context_weights_path, map_location=self.device)
        self.text_encoder.load_state_dict(context_checkpoint["model_state_dict"])
        self.text_encoder.to(self.device)
        self.bilinear_ranking = BilinearRanking()
        bilinear_checkpoint = torch.load(self.bilinear_weights_path, map_location=self.device)
        self.bilinear_ranking.load_state_dict(bilinear_checkpoint["model_state_dict"])
        self.bilinear_ranking.to(self.device)
        self.q_to_descr_emb = load_pickle(self.q_to_descr_emb_path)
        self.special_token_id = special_token_id
        self.zero_emb = np.zeros(768, dtype=np.float32)
        self.batch_size = batch_size

    def __call__(self, contexts_batch: List[str],
                       candidate_entities_batch: List[List[str]]):
        entity_emb_batch = []
        
        num_batches = len(contexts_batch) // self.batch_size + int(len(contexts_batch) % self.batch_size > 0)
        for ii in range(num_batches):
            contexts_list = contexts_batch[ii*self.batch_size:(ii+1)*self.batch_size]
            tokenizer_input = [[text, None] for text in contexts_list]
            encoding = self.tokenizer.batch_encode_plus(
                tokenizer_input, add_special_tokens = True, pad_to_max_length=True,
                return_attention_mask = True)
            context_input_ids = encoding["input_ids"]
            context_attention_mask = encoding["attention_mask"]
            context_input_ids = torch.LongTensor(context_input_ids).to(self.device)
            context_attention_mask = torch.LongTensor(context_attention_mask).to(self.device)
            special_tokens_pos = []
            for input_ids_list in context_input_ids:
                found_n = -1
                for n, input_id in enumerate(input_ids_list):
                    if input_id == self.special_token_id:
                        found_n = n
                        break
                if found_n == -1:
                    found_n = 0
                special_tokens_pos.append(found_n)
            
            context_hidden_states, *_ = self.text_encoder(input_ids=context_input_ids,
                                                          attention_mask=context_attention_mask)
        
            for i in range(len(special_tokens_pos)):
                pos = special_tokens_pos[i]
                entity_emb_batch.append(context_hidden_states[i, pos])
            
        scores_batch = []
        for entity_emb, candidate_entities_list in zip(entity_emb_batch, candidate_entities_batch):
            if candidate_entities_list:
                entity_emb = [entity_emb for _ in candidate_entities_list]
                entity_emb = torch.stack(entity_emb, dim=0).to(self.device)
                candidate_entities_emb = [self.q_to_descr_emb.get(entity, self.zero_emb) for entity in candidate_entities_list]
                candidate_entities_emb = torch.Tensor(candidate_entities_emb).to(self.device)
                scores_list, _ = self.bilinear_ranking(entity_emb, candidate_entities_emb)
                scores_list = scores_list.detach().cpu().numpy()
                scores_list = [score[1] for score in scores_list]
                entities_with_scores = [(entity, score) for entity, score in zip(candidate_entities_list, scores_list)]
                entities_with_scores = sorted(entities_with_scores, key=lambda x: x[1], reverse=True)
                scores_batch.append(entities_with_scores)
            else:
                scores_batch.append([])

        return scores_batch


class TextEncoder(nn.Module):
    def __init__(self, pretrained_bert: str = None,
                 bert_tokenizer_config_file: str = None,
                 bert_config_file: str = None,
                 resize: bool = False,
                 device: str = "gpu"
                 ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.encoder, self.config, self.bert_config = None, None, None
        self.device = device
        self.load()
        self.resize = resize
        '''
        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file)
        else: '''
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert)
        if self.resize:
            self.encoder.resize_token_embeddings(len(self.tokenizer) + 1)
        self.linear = nn.Linear(768, 300)
        
    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                entity_tokens_pos: List[int] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        if self.resize:
            q_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            q_hidden_states = q_outputs.last_hidden_state
            
            entity_emb = []
            for i in range(len(entity_tokens_pos)):
                pos = entity_tokens_pos[i]
                entity_emb.append(q_hidden_states[i, pos])
            
            entity_emb = torch.stack(entity_emb, dim=0)
            entity_emb = self.linear(entity_emb)
            return entity_emb
        else:
            c_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            c_cls_emb = c_outputs.last_hidden_state[:,:1,:].squeeze(1)
            c_cls_emb = self.linear(c_cls_emb)
            return c_cls_emb
            
    def load(self) -> None:
        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            self.config = AutoConfig.from_pretrained(
                self.pretrained_bert, output_hidden_states=True
            )
            self.encoder = AutoModel.from_pretrained(self.pretrained_bert, config=self.config)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.encoder = AutoModel.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")
        self.encoder.to(self.device)


@register('torch_distilbert_entity_ranker')
class TorchDistilBertEntityRanker:
    def __init__(self, pretrained_bert,
                       context_weights_path,
                       bilinear_weights_path,
                       q_to_descr_emb_path,
                       special_token_id: int,
                       add_special_tokens=["[ENT]"],
                       do_lower_case: bool = False,
                       batch_size: int = 5,
                       emb_size: int = 300,
                       device: str = "gpu", **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pretrained_bert = pretrained_bert
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert, do_lower_case=do_lower_case)
        if add_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': add_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.text_encoder, self.config = None, None
        self.config = AutoConfig.from_pretrained(self.pretrained_bert, output_hidden_states=True)
        self.emb_size = emb_size
        self.text_encoder = TextEncoder(pretrained_bert=self.pretrained_bert, resize=True, device=self.device)
        self.context_weights_path = expand_path(context_weights_path)
        self.bilinear_weights_path = bilinear_weights_path
        self.q_to_descr_emb_path = q_to_descr_emb_path
        context_checkpoint = torch.load(self.context_weights_path, map_location=self.device)
        self.text_encoder.load_state_dict(context_checkpoint["model_state_dict"])
        self.text_encoder.to(self.device)
        self.bilinear_ranking = BilinearRanking(emb_size=300, block_size=10)
        bilinear_checkpoint = torch.load(self.bilinear_weights_path, map_location=self.device)
        self.bilinear_ranking.load_state_dict(bilinear_checkpoint["model_state_dict"])
        self.bilinear_ranking.to(self.device)
        self.q_to_descr_emb = load_pickle(self.q_to_descr_emb_path)
        self.special_token_id = special_token_id
        self.zero_emb = np.zeros(300, dtype=np.float32)
        self.batch_size = batch_size

    def __call__(self, contexts_batch: List[str],
                       candidate_entities_batch: List[List[str]]):
        entity_emb_batch = []
        
        num_batches = len(contexts_batch) // self.batch_size + int(len(contexts_batch) % self.batch_size > 0)
        for ii in range(num_batches):
            contexts_list = contexts_batch[ii*self.batch_size:(ii+1)*self.batch_size]
            lengths = []
            for text in contexts_list:
                encoding = self.tokenizer.encode_plus(
                    text, add_special_tokens = True, pad_to_max_length = True, return_attention_mask = True)
                input_ids = encoding["input_ids"]
                lengths.append(len(input_ids))
                
            max_len = max(lengths)
            input_ids_batch = []
            attention_mask_batch = []
            for text in contexts_list:
                encoding = self.tokenizer.encode_plus(text, add_special_tokens = True,
                                                      truncation = True, max_length=max_len,
                                                      pad_to_max_length = True, return_attention_mask = True)
                input_ids_batch.append(encoding["input_ids"][:490])
                attention_mask_batch.append(encoding["attention_mask"][:490])
            
            context_input_ids = torch.LongTensor(input_ids_batch).to(self.device)
            context_attention_mask = torch.LongTensor(attention_mask_batch).to(self.device)
            special_tokens_pos = []
            for input_ids_list in context_input_ids:
                found_n = -1
                for n, input_id in enumerate(input_ids_list):
                    if input_id == self.special_token_id:
                        found_n = n
                        break
                if found_n == -1:
                    found_n = 0
                special_tokens_pos.append(found_n)
            
            cur_entity_emb_batch = self.text_encoder(input_ids=context_input_ids,
                                                 attention_mask=context_attention_mask,
                                                 entity_tokens_pos=special_tokens_pos)
        
            entity_emb_batch += cur_entity_emb_batch.detach().cpu().numpy().tolist()
            
        scores_batch = []
        for entity_emb, candidate_entities_list in zip(entity_emb_batch, candidate_entities_batch):
            if candidate_entities_list:
                entity_emb = [entity_emb for _ in candidate_entities_list]
                entity_emb = torch.Tensor(entity_emb).to(self.device)
                candidate_entities_emb = [self.q_to_descr_emb.get(entity, self.zero_emb) for entity in candidate_entities_list]
                candidate_entities_emb = torch.Tensor(candidate_entities_emb).to(self.device)
                scores_list, _ = self.bilinear_ranking(entity_emb, candidate_entities_emb)
                scores_list = scores_list.detach().cpu().numpy()
                scores_list = [score[1] for score in scores_list]
                entities_with_scores = [(entity, score) for entity, score in zip(candidate_entities_list, scores_list)]
                entities_with_scores = sorted(entities_with_scores, key=lambda x: x[1], reverse=True)
                scores_batch.append(entities_with_scores)
            else:
                scores_batch.append([])

        return scores_batch
