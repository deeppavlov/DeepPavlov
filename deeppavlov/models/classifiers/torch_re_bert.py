import logging
from pathlib import Path
from typing import Tuple, Union, Any

import torch
from torch import Tensor
import torch.nn as nn
from opt_einsum import contract
from transformers import AutoConfig, BertModel

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.models.relation_extraction.losses import ATLoss

log = logging.getLogger(__name__)


class BertWithAdaThresholdLocContextPooling(nn.Module):

    def __init__(
            self,
            n_classes: int = 97,
            cls_token_id: int = 101,
            sep_token_id: int = 201,
            len_tokenizer: int = None,
            pretrained_bert: str = None,
            bert_config_file: str = None,
            emb_size: int = 768,
            block_size: int = 8,       # 64
            device: str = "gpu",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        self.len_tokenizer = len_tokenizer
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.device = device
        self.emb_size = emb_size
        self.block_size = block_size
        self.loss_fnt = ATLoss()

        # initialize parameters that would be filled later
        self.model, self.config, self.bert_config = None, None, None
        self.load()

        self.hidden_size = self.config.hidden_size
        self.head_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.bilinear = nn.Linear(self.emb_size * self.block_size, self.n_classes)

    # def forward(self, input_features: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         x - input features
    #
    #     Return:
    #         probability of classes
    #     """
    #     hidden_states = self.model(**input_features)  # BERT processing
    #     # merging the entities' hidden states
    #     # layers

    def forward(
            self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor, entity_pos: Tensor, hts: Tensor,
            token_type_ids: Tensor
            # todo: add NER tags!!
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        sequence_output, attention = output[0], output[-1][-1]
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.n_classes),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output

    def get_hrt(
            self, sequence_output: Tensor, attention: Tensor, entity_pos: Tensor, hts: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        offset = 1
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def load(self) -> None:
        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            self.config = AutoConfig.from_pretrained(
                self.pretrained_bert, num_labels=self.n_classes, output_attentions=False, output_hidden_states=False
            )
            self.model = BertModel.from_pretrained(self.pretrained_bert, config=self.config)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.model = BertModel.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.model.to(self.device)

        if self.len_tokenizer:  # resize tokenizer if the length of tokenizer is given
            self.model.resize_token_embeddings(self.len_tokenizer)