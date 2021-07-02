import logging
from pathlib import Path
from typing import Tuple, Union, Any, List

import torch
from torch import Tensor
import torch.nn as nn
from opt_einsum import contract
from transformers import AutoConfig, BertModel, BertTokenizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.models.relation_extraction.losses import ATLoss

log = logging.getLogger(__name__)


class BertWithAdaThresholdLocContextPooling(nn.Module):

    def __init__(
            self,
            n_classes: int = 97,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            emb_size: int = 768,
            block_size: int = 8,       # 64
            device: str = "gpu",
            ner_tags_length: int = 6        # number of ner tags
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.device = device
        self.ner_tags_length = ner_tags_length
        self.emb_size = emb_size
        self.block_size = block_size
        self.loss_fnt = ATLoss()

        # initialize parameters that would be filled later
        self.model, self.config, self.bert_config = None, None, None
        self.load()

        # initialize tokenizer to call resize_token_embeddings function for model with increased tokenizer size (due to
        # the additional <ENT> token) and get CLS and SEP token ids
        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        else:
            tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        self.model.resize_token_embeddings(len(tokenizer) + 1)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        self.hidden_size = self.config.hidden_size
        self.head_extractor = nn.Linear(2 * self.hidden_size + self.ner_tags_length, self.emb_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size + self.ner_tags_length, self.emb_size)
        self.bilinear = nn.Linear(self.emb_size * self.block_size, self.n_classes)

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            entity_pos: List,
            ner_tags: List,
            labels: List = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        sequence_output = output[0]  # Tensor (batch_size x input_length x 768)
        attention = output[-1][-1]  # Tensor (batch_size x 12 x input_length x input_length)

        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos)       # Tensors (batch_size x 768)

        # get ner tags of entities
        hs_ner_tags, ts_ner_tags = torch.Tensor([list(ele) for ele in list(zip(*ner_tags))]).to(self.device)
        # out = open("log3.txt", 'a')
        # out.write(str(hs[0])+'\n')
        # out.write("_"*100+'\n')
        # out.write(str(rs[0])+'\n')
        # out.write("_"*100+'\n')
        # out.write(str(hs_ner_tags)+'\n')
        # out.write("_"*100+'\n')
        # out.write(str(hs_ner_tags[0])+'\n')
        # out.write("_"*100+'\n')
        hs_inp = torch.cat([hs, rs, hs_ner_tags], dim=1)
        ts_inp = torch.cat([ts, rs, ts_ner_tags], dim=1)

        # out.write(str(len(hs_inp[0]))+'\t'+str(2*self.hidden_size + 2*self.ner_tags_length)+'\n')
        # out.write("_"*100+'\n')
        # out.close()

        out = open("log.txt", 'a')
        out.write(f"Attention shape: {attention.shape}" + '\n')
        out.write(f"Sequence_output shape: {sequence_output.shape}" + '\n')
        out.write(f"hs shape: {hs.shape}" + '\n')
        out.write(f"rs shape: {rs.shape}" + '\n')
        out.write(f"ts shape: {ts.shape}" + '\n')
        out.write(f"hs_ner_tags shape: {hs_ner_tags.shape}" + '\n')
        out.write(f"ts_ner_tags shape: {ts_ner_tags.shape}" + '\n')
        out.write(f"hs_inp shape: {hs_inp.shape}" + '\n')
        out.write(f"ts_inp shape: {ts_inp.shape}" + '\n')

        hs = torch.tanh(self.head_extractor(hs_inp))
        ts = torch.tanh(self.tail_extractor(ts_inp))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        out.write(f"hs shape: {hs.shape}" + '\n')
        out.write(f"ts shape: {ts.shape}" + '\n')
        out.write(f"b1 shape: {b1.shape}" + '\n')
        out.write(f"b2 shape: {b2.shape}" + '\n')
        out.write(f"bl shape: {bl.shape}" + '\n')
        out.write(f"logits shape: {logits.shape}" + '\n')
        out.close()

        output = (self.loss_fnt.get_label(logits, num_labels=self.n_classes), logits)
        if labels is not None:
            labels_tensors = [torch.tensor(label) for label in labels]
            labels_tensors = torch.stack(labels_tensors).to(logits)
            loss = self.loss_fnt(logits.float(), labels_tensors.float())
            output = (loss.to(sequence_output),) + output
        return output

    def get_hrt(self, sequence_output: Tensor, attention: Tensor, entity_pos: List) -> Tuple[Tensor, Tensor, Tensor]:
        offset = 1
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):            # for each training sample (= doc)
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:             # for each entity (= list of entity mentions)
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:            # for start and end position of each mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                        else:
                            e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                            e_emb = [e_emb]
                            e_att = torch.zeros(h, c).to(attention)
                    if len(e_emb) > 0:
                        out = open("log_e_emb.txt", 'a')
                        out.write(str(e_emb) + '\n')
                        out.write("_" * 100 + '\n')
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)           # get an embedding of an entity
                entity_atts.append(e_att)       # get attention of an entity

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]           # entity embeddings for each document
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            hs = torch.index_select(entity_embs, 0, torch.tensor([0]).to(self.device))  # embeddings of the first entity
            ts = torch.index_select(entity_embs, 0, torch.tensor([1]).to(self.device)) # embeddings of the second entity

            h_att = torch.index_select(entity_atts, 0, torch.tensor([0]).to(self.device))
            t_att = torch.index_select(entity_atts, 0, torch.tensor([1]).to(self.device))
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)  # ht_i.shape[0] x sequence_output.shape[2]
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
