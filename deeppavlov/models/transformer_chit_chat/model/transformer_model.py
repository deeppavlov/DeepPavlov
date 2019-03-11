#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeppavlov.models.transformer_chit_chat.model.transformer_module import TransformerModule

class TransformerModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size,
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 bos_id, eos_id,
                 max_seq_len=256,
                 beam_size=5,
                 sample=False,
                 length_penalty=0.8,
                 annealing_topk=None,
                 annealing=0,
                 diversity_coef=0,
                 diversity_groups=1,
                 n_segments=None,
                 bert_mode=False,
                 type_vocab_size=2,
                 sep_id=None,
                 tie_weights=True,
                 info_bos_id=1,
                 talker1_bos_id=3,
                 talker2_bos_id=5,
                 bos_token_id=7,
                 sep_token_id=102,
                 ):

        super(TransformerModel, self).__init__()

        self.padding_idx = padding_idx
        self.n_embeddings = n_embeddings
        self.n_pos_embeddings = n_pos_embeddings
        self.embeddings_size = embeddings_size

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.sep_id = sep_id

        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.sample = sample
        self.length_penalty_coef = length_penalty
        self.annealing = annealing
        self.annealing_topk = annealing_topk
        self.diversity_coef = diversity_coef
        self.diversity_groups = diversity_groups

        self.transformer_module = TransformerModule(n_layers, n_embeddings, n_pos_embeddings, embeddings_size,
                                                    padding_idx, n_heads, dropout, embed_dropout, attn_dropout,
                                                    ff_dropout, n_segments, bert_mode, type_vocab_size,
                                                    info_bos_id=info_bos_id,
                                                    talker1_bos_id=talker1_bos_id,
                                                    talker2_bos_id=talker2_bos_id,
                                                    bos_token_id=bos_token_id,
                                                    sep_token_id=sep_token_id
                                                    )
        self.pre_softmax = nn.Linear(embeddings_size, n_embeddings, bias=False)
        # self.pre_softmax.weight = self.transformer_module.embeddings.weight
        if tie_weights:
            self.pre_softmax.weight = self.transformer_module.embeddings.weight
        else:
            torch.nn.init.xavier_uniform(self.pre_softmax.weight)

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x):
        return self.transformer_module(x)

    def generate(self, enc_x):
        return self.pre_softmax(enc_x)

    def decode(self, x, enc_contexts=[]):
        x, _ = self.transformer_module(x, enc_contexts)
        return self.generate(x)

    def predict(self, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        prediction = self.beam_search(enc_contexts)

        return prediction

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def beam_search(self, enc_contexts=[], return_beams=False):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long, device=device)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)

            for i in range(self.max_seq_len):
                outputs, _ = self.transformer_module(prevs, beam_enc_contexts)

                logits = self.generate(outputs[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.view(batch_size, self.beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, self.n_embeddings)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            beam_probas = F.softmax(g_beam_scores, dim=-1)
                            if self.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)

                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, self.n_embeddings), torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = self.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            confidence = []
            result = prevs.view(batch_size, self.beam_size, -1)

            if return_beams:
                return result, beam_lens
            if self.sample:
                probs = F.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            # for scores, indx in zip(beam_scores,bests):
            #     confidence.append(float(scores[indx]))

            # for i in range(batch_size):
            #     best_len = beam_lens[i, bests[i]]
            #     best_seq = result[i, bests[i], 1:best_len-1]
            #     predicts.append(best_seq.tolist())

            for r,p,b in zip(result, probs, beam_lens):
                # best_len = beam_lens[i, bests[i]]
                # best_seq = result[i, bests[i], 1:best_len-1]
                predicts.append(r.tolist())
                confidence.append(p.tolist())

        return predicts, confidence
