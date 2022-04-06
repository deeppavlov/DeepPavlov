import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        st_tr = torch.Tensor([0.0 for _ in range(num_tags)])
        self.start_transitions = nn.Parameter(st_tr)
        self.end_transitions = nn.Parameter(st_tr)
        tr = [[0.0 for _ in range(num_tags)] for _ in range(num_tags)]
        tr = torch.Tensor(tr)
        self.transitions = nn.Parameter(tr)

    def forward(self, tags_batch: torch.LongTensor, mask: np.ndarray):
        seq_lengths = np.sum(y_masks, axis=1)
        for seq_len, tags_list in zip(seq_lengths, tags_batch):
            if seq_len > 1:
                for i in range(seq_len - 1):
                    self.stats[tags_list[i]][tags_list[i + 1]] += 1
        tr = [[0.0 for _ in range(self.tags_num)] for _ in range(self.tags_num)]
        for i in range(len(stats)):
            for j in range(len(stats[i])):
                if stats[i][j] > 0:
                    tr[i][j] = 0.0
                else:
                    tr[i][j] = -1000.0
        tr = torch.Tensor(tr)
        self.transitions = nn.Parameter(tr)

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
