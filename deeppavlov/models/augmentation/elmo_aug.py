from deeppavlov.models.bidirectional_lms import elmo_bilm
from deeppavlov.models.augmentation.utils.word_filter import RuWordFilter
from deeppavlov.models.augmentation.utils.word_filter import EnWordFilter
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from logging import getLogger
import numpy as np
import json
from pathlib import Path
from typing import List
from scipy.stats import kurtosis
import pandas as pd

logger = getLogger(__name__)

@register("elmo_augmentation")
class ElmoAug(Component):
    """

    """
    def __init__(self,
                 lang,
                 penalty_for_source_token,
                 isalpha_only,
                 replace_freq,
                 not_replaced_tokens,
                 replaced_pos_tags,
                 with_source_token,
                 model_dir="/cephfs/home/sultanov/elmo_lm/lib/python3.6/site-packages/download/bidirectional_lms/elmo_en_news",
                 scores_of_elmo_vocab_by_kenlm="./"):
        if lang not in ['rus', 'eng']:
            raise ValueError(f"""Argument of {type(self).__name__}: 'lang'
                                should be chosen from set ['rus', 'eng'],
                                'rus' for russian, 'eng' for english""")
        if not 0 <= penalty_for_source_token < 1:
            raise ValueError(f"""Argument of {type(self).__name__}: 'penalty_for_source_token'
                                should be defined in half-interval [0,1)""")
        if not 0 <= replace_freq <= 1:
            raise ValueError(f"""Argument of {type(self).__name__}: 'replace_freq'
                                should be defined in closed-interval [0,1]""")
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"""Elmo model file have not been found in {model_dir}""")
        if not Path(scores_of_elmo_vocab_by_kenlm).exists():
            raise FileNotFoundError(f"""Json file with scores of elmo vocab 
                                        have not been found in {scores_of_elmo_vocab_by_kenlm}""")

        with open(scores_of_elmo_vocab_by_kenlm, 'r') as f:
            self.scores_of_elmo_vocab_by_kenlm = json.load(f)
        self.elmo_lm = elmo_bilm.ELMoEmbedder(model_dir=model_dir)
        self.lang = lang
        self.penalty_for_source_token = penalty_for_source_token
        self.with_source_token = with_source_token
        if self.lang == 'rus':
            self.word_filter = RuWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags)
        elif self.lang == 'eng':
            self.word_filter = EnWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags)

    def _softmax(self, a, axis):
        numerator = np.exp(a - np.max(a))
        denominator = np.expand_dims(np.sum(numerator, axis=axis), 2)
        return numerator / denominator

