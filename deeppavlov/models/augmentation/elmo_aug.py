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
from deeppavlov import build_model, configs
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
                 num_top_tokens,
                 model_dir,
                 scores_of_elmo_vocab_by_kenlm):
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
            self.scores_of_elmo_vocab_by_kenlm = np.array(json.load(f))
        self.elmo_lm = elmo_bilm.ELMoEmbedder(model_dir=model_dir)
        self.lang = lang
        self.penalty_for_source_token = penalty_for_source_token
        self.with_source_token = with_source_token
        self.num_top_tokens = num_top_tokens
        if self.lang == 'rus':
            self.word_filter = RuWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags)
            self.morpho_tagger = build_model(configs.models.morpho_tagger.UD2_0.morpho_ru_syntag, download=True)
        elif self.lang == 'eng':
            self.word_filter = EnWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags)
            self.morpho_tagger = build_model(configs.models.morpho_tagger.UD2_0.morpho_en, download=True)

        def _softmax(self, a, axis):
            numerator = np.exp(a - np.max(a))
            denominator = np.expand_dims(np.sum(numerator, axis=axis), 2)
            return numerator / denominator

        def _unite_distr(self, elmo_distr):
            elmo_distr = np.log(elmo_distr)
            elmo_distr = np.sum(elmo_distr, axis=1)
            elmo_distr = elmo_distr - self.scores_of_elmo_vocab_by_kenlm
            return self._softmax(elmo_distr, axis=1)

        def _multi_argmax(self, values: np.ndarray, n_instances: int = 1) -> np.ndarray:
            """
            Selects the indices of the n_instances highest values.
            Args:
                values: Contains the values to be selected from.
                n_instances: Specifies how many indices to return.
            Returns:
                Contains the indices of the n_instances largest values.
            """
            assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

            max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
            return max_idx

        def _sample_token_from_distr(self, distr):
            idxs = _multi_argmax(self, distr, self.num_top_tokens)
            token_idx = np.random.choice(idxs, 1, False, distr[idxs])
            return self.elmo_lm.get_vocab()[token_idx]

        def _transform_sentence(self, elmo_distr, morpho_tags, tokens):
            filter_res = self.word_filter.filter_words(tokens, morpho_tags)
            transformed_sentence = [self._sample_token_from_distr(vocab_distr) if not_filtered else token
                                    for token, not_filtered, vocab_distr in zip(tokens, filter_res, elmo_distr)]
            return transformed_sentence

        def __call__(self, batch_tokens: List[List[str]]) -> List[List[str]]:
            batch_morpho_tags = self.morpho_tagger(batch_tokens)
            batch_elmo_distr = self.elmo_lm(batch_tokens)
            batch_elmo_distr = [self._unite_distr(elmo_distr) for elmo_distr in batch_elmo_distr]
            ziped = zip(batch_elmo_distr, batch_morpho_tags, batch_tokens)
            transformed_batch = [self._transform_sentence(elmo_distr, morpho_tags, tokens)
                                 for elmo_distr, morpho_tags, tokens in ziped]
            return transformed_batch

