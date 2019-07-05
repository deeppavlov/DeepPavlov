from logging import getLogger
from itertools import zip_longest
from typing import List, Union, Optional
from math import log as ln
import re

from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag
from deeppavlov.models.augmentation.utils.inflection import RuInflector
from deeppavlov.models.augmentation.utils.inflection import EnInflector
from deeppavlov.models.augmentation.utils.lettercaser import Lettercaser
from deeppavlov.models.augmentation.utils.thesaurus_wrapper import RuThesaurus
from deeppavlov.models.augmentation.utils.thesaurus_wrapper import EnThesaurus
from deeppavlov.models.augmentation.utils.word_filter import RuWordFilter
from deeppavlov.models.augmentation.utils.word_filter import EnWordFilter
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = getLogger(__name__)


@register("thesaurus_augmentation")
class ThesaurusAug(Component):
    """Component for augmentation, based on replacing words with synonyms from thesaurus

    Args:
        lang: language of text, that will be augmented, 'eng' for english, 'rus' for russian
        penalty_for_source_token: [0, 1] penalty for using source token
        replace_freq: [0,1] frequence of replacing tokens,
                      it calculates respecting to tokens that passed other filters
        isalpha_only: flag that activate filter based on method str.isalpha()
        not_replaced_tokens: list of tokens that should not be replaced
        with_source_token: flag that decides source tokens is synonyms for itself or not
        cases: dictionary that describes map:
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string,
                      when lettercase was not be detected in 'put_in_case' func
        replaced_pos_tags: List of pos_tags that can be replaced,
                           e.g. 'NOUN' for Noun, 'VERB' for Verb, 'ADJ' for Adjective, 'ADV' for Adverb
        ru_thes_dirpath: path which contains russian thesaurus 'RuThes-Lite2' in csv format
        en_classical_pluralize: arg from pattern.en.inflect.pluralize function

    Attributes:
        inflector: object of {RuInflector, EnInflector} class, it inflects tokens and defines lemma form
        thesaurus: object of {RuThesaurus, EnThesaurus} class,
                   wrap for thesaurus, e.g. ruthes-lite2 for russian, wordnet for english
        word_filter: object of {RuWordFilter, EnWordFilter} class, it decides which token can be replaced
        lettercaser: object of Lettercaser class, it defines lettercases and restores them
        penalty_for_source_token: [0, 1] penalty for using source token
        with_source_token: flag that decides source tokens is synonyms for itself or not
        lang: language of text, that will be augmented, 'eng' for english, 'rus' for russian
    """

    def __init__(self,
                 lang: str,
                 penalty_for_source_token: float=0.0001,
                 replace_freq: float=1.0,
                 isalpha_only: bool=True,
                 not_replaced_tokens: List[str]=None,
                 with_source_token: bool=True,
                 cases: dict=None,
                 default_case=None,
                 replaced_pos_tags: List[str]=None,
                 ru_thes_dirpath = None,
                 en_classical_pluralize: bool=True, *args, **kwargs):
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
        if lang == 'rus':
            self.inflector = RuInflector()
            self.thesaurus = RuThesaurus(dir_path=ru_thes_dirpath, with_source_token=False)
            self.word_filter = RuWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags)
            self.reflexive_exp = re.compile(r'.*ся\b|.*сь\b')
        elif lang == 'eng':
            self.inflector = EnInflector(classical_pluralize=en_classical_pluralize)
            self.thesaurus = EnThesaurus(with_source_token=False)
            self.word_filter = EnWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags)
        self.lettercaser = Lettercaser(cases=cases, default_case=default_case)
        self.penalty_for_source_token = penalty_for_source_token
        self.with_source_token = with_source_token
        self.lang = lang

    def _unite_phrasal_verbs(self, tokens, morpho_tags):
        res_tokens, res_morpho_tags, is_pass_next_token = [], [], False
        ziped = zip_longest(tokens, tokens[1:], morpho_tags, morpho_tags[1:])
        for cur_token, next_token, cur_morpho_tag, next_morpho_tag in ziped:
            if is_pass_next_token:
                is_pass_next_token = False
                continue
            elif cur_morpho_tag['pos_tag'] == 'VERB' and next_morpho_tag['pos_tag'] == 'ADP':
                res_tokens.append("_".join((cur_token, next_token)))
                res_morpho_tags.append(cur_morpho_tag)
                is_pass_next_token = True
            else:
                res_tokens.append(cur_token)
                res_morpho_tags.append(cur_morpho_tag)
        return res_tokens, res_morpho_tags

    def _disunit_phrasal_verbs(self, synonyms, filter_res):
        ziped = zip(synonyms, filter_res)
        [list(map(lambda x: x.replace("_", " "), syns)) if not_filtered else None for syns, not_filtered in ziped]

    def _get_cases(self, tokens, filter_res):
        ziped = zip(tokens, filter_res)
        return [self.lettercaser.determine_lettercase(token)
                if not_filtered else None for token, not_filtered in ziped]

    def _get_lemmas(self, tokens, morpho_tags, filter_res):
        ziped = zip(tokens, morpho_tags, filter_res)
        return [self.inflector.lemmatize(token, morpho_tag)
                if not_filtered else None for token, morpho_tag, not_filtered in ziped]

    def _get_synonyms(self, lemmas, morpho_tags, filter_res):
        ziped = zip(lemmas, morpho_tags, filter_res)
        return [self.thesaurus.get_syn(lemma, morpho_tag)
                if not_filtered else None for lemma, morpho_tag, not_filtered in ziped]

    def _inflect_synonyms(self, synonyms, morpho_tags, filter_res):
        ziped = zip(synonyms, morpho_tags, filter_res)
        return [list(map(lambda syn: self.inflector.inflect_token(syn, morpho_tag), syns))
                if not_filtered else None for syns, morpho_tag, not_filtered in ziped]

    def _rest_cases(self, synonyms, cases, filter_res):
        ziped = zip(synonyms, cases, filter_res)
        return [list(map(lambda syn: self.lettercaser.put_in_lettercase(syn, case), syns))
                if not_filtered else None for syns, case, not_filtered in ziped]

    def _insert_source_tokens(self, synonyms, source_tokens, filter_res):
        ziped = zip(synonyms, source_tokens, filter_res)
        if self.with_source_token:
            return [[source_token] + syns
                    if (not_filtered and syns) else [source_token] for syns, source_token, not_filtered in ziped]
        return [syns
                if (not_filtered and syns) else [source_token] for syns, source_token, not_filtered in ziped]

    def _filter_none_value(self, synonyms: List[List[Union[str, None]]]):
        return [list(filter(lambda x: x is not None, syns)) if syns is not None else None for syns in synonyms]

    def __insert_prior_prob(self, syns):
        return [(0, syn) for syn in syns]

    def __penalty_for_source_token(self, syns_with_prob):
        if self.penalty_for_source_token != 0:
            source_token_prob = ln(1-self.penalty_for_source_token)
            syns_with_prob[0] = (source_token_prob, syns_with_prob[0][1])
        return syns_with_prob

    def _transform_for_kenlm_elector(self, candidates, filter_res):
        with_prob = [self.__insert_prior_prob(candidate) for candidate in candidates]
        if self.with_source_token:
            return [self.__penalty_for_source_token(candidate)
                    if not_filtered else candidate for candidate, not_filtered in zip(with_prob, filter_res)]
        return with_prob

    def _transform_morpho_tags_in_dict(self, morpho_tags: str) -> List[dict]:
        morpho_tags, result = morpho_tags.split('\n'), []
        for morpho_tag in morpho_tags[:-1]:
            morpho_tag = morpho_tag.split('\t', maxsplit=2)
            pos, morpho_features = make_pos_and_tag(morpho_tag[-1].replace('\t', ' '), sep=' ', return_mode='dict')
            result.append({
                'source_token': morpho_tag[1],
                'pos_tag': pos,
                'features': morpho_features
            })
        return result

    def _filter_based_on_reflexive_feature(self, synonyms, tokens, morpho_tags):
        ziped = zip(synonyms, tokens, morpho_tags)
        #res = []
        #for syns, token, morpho_tag in ziped:
        #    if syns and morpho_tag['pos_tag'] == 'VERB':
        #        filtered = list(filter(lambda x: self.reflexive_exp.fullmatch(x) == self.reflexive_exp.fullmatch(token), syns))
        #        res.append(filtered)
        #    else:
        #        res.append(syns)
        #return res
        return [list(filter(lambda x: self.reflexive_exp.fullmatch(x) == self.reflexive_exp.fullmatch(token), syns))
                if syns and morpho_tag['pos_tag'] == 'VERB' else syns
                for syns, token, morpho_tag in ziped]

    def transform_sentence(self, tokens: List[str], morpho_tags: str) -> List[str]:
        morpho_tags = self._transform_morpho_tags_in_dict(morpho_tags)
        if self.lang == 'eng':
            self._unite_phrasal_verbs(tokens, morpho_tags)
        filter_res = self.word_filter.filter_words(tokens, morpho_tags)
        cases = self._get_cases(tokens, filter_res)
        lemmas = self._get_lemmas(tokens, morpho_tags, filter_res)
        synonyms = self._get_synonyms(lemmas, morpho_tags, filter_res)
        synonyms = self._inflect_synonyms(synonyms, morpho_tags, filter_res)
        synonyms = self._filter_none_value(synonyms)
        if self.lang == 'rus':
            synonyms = self._filter_based_on_reflexive_feature(synonyms, tokens, morpho_tags)
        synonyms = self._rest_cases(synonyms, cases, filter_res)
        if self.lang == 'eng':
            self._disunit_pharasal_verbs(synonyms, filter_res)
        candidates = self._insert_source_tokens(synonyms, tokens, filter_res)
        candidates = self._transform_for_kenlm_elector(candidates, filter_res)
        return candidates

    def __call__(self, batch_tokens: List[List[str]], batch_morpho_tags: List[str]):
        transformed = [self.transform_sentence(tokens, morpho_tags) for tokens, morpho_tags in zip(batch_tokens, batch_morpho_tags)]
        return transformed



