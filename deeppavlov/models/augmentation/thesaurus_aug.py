from deeppavlov.models.augmentation.utils import Lettercaser
from deeppavlov.models.augmentation.utils import RuInflector
from deeppavlov.models.augmentation.utils import RuThesaurus
from deeppavlov.models.augmentation.utils import EnInflector
from deeppavlov.models.augmentation.utils import EnThesaurus
from deeppavlov.models.augmentation.utils import EnWordFilter
from deeppavlov.models.augmentation.utils import RuWordFilter
from deeppavlov.models.spelling_correction.electors.kenlm_elector import KenlmElector
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from logging import getLogger
from nltk import pos_tag as nltk_pos_tagger
from itertools import zip_longest
from typing import List
from math import log as ln

logger = getLogger(__name__)

@register("thesaurus_augmentation")
class ThesaurusAug(Component):
    """Component for augmentation, based on replacing words with synonyms from thesaurus
    Args:
        lang: lang of text, 'eng' for english, 'rus' for russian
        penalty_for_source_token: [0, 1] penalty for using source token for language model
        kenlm_model_path: path to kenlm model file
        kenlm_beam_size: beam size of kenlm_elector
        replace_freq: [0,1] frequence of replacing tokens, it calculates respecting to tokens that passed other filters
        isalpha_only: replace only isalpha tokens
        not_replaced_tokens: list of tokens that should not be replaced
        with_source_token: source tokens is synonyms for itself or not
        cases: Lettercaser arg
        default_case: Lettercaser arg
        replaced_pos_tags: WordFilter arg
        ru_thes_dirpath: path to russian thesaurus 'RuThes-Lite2'
        ru_is_replace_numeral_adjective: RuWordFilter arg
        en_classical_pluralize: EnInflector arg
        en_is_replace_proper_noun: EnWordFilter arg
        en_is_replace_modal_verb: EnWordFilter arg
        force_inflect: RuInflector arg
    Attributes:
        inflector: it inflects tokens, defines lemma form and morpho tags
        thesaurus: interface for thesaurus
        word_filter: it decides which token can be replaced
        kenlm_elector: Kenlm object
        lettercaser: it defines lettercases and restore them
        penalty_for_source_token: [0, 1] penalty for using source token for language model
        with_source_token: source tokens is synonyms for itself or not
        force_inflect: RuInflector arg
        lang: lang of text, 'eng' for english, 'rus' for russian
    """

    def __init__(self,
                 lang: str,
                 penalty_for_source_token: float=0.0001,
                 kenlm_model_path: str=None,
                 kenlm_beam_size: int=4,
                 replace_freq: float=1.0,
                 isalpha_only: bool=True,
                 not_replaced_tokens: List[str]=None,
                 with_source_token: bool=True,
                 cases: dict=None,
                 default_case=None,
                 replaced_pos_tags: List[str]=None,
                 ru_thes_dirpath = None,
                 ru_is_replace_numeral_adjective: bool=True,
                 en_classical_pluralize: bool=True,
                 en_is_replace_proper_noun: bool=True,
                 en_is_replace_modal_verb: bool=True,
                 force_inflect: bool=True, *args, **kwargs):
        assert lang in ['rus', 'eng'], "it supports only russian and english languages, e.g. 'rus' for russian, 'eng' for english"
        if lang == 'rus':
            self.inflector = RuInflector()
            self.thesaurus = RuThesaurus(dir_path=ru_thes_dirpath, with_source_token=with_source_token)
            self.word_filter = RuWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags,
                                            is_replace_numeral_adjective=ru_is_replace_numeral_adjective)
        elif lang == 'eng':
            self.inflector = EnInflector(classical_pluralize=en_classical_pluralize)
            self.thesaurus = EnThesaurus(with_source_token=with_source_token)
            self.word_filter = EnWordFilter(replace_freq=replace_freq,
                                            isalpha_only=isalpha_only,
                                            not_replaced_tokens=not_replaced_tokens,
                                            replaced_pos_tags=replaced_pos_tags,
                                            is_replace_proper_noun=en_is_replace_proper_noun,
                                            is_replace_modal_verb=en_is_replace_modal_verb)
        self.kenlm_elector = KenlmElector(load_path=kenlm_model_path, beam_size=kenlm_beam_size)
        self.lettercaser = Lettercaser(cases=cases, default_case=default_case)
        self.penalty_for_source_token = penalty_for_source_token
        self.with_source_token = with_source_token
        self.force_inflect = force_inflect
        self.lang = lang

    def _unite_phrasal_verbs(self, tokens, pos_tags):
        res_tokens, res_pos_tags, is_pass_next_token = [], [], False
        for cur_token, next_token, cur_pos, next_pos in zip_longest(tokens, tokens[1:], pos_tags, pos_tags[1:]):
            if is_pass_next_token:
                is_pass_next_token = False
                continue
            elif cur_pos.startswith('V') and next_pos == 'RP':
                res_tokens.append("_".join([cur_token, next_token]))
                res_pos_tags.append(cur_pos)
                is_pass_next_token = True
            else:
                res_tokens.append(cur_token)
                res_pos_tags.append(cur_pos)
        return res_tokens, res_pos_tags

    def _get_cases(self, tokens, filter_res):
        ziped = zip(tokens, filter_res)
        return [self.lettercaser.get_case(token) if not_filtered else None for token, not_filtered in ziped]

    def _get_morpho_tags(self, tokens, pos_tags, filter_res):
        ziped = zip(tokens, pos_tags, filter_res)
        return [self.inflector.get_morpho_tag(token, pos_tag) if not_filtered else None for token, pos_tag, not_filtered in ziped]

    def _get_lemmas(self, tokens, filter_res):
        ziped = zip(tokens, filter_res)
        return [self.inflector.get_lemma_form(token) if not_filtered else None for token, not_filtered in ziped]

    def _get_synonyms(self, lemmas, pos_tags, filter_res):
        ziped = zip(lemmas, pos_tags, filter_res)
        return [self.thesaurus.get_syn(lemma, pos) if not_filtered else None for lemma, pos, not_filtered in ziped]

    def _inflect_synonyms(self, synonyms, morpho_tags, filter_res):
        ziped = zip(synonyms, morpho_tags, filter_res)
        return [list(map(lambda syn: self.inflector.inflect_token(syn, morpho), syns)) if not_filtered else None for syns, morpho, not_filtered in ziped]

    def _rest_cases(self, synonyms, cases, filter_res):
        ziped = zip(synonyms, cases, filter_res)
        return [list(map(lambda syn: self.lettercaser.put_in_case(syn, case), syns)) if not_filtered else None for syns, case, not_filtered in ziped]

    def _insert_filtered_tokens(self, synonyms, source_tokens, filter_res):
        ziped = zip(synonyms, source_tokens, filter_res)
        return [syns if (not_filtered and syns) else [source_token] for syns, source_token, not_filtered in ziped]

    def __insert_prob(self, syns):
        return [(1.0/len(syns), syn) for syn in syns]

    def __penalty_for_source_token(self, syns_with_prob):
       syns_with_prob[0] = (ln(self.penalty_for_source_token) * syns_with_prob[0][0], syns_with_prob[0][1])
       return syns_with_prob

    def _transform_for_kenlm_elector(self, candidates):
       with_prob = [self.__insert_prob(candidate) for candidate in candidates]
       if self.with_source_token:
           return [self.__penalty_for_source_token(candidate) for candidate in with_prob]
       return with_prob

    def _disunit_phrasal_verbs(self, synonyms, filter_res):
        ziped = zip(synonyms, filter_res)
        [list(map(lambda x: x.replace("_", " "), syns)) if not_filtered else None for syns, not_filtered in ziped]

    def _get_pos_tags(self, tokens):
        pos_tags = nltk_pos_tagger(tokens, lang=self.lang)
        return list(map(lambda x: x[1], pos_tags))

    def _filter_none_value(self, synonyms):
        return [list(filter(lambda x: x is not None, syns)) if syns is not None else None for syns in synonyms]

    def transform_sentence(self, tokens: List[str]) -> List[str]:
        pos_tags = self._get_pos_tags(tokens)
        if self.lang == 'eng':
            self._unite_phrasal_verbs(tokens, pos_tags)
        filter_res = self.word_filter.filter_words(tokens, pos_tags)
        cases = self._get_cases(tokens, filter_res)
        morpho_tags = self._get_morpho_tags(tokens, pos_tags, filter_res)
        lemmas = self._get_lemmas(tokens, filter_res)
        synonyms = self._get_synonyms(lemmas, pos_tags, filter_res)
        synonyms = self._inflect_synonyms(synonyms, morpho_tags, filter_res)
        synonyms = self._filter_none_value(synonyms)
        synonyms = self._rest_cases(synonyms, cases, filter_res)
        if self.lang == 'eng':
            self._disunit_pharasal_verbs(synonyms, filter_res)
        candidates = self._insert_filtered_tokens(synonyms, tokens, filter_res)
        candidates = self._transform_for_kenlm_elector(candidates)
        return self.kenlm_elector._infer_instance(candidates)

    def __call__(self, batch: List[List[str]]):
        transformed = [self.transform_sentence(tokens) for tokens in batch]
        print(transformed)
        return transformed + batch

if __name__ == '__main__':
    from nltk import word_tokenize
    test = word_tokenize("""Широко распространён и довольно обычен в Европе и на юге Сибири,
                            завезён в Северную Америку, где активно осваивает новые пространства.""")
    t = ThesaurusAug(lang='rus',
                     kenlm_model_path='/Users/sultanovar/.deeppavlov/models/lms/ru_wiyalen_no_punkt.arpa.binary',
                     ru_thes_dirpath='/Users/sultanovar/.deeppavlov/downloads/ruthes_lite2')
    print(t([test]))


