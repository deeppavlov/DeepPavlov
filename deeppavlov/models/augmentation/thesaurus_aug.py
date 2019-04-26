from utils import CaseSaver
from utils import RuInflector
from utils import RuThesaurus
from utils import EnInflector
from utils import EnThesaurus
from utils import EnWordFilter
from utils import RuWordFilter
from deeppavlov.models.spelling_correction.electors.kenlm_elector import KenlmElector
from nltk import pos_tag as nltk_pos_tagger

class ThesaurusAug:
    """
    """

    def __init__(self,
                 threshold_of_lm_confidence: float=0.3,
                 penalty_for_source_token: float=0.5,
                 kenlm_model_path = None,
                 kenlm_beam_size: int=4,
                 replace_freq: float=1.0,
                 isalpha_only: bool=True,
                 not_replaced_tokens: List[str]=None,
                 with_source_token: bool=True,
                 cases: dict=None,
                 default_case = None,
                 lang: str,
                 replaced_pos_tags: List[str]=None,
                 ru_thes_dirpath = None,
                 ru_is_replace_numeral_adjective: bool=True,
                 en_classical_pluralize: bool=True,
                 en_is_replace_proper_noun: bool=True,
                 en_is_replace_modal_verb: bool=True)
        self.kenlm_elector = KenlmElector(load_path=kenlm_model_path,
                                          beam_size=kenlm_beam_size)
        self.case_saver = CaseSaver(cases=cases,
                                    default_case=default_case)
        assert lang in ['rus', 'eng'], "it supports only russian and english languages, e.g. 'rus' for russian, 'eng' for english"
        if lang == 'rus':
            self.inlfector = RuInflector()
            self.thesaurus = RuThesaurus(dir_path=ru_thes_dirpath,
                                         with_source_token=with_source_token)
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
        self.threshold_of_lm_confidence = threshold_of_lm_confidence
        self.lang = lang
        self.penalty_for_source_token = penalty_for_source_token

        def transform_sentece(self, tokens: List[str]) -> List[str]:
            self.case_saver.save_cases(tokens)
            tokens_pos_tags = nltk_pos_tagger(tokens, lang=self.)
            filter_result = self.word_filter