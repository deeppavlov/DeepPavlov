from deeppavlov.models.augmentation.utils.inflector import RuInflector
from deeppavlov.models.augmentation.utils.inflector import EnInflector
from deeppavlov.models.augmentation.utils.lettercaser import Lettercaser
from deeppavlov.models.augmentation.utils.thesaurus_wrapper import RuThesaurus
from deeppavlov.models.augmentation.utils.thesaurus_wrapper import EnThesaurus
from deeppavlov.models.augmentation.utils.word_filter import RuWordFilter
from deeppavlov.models.augmentation.utils.word_filter import EnWordFilter
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from logging import getLogger
from itertools import zip_longest
from typing import List
from math import log as ln


from deeppavlov.models.spelling_correction.electors.kenlm_elector import KenlmElector
from deeppavlov import build_model, configs

logger = getLogger(__name__)

@register("thesaurus_augmentation")
class ThesaurusAug(Component):
    """Component for augmentation, based on replacing words with synonyms from thesaurus
    Args:
        lang: lang of text, 'eng' for english, 'rus' for russian
        penalty_for_source_token: [0, 1] penalty for using source token for language model
        replace_freq: [0,1] frequence of replacing tokens, it calculates respecting to tokens that passed other filters
        isalpha_only: replace only isalpha tokens
        not_replaced_tokens: list of tokens that should not be replaced
        with_source_token: source tokens is synonyms for itself or not
        cases: Lettercaser arg
        default_case: Lettercaser arg
        replaced_pos_tags: WordFilter arg
        ru_thes_dirpath: path which contains russian thesaurus 'RuThes-Lite2' in csv format
        en_classical_pluralize: EnInflector arg
    Attributes:
        inflector: it inflects tokens and defines lemma form
        thesaurus: interface for thesaurus, e.g. ruthes-lite2 for russian, wordnet for english
        word_filter: it decides which token can be replaced
        lettercaser: it defines lettercases and restore them
        penalty_for_source_token: [0, 1] penalty for using source token for language model
        with_source_token: source tokens is synonyms for itself or not
        lang: lang of text, 'eng' for english, 'rus' for russian
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
        if self.lang == 'rus':
            self.morpho_tagger = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus, download=True)
        if self.lang == 'eng':
            self.morpho_tagger = build_model(configs.morpho_tagger.UD2_0.morpho_en, download=True)

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

    def _get_cases(self, tokens, filter_res):
        ziped = zip(tokens, filter_res)
        return [self.lettercaser.get_case(token) if not_filtered else None for token, not_filtered in ziped]

    def _get_lemmas(self, tokens, morpho_tags, filter_res):
        ziped = zip(tokens, morpho_tags, filter_res)
        return [self.inflector.get_lemma_form(token, morpho_tag)
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
        return [list(map(lambda syn: self.lettercaser.put_in_case(syn, case), syns))
                if not_filtered else None for syns, case, not_filtered in ziped]

    def _insert_source_tokens(self, synonyms, source_tokens, filter_res):
        ziped = zip(synonyms, source_tokens, filter_res)
        if self.with_source_token:
            return [[source_token] + syns if (not_filtered and syns) else [source_token] for syns, source_token, not_filtered in ziped]
        else:
            return [syns if (not_filtered and syns) else [source_token] for syns, source_token, not_filtered in ziped]

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

    def _disunit_phrasal_verbs(self, synonyms, filter_res):
        ziped = zip(synonyms, filter_res)
        [list(map(lambda x: x.replace("_", " "), syns)) if not_filtered else None for syns, not_filtered in ziped]

    def __get_morpho_features_from_morpho_tag(self, features: str):
        if features == '_':
            return {}
        features = features.split('|')
        features = list(map(lambda x: tuple(x.split('=')), features))
        return dict(features)

    def _transform_morpho_tags(self, morpho_tags):
        splited, res = morpho_tags.split('\n'), []
        for token_morpho in splited[:-1]:
            token_morpho = token_morpho.split('\t')
            token_morpho_dict = {}
            token_morpho_dict.update({'source_token': token_morpho[1]})
            token_morpho_dict.update({'pos_tag': token_morpho[2]})
            token_morpho_dict.update({'features': self.__get_morpho_features_from_morpho_tag(token_morpho[3])})
            res.append(token_morpho_dict)
        return res

    def _filter_none_value(self, synonyms):
        return [list(filter(lambda x: x is not None, syns)) if syns is not None else None for syns in synonyms]

    def transform_sentence(self, tokens: List[str], morpho_tags: str) -> List[str]:
        morpho_tags = self._transform_morpho_tags(morpho_tags)
        if self.lang == 'eng':
            self._unite_phrasal_verbs(tokens, morpho_tags)
        filter_res = self.word_filter.filter_words(tokens, morpho_tags)
        cases = self._get_cases(tokens, filter_res)
        lemmas = self._get_lemmas(tokens, morpho_tags, filter_res)
        synonyms = self._get_synonyms(lemmas, morpho_tags, filter_res)
        synonyms = self._inflect_synonyms(synonyms, morpho_tags, filter_res)
        synonyms = self._filter_none_value(synonyms)
        synonyms = self._rest_cases(synonyms, cases, filter_res)
        if self.lang == 'eng':
            self._disunit_pharasal_verbs(synonyms, filter_res)
        candidates = self._insert_source_tokens(synonyms, tokens, filter_res)
        candidates = self._transform_for_kenlm_elector(candidates, filter_res)
        return candidates

    def __call__(self, batch_tokens: List[List[str]]):
        batch_morpho_tags = self.morpho_tagger(batch_tokens)
        transformed = [self.transform_sentence(tokens, morpho_tags) for tokens, morpho_tags in zip(batch_tokens, batch_morpho_tags)]
        return transformed

if __name__ == '__main__':
    from nltk import word_tokenize
    test = word_tokenize("""Широко распространён и довольно обычен в Европе и на юге Сибири, завезён в Северную Америку, где активно осваивает новые пространства.""")
    #test = word_tokenize("""смотри там случайно в аспирантуру не попади наука никого до добра не доводила""".lower())
    #test = word_tokenize("""зря вы с этой хуйней шутите я много историй наблюдал в режиме рил времени по работе с русскими которые разбивались на них""".lower())
    #test = word_tokenize("""Размер, цвет и форма могут значительно различаться в зависимости от вида или сорта, но чаще всего они имеют продолговатую цилиндрическую или трёхгранную форму, выпрямленную либо закруглённую. Длина плода варьирует в пределах от 3 до 40 см, толщина — от 2 до 8 см. Цвет кожицы может быть жёлтым, зелёным, красным или даже серебристым. Мякоть белая, кремовая, жёлтая или оранжевая.""".lower())
    #test = word_tokenize("""Одно из наиболее значительных произведений в творчестве Андрея Тарковского, который говорил, что в нём он «легально коснулся трансцендентного». Производство фильма сопровождалось множеством проблем и заняло около трёх лет. При проявке плёнки практически полностью испорчен по техническим причинам первый вариант, и картину переснимали трижды, с тремя разными операторами и художниками- постановщиками.""".lower())
    #test = word_tokenize("""В Москве 15-летний подросток, который несколько дней назад был госпитализирован с ожогами 98% тела, умер в больнице.""".lower())
    #test = word_tokenize("""Способности он имел совершенно исключительные, обладал огромной памятью, отличался ненасытной научной любознательностью и необычайной работоспособностью... Воистину, это была ходячая энциклопедия...""".lower())
    t = ThesaurusAug(lang='rus', ru_thes_dirpath='/home/azat/.deeppavlov/downloads/ruthes_lite2', penalty_for_source_token=0.95)
    kenlm = KenlmElector('/home/azat/.deeppavlov/models/lms/ru_wiyalen_no_punkt.arpa.binary')
    res = t([test])
    print(res)
    print(kenlm(res))


