import pymorphy2
from typing import List, Set
import pattern.en as en
from pathlib import Path
from nltk.corpus import wordnet as wn
import pandas as pd
from numpy.random import sample
from itertools import repeat
from nltk.stem import WordNetLemmatizer
from itertools import dropwhile


class Lettercaser:
    """It defines lettercases of tokens and can restore them.
    By default it detects only ['lower', 'upper', 'capitalize'] lettercases,
    but there is opportunity to expand that list with 'cases' argument
    Args:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string when lettercase was not be detected
    Attributes:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string when lettercase was not be detected
    """

    def __init__(self, cases: dict = None, default_case = None):
        if default_case is None:
            self.default_case = lambda x: x.lower()
        else:
            self.default_case = default_case
        if cases is None:
            self.cases = {
                "lower": lambda x: x.lower(),
                "capitalize": lambda x: x.capitalize(),
                "upper": lambda x: x.upper()
            }
        else:
            self.cases = cases

    def get_case(self, token):
        """It detects case of token with 'cases' attribute
        Args:
            token: token lettercases of that will be detected
        """
        for case in self.cases:
            if token == self.cases[case](token):
                return case
        return None

    def put_in_case(self, token: str, case: str):
        """It restore lettercases of tokens according to 'case' arg,
        if lettercase was not detected (case==None), 'default_case' func would be used
        Args:
            tokens: token that will be put in case
            case: name of lettercase
        Return:
            tokens in certain lettercase
            if lettercase was not detected then 'default_case'would be used
        """
        if case is None:
            return self.default_case(token)
        return self.cases[case](token)


class RuInflector:
    """Works only with noun verb adjective and adverb
    """

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def _get_pymorphy_postag(self, morpho_tag):
        if morpho_tag['pos_tag'] == 'NOUN':
            return 'NOUN'

        elif morpho_tag['pos_tag'] == 'ADJ' and\
             morpho_tag['features'].get('Variant') != 'Short' and\
             morpho_tag['features'].get('Degree') == 'Pos':
            return 'ADJF'

        elif morpho_tag['pos_tag'] == 'ADJ' and \
             morpho_tag['features'].get('Variant') == 'Short' and \
             morpho_tag['features'].get('Degree') == 'Pos':
            return 'ADJS'

        elif morpho_tag['features'].get('Degree') == 'Cmp':
            return 'COMP'

        elif morpho_tag['pos_tag'] == 'VERB' and\
             morpho_tag['features'].get('VerbForm') != 'Inf' and\
             morpho_tag['features'].get('VerbForm') != 'Part' and\
             morpho_tag['features'].get('VerbForm') != 'Conv':
            return 'VERB'

        elif morpho_tag['pos_tag'] == 'VERB' and\
             morpho_tag['features'].get('VerbForm') == 'Inf':
            return 'INFN'

        elif morpho_tag['pos_tag'] == 'VERB' and\
             morpho_tag['features'].get('VerbForm') == 'Part' and\
             morpho_tag['features'].get('Variant') != 'Short':
            return 'PRTF'

        elif morpho_tag['pos_tag'] == 'VERB' and\
             morpho_tag['features'].get('VerbForm') == 'Part' and\
             morpho_tag['features'].get('Variant') == 'Short':
            return 'PRTS'

        elif morpho_tag['pos_tag'] == 'VERB' and\
             morpho_tag['features'].get('VerbForm') == 'Conv':
            return 'GRND'

        elif morpho_tag['pos_tag'] == 'NUM':
            return 'NUMR'

        elif morpho_tag['pos_tag'] == 'ADV' and\
             morpho_tag['features'].get('Degree') != 'Cmp':
            return 'ADVB'



    def get_morpho_tag(self, token: str, pos_tag=None) -> pymorphy2.tagset.OpencorporaTag:
        """Return morpho-form of token
        Args:
            token: token that will be morpho-analysed
        Return:
            result of morph-analyse in pymorphy2.tagset.OpencorporaTag form
        """
        return self.morph.parse(token)[0].tag

    def get_lemma_form(self, token: str, morpho_tag = None) -> str:
        """Return lemma-form of given token
        """
        return self.morph.parse(token)[0].normal_form

    def inflect_token(self, token: str, morpho_tag: pymorphy2.tagset.OpencorporaTag, force: bool=True) -> str:
        """It inflects token in certain morpho-form
        Args:
            token: token that will be inflected
            morpho_tag: morpho_tag in pymorphy2.tagset.OpencorporaTag form
            force: to use only main information of 'morpho_tag' or all
        Return:
            Inflected word
        """
        if force:
            morpho_tag = self.extract_morpho_requirements(morpho_tag)
        else:
            morpho_tag = morpho_tag.grammemes
        inflected = self.morph.parse(token)[0].inflect(morpho_tag)
        if inflected is None:
            return None
        return inflected.word

    @staticmethod
    def extract_morpho_requirements(tag: pymorphy2.tagset.OpencorporaTag) -> Set:
        #Not all morphems need for inflection
        if tag.POS in ["NOUN"]:
            keys = {"NOUN", "case", "number", "animacy", "person", "gender"}
        elif tag.POS in ["ADJF"]:
            keys = {"ADJF", "case", "gender", "number"}
        elif tag.POS == "ADJS":
            keys = {"ADJS", "gender", "number"}
        elif tag.POS == "VERB":
            keys = {"VERB", "mood", "tense", "aspect", "person", "voice", "gender", "number"}
        elif tag.POS == "INFN":
            keys = {"INFN", "aspect"}
        elif tag.POS in ["PRTF"]:
            keys = {"PRTF", "case", "gender", "number", "voice", "aspect"}
        elif tag.POS == "PRTS":
            keys = {"PRTS", "gender", "number", "voice", "aspect"}
        elif tag.POS == "GRND":
            keys = {"GRND", "voice", "aspect"}
        elif tag.POS == "COMP":
            keys = {"COMP"}
        else:
            keys = {}
        values = {(getattr(tag, key) if key.islower() else key) for key in keys}
        return {x for x in values if x is not None}


class RuThesaurus:
    """Class that finds synonyms, for russian language.
    Args:
        with_source_token: wheither source symbol is considered as synonyms to itself
        dir_path: path to ruthes_lite2 thesaurus
    Attributes:
        with_source_token: wheither source symbol is considered as synonyms to itself
        dir_path: path to ruthes_lite2 thesaurus
        synonyms_data: pd.DataFrame that contains synonyms relations
        text_entry_data: pd.DataFrame that contains lemma and entry_id
    """

    def __init__(self, dir_path, with_source_token: bool = False):
        self.dir_path = Path(dir_path)
        required_files = (f"{dt}.csv" for dt in ('synonyms', 'text_entry'))
        if not all(Path(dir_path, f).exists() for f in required_files):
            raise FileNotFoundError(f"""Files: synonyms.csv, text_entry.csv was not found,
                                        in specified path - {self.dir_path}""")
        self.with_source_token = with_source_token
        self.synonyms_data = pd.read_csv(self.dir_path / "synonyms.csv", index_col=0)
        self.text_entry_data = pd.read_csv(self.dir_path / "text_entry.csv", index_col=0)
        self.to_wn_postag = {
            'NOUN': ['N', 'Num', 'NumG', 'Prtc'],
            'VERB': ['V'],
            'ADJ':  ['Adj', 'AdjGprep', 'Num', 'Pron'],
            'ADV':  ['Adv', 'Prdc', 'PrepG']
        }

    def _find_synonyms(self, lemma: str, morpho_tag) -> List[str]:
        lemma = lemma.upper()
        entry_id_set = set(self.text_entry_data[self.text_entry_data['lemma'] == lemma]['entry_id'])
        concept_id_set = set()
        for entry_id in entry_id_set:
            concept_id_set.update(set(self.synonyms_data[self.synonyms_data['entry_id'] == entry_id]['concept_id']))
        syn_entry_id_set = set()
        for concept_id in concept_id_set:
            syn_entry_id_set.update(set(self.synonyms_data[self.synonyms_data['concept_id'] == concept_id]['entry_id']))
        synlist = list()
        for syn_entry_id in syn_entry_id_set:
            filter_by_pos_tag = self.text_entry_data['pos'].isin(self.to_wn_postag[morpho_tag['pos_tag']])
            filter_by_entry_id = (self.text_entry_data['entry_id'] == syn_entry_id)
            synonyms_table = self.text_entry_data[filter_by_pos_tag & filter_by_entry_id]
            synlist += list(synonyms_table['lemma'])
        return synlist

    def _filter(self, synlist: List[str], source_lemma: str) -> List[str]:
        filtered_syn = set(filter(lambda x: len(x.split('_')) == 1, synlist))
        filtered_syn.discard(source_lemma)
        if self.with_source_token:
            return [source_lemma] + list(filtered_syn)
        else:
            return list(filtered_syn)

    def get_syn(self, lemma: str, morpho_tag: str) -> List[str]:
        """It returns synonyms for certain word
        Args:
            lemma: word for that it will search synonyms
            pos_tag: pos_tag in nltk.pos_tag format of 'lemma'
        Return:
             List of synonyms
        """
        lemma = lemma.upper()
        synonyms = self._find_synonyms(lemma, morpho_tag)
        synonyms = self._filter(synonyms, lemma)
        return synonyms


class EnInflector:
    """Class that morpho-analyses given token and inflects
    random token into certain morpho-form, for english language.
    It is based on pattern.en library. Phrasal verbs should be linked with '_' symbol.
    Args:
        classical_pluralize: arg from pattern.en.inflect.pluralize function
    Attribute:
        classical_pluralize: arg from pattern.en.inflect.pluralize function
    """

    def __init__(self, classical_pluralize: bool=True):
        self.classical_pluralize = classical_pluralize
        self.lemmatizer = WordNetLemmatizer()
        self.ud_to_en_tense = {
            'Past': 'past',
            'Pres': 'present',
            'Fut': 'future'
        }
        self.ud_to_en_person = {
            '1': 1,
            '2': 2,
            '3': 3
        }
        self.ud_to_en_number = {
            'Sing': 'singular',
            'Plur': 'plural'
        }
        self.ud_to_en_mood = {
            'Ind': 'indicative',
            'Imp': 'imperative',
            'Cnd': 'conditional',
            'Sub': 'subjunctive'
        }
        self.ud_to_en_aspect = {
            'Imp': 'imperfective',
            'Perf': 'perfective',
            'Prog': 'progressive'
        }

    def get_lemma_form(self, token: str, morpho_tag):
        if morpho_tag['pos_tag'] == 'VERB':
            return self.get_lemma_verb(token)
        return self.lemmatizer.lemmatize(token)

    def get_lemma_verb(self, token: str):
        """"""
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join([en.lemma(splited[0])] + splited[1:])
        return en.lemma(token)

    def pluralize(self, token, morpho_tag):
        """"""
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join(splited[:-1] + en.pluralize(splited[-1],
                                                        self.to_en_pos(morpho_tag['pos_tag']),
                                                        self.classical_pluralize))
        return en.pluralize(token, self.to_en_pos(morpho_tag['pos_tag']), self.classical_pluralize)

    def singularize(self, token, morpho_tag):
        """"""
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join(splited[:-1] + en.singularize(splited[-1], self.to_en_pos(morpho_tag['pos_tag'])))
        return en.pluralize(token, self.to_en_pos(morpho_tag['pos_tag']), self.classical_pluralize)

    def _get_verb_tense(self, morpho_tag):
        return self.ud_to_en_tense.get(morpho_tag['features'].get('tense'), 'INFINITIVE')

    def _get_verb_person(self, morpho_tag):
        return self.ud_to_en_person.get(morpho_tag['features'].get('person'), None)

    def _get_verb_number(self, morpho_tag):
        return self.ud_to_en_number.get(morpho_tag['features'].get('number'), None)

    def _get_verb_mood(self, morpho_tag):
        return self.ud_to_en_mood.get(morpho_tag['features'].get('mood'), None)

    def _get_verb_aspect(self, morpho_tag):
        aspect = self.ud_to_en_aspect.get(morpho_tag['features'].get('aspect'), None)
        if aspect is None:
            if morpho_tag['features'].get('VerbForm') == 'Part':
                aspect = 'progressive'
            elif morpho_tag['features'].get('VerbForm') == 'Fin':
                aspect = 'imperfective'
        return aspect

    def _transform_to_pattern_en_form(self, morpho_tag):
        return (self._get_verb_tense(morpho_tag),
                self._get_verb_person(morpho_tag),
                self._get_verb_number(morpho_tag),
                self._get_verb_mood(morpho_tag),
                self._get_verb_aspect(morpho_tag))

    def _tense_similarity(self, first_tense, second_tense):
        if first_tense[0] != second_tense[0]:
            return 0
        if first_tense[-1] != second_tense[-1]:
            return 0
        return sum([int(x == y) for x, y in zip(first_tense, second_tense)])

    def _sort_and_filter_candidates(self, source, candidates):
        candidates = [(cand, self._tense_similarity(source, cand)) for cand in candidates]
        candidates = list(filter(lambda x: x[1] > 0, candidates))
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = map(lambda x: x[0], candidates)
        return candidates

    def _inflect_verb(self, token, morpho_tag):
        candidate_tenses = en.tenses(morpho_tag['source_token']) #maybe all possible tenses?
        morpho_tense = self._transform_to_pattern_en_form(morpho_tag)
        candidate_tenses = list(self._sort_and_filter_candidates(morpho_tense, candidate_tenses))
        tense_for_inflection = list(dropwhile(lambda cand: en.conjugate(token, cand) is None, candidate_tenses))
        lemma = self.get_lemma_verb(token)
        if tense_for_inflection is not []:
            splited = lemma.split('_')
            if len(splited) > 1:
                return "_".join([en.conjugate(splited[0], tense_for_inflection[0])] + splited[1:])
            else:
                return en.conjugate(lemma, tense_for_inflection[0])
        return None

    def inflect_token(self, token, morpho_tag, force: bool = False):
        if morpho_tag['pos_tag'] in ['NOUN', 'PROPN']:
            if morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, morpho_tag)
            if morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, morpho_tag)
        if morpho_tag['pos_tag'] in ['ADJ']:
            if morpho_tag['features'].get('Degree') == 'Cmp':
                token = en.inflect.comparative(token)
            if morpho_tag['features'].get('Degree') == 'Sup':
                token = en.inflect.superlative(token)
            if morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, morpho_tag)
            if morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, morpho_tag)
        if morpho_tag['pos_tag'] in ['VERB']:
            token = self._inflect_verb(token, morpho_tag)
        return token


class EnThesaurus:
    """Class that finds synonyms, for english language.
    Args:
        with_source_token: wheither source symbol is considered as synonyms to itself
    Attributes:
        with_source_token: wheither source symbol is considered as synonyms to itself
    """

    def __init__(self, with_source_token: bool=False, without_phrase: bool=False):
        self.with_source_token = with_source_token
        self.without_phrase = without_phrase
        self.to_wn_postags = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
            'ADJ':  wn.ADJECTIVE,
            'ADV':  wn.ADVERB
        }

    def _find_synonyms(self, lemma, morpho_tag):
        wn_synonyms = wn.synsets(lemma, pos=self.to_wn_postags(morpho_tag['pos_tag']))
        if morpho_tag['pos_tag'] == 'ADJ':
            wn_synonyms = wn_synonyms.extend(wn.synsets(lemma, pos='s'))
        lemmas = sum((map(lambda x: x.lemmas(), wn_synonyms)), [])
        synonyms = set(map(lambda x: x.name(), lemmas))
        return synonyms

    def _filter(self, synlist: List[str], source_lemma: str) -> List[str]:
        filtered_syn = set(filter(lambda x: len(x.split('_')) == 1, synlist)) if self.without_phrase else set(synlist)
        filtered_syn.discard(source_lemma)
        if self.with_source_token:
            return [source_lemma] + list(filtered_syn)
        else:
            return list(filtered_syn)

    def get_syn(self, lemma: str, morpho_tag) -> List[str]:
        """It returns synonyms for certain word
        Args:
            lemma: word for that it will search synonyms
            pos_tag: pos_tag in nltk.pos_tag format of 'lemma'
        Return:
             List of synonyms, if with_source_token == True then source_token will be placed in the begin of list
        """
        synonyms = self._find_synonyms(lemma, morpho_tag)
        synonyms = self._filter(synonyms, lemma)
        return synonyms


class WordFilter:
    """Class that decides which tokens should not be replaced
    Args:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
    Attributes:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
    """

    def __init__(self,
                 replace_freq: float,
                 isalpha_only: bool,
                 not_replaced_tokens: List[str]):
        self.replace_freq = replace_freq
        self.isalpha_only = isalpha_only
        if not_replaced_tokens is None:
            self.not_replaced_tokens = []
        else:
            self.not_replaced_tokens = not_replaced_tokens

    def filter_isalpha_only(self, tokens):
        if self.isalpha_only:
            return map(lambda x: x.isalpha(), tokens)
        else:
            return repeat(True, len(tokens))

    def filter_not_replaced_token(self, tokens):
        return map(lambda x: not x in self.not_replaced_tokens, tokens)

    def filter_based_on_pos_tag(self, tokens, pos_tags):
        pass

    def filter_frequence(self, prev_decision):
        return map(lambda x: sample() < self.replace_freq if x else x, prev_decision)

    def filter_united(self, tokens, pos_tags):
        return list(map(lambda x, y, z: all([x,y,z]),
                        self.filter_based_on_pos_tag(tokens, pos_tags),
                        self.filter_not_replaced_token(tokens),
                        self.filter_isalpha_only(tokens)))

    def filter_words(self, tokens, pos_tags):
        """It filters tokens based on replace_freq, isalpha_only, not_replaced_token and pos_tags of tokens
        Args:
            tokens: tokens that will be filtered
        Return:
            List of boolean values,
            'False' for tokens that should not be replaced,
            'True' for tokens that can be replaced
        """
        filtered = self.filter_united(tokens, pos_tags)
        return list(self.filter_frequence(filtered))


class EnWordFilter(WordFilter):
    """Class that decides which tokens should not be replaced, for english language
    Args:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
        replaced_pos_tags: List of pos_tags that can be replaced,
                           e.g. 'n' for Noun, 'v' for Verb, 'a' for Adjective, 'r' for Adverb
        is_replace_proper_noun: to replace proper noun or not, based on pos_tag
        is_replace_modal_verb: to replace modal verb or not, based on pos_tag
    Attributes:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
        replaced_pos_tags: List of pos_tags that can be replaced,
                           e.g. 'n' for Noun, 'v' for Verb, 'a' for Adjective, 'r' for Adverb
        is_replace_proper_noun: to replace proper noun or not, based on pos_tag
        is_replace_modal_verb: to replace modal verb or not, based on pos_tag
    """

    def __init__(self,
                 replace_freq: float,
                 isalpha_only: bool,
                 not_replaced_tokens: List[str] = None,
                 replaced_pos_tags: List[str] = None):
        super(EnWordFilter, self).__init__(replace_freq, isalpha_only, not_replaced_tokens)
        if not_replaced_tokens is None:
            self.not_replaced_tokens = []
        self.not_replaced_tokens = not_replaced_tokens
        if replaced_pos_tags is None:
            replaced_pos_tags = ['ADJ', 'ADV', 'NOUN', 'VERB']
        self.replaced_pos_tags = replaced_pos_tags



    def filter_based_on_pos_tag(self, morpho_tags):
        """Function that filters tokens with pos_tags and rules
        Args:
            tokens: tokens that will be filtered
            pos_tags: pos_tags for 'tokens', in nltk.pos_tag format
        Return:
            List of boolean values,
            'False' for tokens that should not be replaced,
            'True' for tokens that can be replaced
        """
        prev_is_there, result = False, []
        for morpho_tag in morpho_tags:
            if morpho_tag['pos_tag'] == 'PRON' and morpho_tag['source_token'] == 'there':
                prev_is_there = True
                result.append(False)
            elif prev_is_there and morpho_tag['pos_tag'] == 'VERB':
                prev_is_there = False
                result.append(False)
            else:
                result.append(morpho_tag['pos_tag'] in self.replaced_pos_tags)
        return result


class RuWordFilter(WordFilter):
    """Class that decides which tokens should not be replaced, for russian language
    Args:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
        replaced_pos_tags: List of pos_tags that can be replaced,
                           e.g. 's' for Noun, 'v' for Verb, 'a' for Adjective, 'r' for Adverb, 'num' for Numerical
        is_replace_numeral_adjective: to replace numeral adjective or not, based on pos_tag
    Attributes:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
        replaced_pos_tags: List of pos_tags that can be replaced,
                           e.g. 's' for Noun, 'v' for Verb, 'a' for Adjective, 'r' for Adverb, 'num' for Numerical
        is_replace_numeral_adjective: to replace numeral adjective or not, based on pos_tag
    """

    def __init__(self,
                 replace_freq: float,
                 isalpha_only: bool,
                 not_replaced_tokens: List[str]=None,
                 replaced_pos_tags: List[str]=None):
        super(RuWordFilter, self).__init__(replace_freq, isalpha_only, not_replaced_tokens)
        if not_replaced_tokens is None:
            self.not_replaced_tokens = ['блин', 'люблю', 'заебись']
        if replaced_pos_tags is None:
            replaced_pos_tags = ['ADJ', 'ADV', 'NOUN', 'VERB', 'NUM']
        self.replaced_pos_tags = replaced_pos_tags

    def filter_based_on_pos_tag(self, tokens, morpho_tags):
        """Function that filters tokens with pos_tags and rules
        Args:
            tokens: tokens that will be filtered
            pos_tags: pos_tags for 'tokens', in nltk.pos_tag format
        Return:
            List of boolean values,
            'False' for tokens that should not be replaced,
            'True' for tokens that can be replaced
        """
        return list(map(lambda x: x in self.replaced_pos_tags, morpho_tags))

