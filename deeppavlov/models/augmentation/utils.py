import pymorphy2
from typing import List, Set
import pattern.en as en
from pathlib import Path
from nltk.corpus import wordnet as wn
import pandas as pd

class CaseSaver:
    """It can save lettercases of tokens and then restore them.
    By default it detects only ['lower', 'upper', 'capitalize'] lettercases,
    but there is opportunity to expand that list with 'cases' argument
    Args:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
    Attributes:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
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

    def _determine_case(self, token):
        for case in self.cases:
            if token == self.cases[case](token):
                return case
        return None

    def save_cases(self, tokens: List[str]):
        """It detects case of tokens with 'cases' attribute
        and saves them into 'saved_cases' attribute
        Args:
            tokens: List of tokens lettercases of that will be saved
        """
        self.saved_cases = [self._determine_case(token) for token in tokens]

    def _rest_case(self, token: str, case: str):
        if case is None:
            return self.default_case(token)
        return self.cases[case](token)

    def rest_cases(self, tokens: List[str]) -> List[str]:
        """It restore lettercases of tokens according to 'saved_cases' attribute,
        if lettercase was not detected, 'default_case' func would be used
        Args:
            tokens: List of tokens lettercases of that will be restored
                    according to 'saved_cases' attribute
        Return:
            Tokens with restored lettercases, if lettercase was not detected
            then 'default_case'would be used
        """
        return [self._rest_case(token, saved_case) for saved_case, token in zip(self.saved_cases, tokens)]


class RuInflector:
    """Class that morpho-analyses given token and inflects
    random token into certain morpho-form, for russian language.
    It is based on pymorphy2 library
    Attributes:
        morph: pymorphy2.MorphAnalyzer object
    """

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def get_morpho_tag(self, token: str) -> pymorphy2.tagset.OpencorporaTag:
        """Return morpho-form of token
        Args:
            token: token that will be morpho-analysed
        Return:
            result of morph-analyse in pymorphy2.tagset.OpencorporaTag form
        """
        return self.morph.parse(token)[0].tag

    def get_lemma_form(self, token: str) -> str:
        """Return lemma-form of given token
        """
        return self.morph.parse(token)[0].normal_form

    def inflect_token(self, token: str, morpho_tag, force: bool=False) -> str:
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
        return self.morph.parse(token)[0].inflect(morpho_tag)

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
            keys = {"mood", "tense", "aspect", "person", "voice", "gender", "number"}
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
        self.with_source_token = with_source_token
        self.synonyms_data = pd.read_csv(self.dir_path / "synonyms.csv", index_col=0)
        self.text_entry_data = pd.read_csv(self.dir_path / "text_entry.csv", index_col=0)

    def _find_synonyms(self, lemma: str) -> List[str]:
        lemma = lemma.upper()
        #1
        entry_id_set = set(self.text_entry_data[self.text_entry_data['lemma'] == lemma]['entry_id'])
        #2
        concept_id_set = set()
        for entry_id in entry_id_set:
            concept_id_set.update(set(self.synonyms_data[self.synonyms_data['entry_id'] == entry_id]['concept_id']))
        #3
        syn_entry_id_set = set()
        for concept_id in concept_id_set:
            syn_entry_id_set.update(set(self.synonyms_data[self.synonyms_data['concept_id']==concept_id]['entry_id']))
        #4
        synlist = list()
        for syn_entry_id in syn_entry_id_set:
            synlist += list(self.text_entry_data[self.text_entry_data['entry_id']==syn_entry_id]['lemma'])
        return synlist

    def _filter(self, synlist: List[str], source_lemma: str) -> List[str]:
        filtered_syn = set(filter(lambda x: len(x.split()) == 1, synlist))
        if self.with_source_token:
            filtered_syn.update([source_lemma])
        else:
            filtered_syn.discard(source_lemma)
        return list(filtered_syn)

    def get_syn(self, lemma: str, pos_tag: str = None) -> List[str]:
        """It returns synonyms for certain word
        Args:
            lemma: word for that it will search synonyms
            pos_tag: pos_tag in nltk.pos_tag format of 'lemma'
        Return:
             List of synonyms
        """
        lemma = lemma.upper()
        synonyms = self._find_synonyms(lemma)
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

    def _nltk_postag_to_pattern_postag(self, pos_tag: str):
        if pos_tag.startswith('N'):
            return en.NOUN
        elif pos_tag.startswith('V'):
            return en.VERB
        elif pos_tag.startswith('J'):
            return en.ADJECTIVE
        elif pos_tag.startswith('R'):
            return en.ADVERB

    def get_lemma_form(self, token: str):
        """It returns lemma-form of given token. Phrasal verbs should be linked with '_' symbol.
        Args:
            token: token that will be converted into lemma-form
        Return:
            token in lemma-form
        """
        split = token.split('_')
        if len(split) == 1:
            return en.lemma(token)
        else:
            return '_'.join([en.lemma(split[0])] + split[1:])

    def _is_plur(self, token, pos_tag):
        return token == en.inflect.pluralize(token,
                                             pos=self._nltk_postag_to_pattern_postag(pos_tag),
                                             classical=self.classical_pluralize)

    def _is_comp(self, token):
        return token == en.inflect.comparative(token)

    def _is_supr(self, token):
        return token == en.inflect.superlative(token)

    def _get_verb_tense(self, token):
        candidates = en.tenses(token)
        if len(candidates) == 0:
            return None
        else:
            return candidates[0]

    def get_morpho_tag(self, token: str, pos_tag):
        """Return morpho-form of token. Phrasal verbs should be linked with '_' symbol.
        Args:
            token: token that will be morpho-analysed
            pos_tag: pos tag in nltk.pos_tag format
        Return:
            morpho_tags in format {'pos_tag', 'plur', 'tense', 'comp', 'supr'}
        """
        morpho_tag = {}
        split = token.split('_')
        if pos_tag.startswith('N'):
            morpho_tag.update({'pos_tag': pos_tag, 'plur': self._is_plur(split[-1], pos_tag)})
        elif pos_tag.startswith('V'):
            morpho_tag.update({'pos_tag': pos_tag, 'tense': self._get_verb_tense(split[0])})
        elif pos_tag.startswith('J'):
            morpho_tag.update({'pos_tag': pos_tag, 'comp': self._is_comp(token), 'supr': self._is_supr(token)})
        elif pos_tag.startswith('R'):
            morpho_tag.update({'pos_tag': pos_tag, 'comp': self._is_comp(token), 'supr': self._is_supr(token)})
        return morpho_tag
    
    def inflect_token(self, token, morpho_tag):
        """It inflects token in certain morpho-form. Phrasal verbs should be linked with '_' symbol.
        Args:
            token: token that will be inflected
            morpho_tag: morpho_tags in {'pos_tag', 'plur', 'tense', 'comp', 'supr'} format
        Return:
            inflected token
        """
        split = token.split('_')
        if morpho_tag['tense']:
            return "_".join([en.conjugate(split[0], morpho_tag['tense'])] + split[1:])
        if morpho_tag['plur']:
            return "_".join(split[:-1] + [en.inflect.pluralize(split[-1],
                                                               pos=self._nltk_postag_to_pattern_postag(morpho_tag['pos_tag']),
                                                               classical=self.classical_pluralize)])
        if morpho_tag['comp']:
            return en.inflect.comparative(token)
        if morpho_tag['supr']:
            return en.inflect.superlative(token)
        return token


class EnThesaurus:
    """Class that finds synonyms, for english language.
    Args:
        with_source_token: wheither source symbol is considered as synonyms to itself
    Attributes:
        with_source_token: wheither source symbol is considered as synonyms to itself
    """

    def __init__(self, with_source_token: bool=False):
        self.with_source_token = with_source_token

    def _nltk_postag_to_wordnet_postag(self, pos_tag: str):
        if pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('J'):
            return wn.ADJECTIVE
        elif pos_tag.startswith('R'):
            return wn.ADVERB

    def _find_synonyms(self, lemma, pos_tag):
        pos_tag = self._nltk_postag_to_wordnet_postag(pos_tag)
        wn_synonyms = wn.synsets(lemma, pos=pos_tag)
        if pos_tag == wn.ADJ:
            wn_synonyms = wn_synonyms.extend(wn.synsets(lemma, pos='s'))
        lemmas = sum((map(lambda x: x.lemmas(), wn_synonyms)), [])
        synonyms = set(map(lambda x: x.name(), lemmas))
        return synonyms

    def _filter(self, synlist: List[str], source_lemma: str) -> List[str]:
        filtered_syn = set(filter(lambda x: len(x.split('_')) == 1, synlist))
        if self.with_source_token:
            filtered_syn.add(source_lemma)
        else:
            filtered_syn.discard(source_lemma)
        return list(filtered_syn)

    def get_syn(self, lemma: str, pos_tag: str=None) -> List[str]:
        """It returns synonyms for certain word
        Args:
            lemma: word for that it will search synonyms
            pos_tag: pos_tag in nltk.pos_tag format of 'lemma'
        Return:
             List of synonyms
        """
        synonyms = self._find_synonyms(lemma, pos_tag)
        synonyms = self._filter(synonyms, lemma)
        return synonyms



