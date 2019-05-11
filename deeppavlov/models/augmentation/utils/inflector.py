import pymorphy2
from typing import Union, Optional
import pattern.en as en
from nltk.stem import WordNetLemmatizer
from itertools import dropwhile
from russian_tagsets import converters

class RuInflector:
    """Class that inflects and defines lemma form words (nouns, adjectives, verbs, numbers, adverbs) according with
    morpho tag in UD2.0 format. Inflection bases on pymorphy2 library.
    Phrasal verbs should be linked with '_' symbol.
    Args:
        force_inflect: if it is True, it will be more probability that token will be inflected
    Args:
        force_inflect: if it is True, it will be more probability that token will be inflected
        convetor: it converts morpho_tags from opencorpora format into UD2.0 format
        moprh: MorphAnalyzer object
    """

    def __init__(self, force_inflect: bool=True):
        self.morph = pymorphy2.MorphAnalyzer()
        self.convertor = converters.converter('opencorpora-int', 'ud20')
        self.force_inflect = force_inflect

    def __get_morpho_features(self, features: str):
        if features == '_':
            return {}
        features = features.split('|')
        features = list(map(lambda x: tuple(x.split('=')), features))
        return dict(features)

    def _pymorphy_to_ud20(self, parse, force: bool=True):
        ud20 = self.convertor(str(parse.tag)).split()
        ud20 = {'source_token': parse.word, 'pos_tag': ud20[0], 'features': self.__get_morpho_features(ud20[1])}
        if force:
            ud20 = self.extract_requirement_morpho_tags(ud20)
        return {'pymorphy': parse, 'ud20': ud20}

    def get_lemma_form(self, token: str, morpho_tag: Optional[dict]=None) -> str:
        """It returns lemma-form of given token
        Args:
            token: token that will be lemmatized
            morpho_tag: morpho tag of token in UD2.0 format e.g. {'source_token': '', 'pos_tag': 'VERB', 'features': {}}
        """
        parses = self.morph.parse(token)
        if morpho_tag is None:
            return parses[0].normal_form
        parses_in_ud2 = [self._pymorphy_to_ud20(parse) for parse in parses]
        parses_in_ud2 = self._filter_and_sort_parses(parses_in_ud2, morpho_tag)
        if len(parses_in_ud2) > 0:
            return parses_in_ud2[0]['pymorphy'].normal_form
        return parses[0].normal_form

    def _get_similarity_morpho_tag(self, first_morpho, second_morpho):
        if first_morpho['pos_tag'] == second_morpho['pos_tag'] or\
                first_morpho['features'].get('Degree') == 'Cmp' or\
                second_morpho['features'].get('Degree') == 'Cmp':
            num_matches = len(set(first_morpho['features'].items()) & set(second_morpho['features'].items()))
            return num_matches + 1
        return 0

    def _get_similarity_morpho_tag_binary(self, first_morpho, second_morpho):
        if first_morpho['features'].get('Degree') == 'Cmp' and\
                second_morpho['features'].get('Degree') == 'Cmp' and\
                set(first_morpho['features'].items()) >= set(second_morpho['features'].items()):
            return 1
        if first_morpho['pos_tag'] == second_morpho['pos_tag'] and\
                set(first_morpho['features'].items()) >= set(second_morpho['features'].items()):
            return 1
        return 0

    def extract_requirement_morpho_tags(self, morpho_tag):
        keys = []
        if morpho_tag['pos_tag'] in ['NOUN', 'NUM']:
            keys = {'Case', 'Number', 'Gender'}
        elif morpho_tag['pos_tag'] in ['ADJ']:
            keys = {'Case', 'Gender', 'Number', 'Animacy'}
            if morpho_tag['features'].get('Number') == 'Plur':
                keys.discard('Gender')
        elif morpho_tag['pos_tag'] in ["ADV"]:
            keys = {'Degree'}
            if morpho_tag['features'].get('Degree') == 'Pos':
                keys.discard('Degree')
        elif morpho_tag['pos_tag'] == "VERB":
            keys = {"Mood", "Tense", "Aspect", "Person", "Voice", "Gender", "Number", "VerbForm"}
            if morpho_tag['features'].get('Voice') == 'Act':
                keys.discard('Voice')
        values = [morpho_tag['features'].get(key) for key in keys]
        morpho_tag['features'] = dict(filter(lambda x: bool(x[1]), zip(keys, values)))
        return morpho_tag

    def _filter_and_sort_parses(self, parses, morpho_tag, identicaly=False):
        if self.force_inflect:
            morpho_tag = self.extract_requirement_morpho_tags(morpho_tag)
        similarity_func = self._get_similarity_morpho_tag_binary if identicaly else self._get_similarity_morpho_tag
        parses_with_similarity = [(parse, similarity_func(morpho_tag, parse['ud20'])) for parse in parses]
        filtered = list(filter(lambda x: x[1] > 0, parses_with_similarity))
        filtered.sort(reverse=True, key=lambda x: x[1])
        return [i[0] for i in filtered]

    def inflect_token(self, token: str, morpho_tag: dict) -> Union[str, None]:
        """It inflects token in certain morpho-form
        Args:
            token: token that will be inflected
            morpho_tag: morpho tag in which token will be inflected
                        in UD2.0 format e.g. {'source_token': '', 'pos_tag': 'VERB', 'features': {}}
        """
        parses = self.morph.parse(token)
        parses = [self._pymorphy_to_ud20(parse) for parse in parses]
        parses = self._filter_and_sort_parses(parses, morpho_tag, False)
        if not parses:
            return None
        lexemes = parses[0]['pymorphy'].lexeme
        lexemes = [self._pymorphy_to_ud20(lexeme) for lexeme in lexemes]
        lexemes = self._filter_and_sort_parses(lexemes, morpho_tag, True)
        if not lexemes:
            return None
        return lexemes[0]['pymorphy'].word


class EnInflector(object):
    """Class that inflects and defines lemma form words (nouns, adjectives, verbs) according with
    morpho tag in UD2.0 format. Inflection bases on pattern.en library.
    Phrasal verbs should be linked with '_' symbol.
    Lemmatization bases on nltk.WordLemmatizer
    Args:
        classical_pluralize: arg from pattern.en.inflect.pluralize function
    Attribute:
        classical_pluralize: arg from pattern.en.inflect.pluralize function
        lemmatizer: nltk.WordLemmatizer object
        verb_convertor_ud20_pattern: VerbConvertorUD20Patternen object,
                                     it converts morpho tags of verb between UD2.0 and pattern.en formats
    """

    def __init__(self, classical_pluralize: bool=True):
        self.classical_pluralize = classical_pluralize
        self.lemmatizer = WordNetLemmatizer()
        self.verb_convertor_ud20_pattern = VerbConvertorUD20Patternen()
        self.to_wn_postag = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
        self.to_en_postag = {'NOUN': en.NOUN, 'VERB': en.VERB, 'ADJ': en.ADJECTIVE, 'ADV': en.ADVERB}
        self.possible_verb_morph = [('infinitive', None, None, None, None),
                                     ('past', 1, 'plural', 'indicative', 'imperfective'),
                                     ('past', 1, 'singular', 'indicative', 'imperfective'),
                                     ('past', 2, 'plural', 'indicative', 'imperfective'),
                                     ('past', 2, 'singular', 'indicative', 'imperfective'),
                                     ('past', 3, 'plural', 'indicative', 'imperfective'),
                                     ('past', 3, 'singular', 'indicative', 'imperfective'),
                                     ('past', None, 'plural', 'indicative', 'imperfective'),
                                     ('past', None, None, 'indicative', 'imperfective'),
                                     ('past', None, None, 'indicative', 'progressive'),
                                     ('present', 1, 'plural', 'indicative', 'imperfective'),
                                     ('present', 1, 'singular', 'indicative', 'imperfective'),
                                     ('present', 2, 'plural', 'indicative', 'imperfective'),
                                     ('present', 2, 'singular', 'indicative', 'imperfective'),
                                     ('present', 3, 'plural', 'indicative', 'imperfective'),
                                     ('present', 3, 'singular', 'indicative', 'imperfective'),
                                     ('present', None, 'plural', 'indicative', 'imperfective'),
                                     ('present', None, None, 'indicative', 'progressive')]

    def get_lemma_form(self, token: str, token_morpho_tag={}):
        """It returns lemma-form of given token
        Args:
            token: token that will be lemmatized
            morpho_tag: morpho tag of token in UD2.0 format e.g. {'source_token': '', 'pos_tag': 'VERB', 'features': {}}
        """
        if token_morpho_tag.get('pos_tag') == 'VERB':
            return self.get_lemma_verb(token)
        if token_morpho_tag.get('pos_tag') == 'NOUN' and token_morpho_tag.get('features', {}).get('Number') == 'Sing':
            return token
        pos = self.to_wn_postag.get(token_morpho_tag.get('pos_tag'))
        if pos:
            return self.lemmatizer.lemmatize(token, pos)
        return self.lemmatizer.lemmatize(token)

    def get_lemma_verb(self, token: str):
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join([en.lemma(splited[0])] + splited[1:])
        return en.lemma(token)

    def pluralize(self, token, token_morpho_tag):
        splited = token.split('_')
        pos = self.to_en_postag[token_morpho_tag['pos_tag']]
        if token_morpho_tag['features'].get('Number') == 'Plur':
            return token
        if len(splited) > 1:
            return "_".join(splited[:-1] + [en.pluralize(splited[-1], pos, classical=self.classical_pluralize)])
        return en.pluralize(token, pos, classical=self.classical_pluralize)

    def singularize(self, token, token_morpho_tag):
        splited = token.split('_')
        pos = self.to_en_postag[token_morpho_tag['pos_tag']]
        if token_morpho_tag['features'].get('Number') == 'Sing':
            return token
        if len(splited) > 1:
            return "_".join(splited[:-1] + [en.singularize(splited[-1], pos)])
        return en.singularize(token, pos)

    def _tense_similarity(self, first_tense, second_tense):
        if first_tense[0] == second_tense[0] and first_tense[-1] == second_tense[-1]:
            return sum([int(x == y) for x, y in zip(first_tense, second_tense)])
        return 0

    def _sort_and_filter_candidates(self, source, candidates):
        candidates = [(cand, self._tense_similarity(source, cand)) for cand in candidates]
        candidates = list(filter(lambda x: x[1] > 0, candidates))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cand[0] for cand in candidates]

    def _inflect_verb(self, token, morpho_tag):
        lemma = self.get_lemma_verb(token)
        morpho_tense = self.verb_convertor_ud20_pattern.convert(morpho_tag)
        candidate_tenses = self._sort_and_filter_candidates(morpho_tense, self.possible_verb_morph)
        tense_for_inflection = list(dropwhile(lambda cand: en.conjugate(lemma, cand) is None, candidate_tenses))
        if len(tense_for_inflection) > 0:
            splited = lemma.split('_')
            if len(splited) > 1:
                return "_".join([en.conjugate(splited[0], tense_for_inflection[0])] + splited[1:])
            else:
                return en.conjugate(lemma, tense_for_inflection[0])
        return None

    def inflect_token(self, token, morpho_tag):
        """It inflects token in certain morpho-form
        Args:
            token: token that will be inflected
            morpho_tag: morpho tag in which token will be inflected
                        in UD2.0 format e.g. {'source_token': '', 'pos_tag': 'VERB', 'features': {}}
        """
        if morpho_tag['pos_tag'] in ['NOUN', 'PROPN']:
            if morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, {'pos_tag': 'NOUN', 'features': {}})
            if morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, {'pos_tag': 'NOUN', 'features': {}})
        if morpho_tag['pos_tag'] in ['ADJ']:
            if morpho_tag['features'].get('Degree') == 'Cmp':
                token = en.inflect.comparative(token)
            if morpho_tag['features'].get('Degree') == 'Sup':
                token = en.inflect.superlative(token)
            if morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, {'pos_tag': 'ADJ', 'features': {}})
            if morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, {'pos_tag': 'ADJ', 'features': {}})
        if morpho_tag['pos_tag'] in ['VERB']:
            token = self._inflect_verb(token, morpho_tag)
        return token


class VerbConvertorUD20Patternen(object):

    def __init__(self):
        self.ud_to_en_tense = {'Past': 'past',
                               'Pres': 'present',
                               'Fut': 'future'}
        self.ud_to_en_person = {'1': 1,
                                '2': 2,
                                '3': 3}
        self.ud_to_en_number = {'Sing': 'singular',
                                'Plur': 'plural'}
        self.ud_to_en_mood = {'Ind': 'indicative',
                              'Imp': 'imperative',
                              'Cnd': 'conditional',
                              'Sub': 'subjunctive'}
        self.ud_to_en_aspect = {'Imp': 'imperfective',
                                'Perf': 'perfective',
                                'Prog': 'progressive'}

    def _get_verb_tense(self, morpho_tag):
        if morpho_tag['features'].get('VerbForm') == 'Inf':
            return 'infinitive'
        return self.ud_to_en_tense.get(morpho_tag['features'].get('Tense'))

    def _get_verb_person(self, morpho_tag):
        return self.ud_to_en_person.get(morpho_tag['features'].get('Person'))

    def _get_verb_number(self, morpho_tag):
        return self.ud_to_en_number.get(morpho_tag['features'].get('Number'))

    def _get_verb_mood(self, morpho_tag):
        return self.ud_to_en_mood.get(morpho_tag['features'].get('Mood'))

    def _get_verb_aspect(self, morpho_tag):
        aspect = self.ud_to_en_aspect.get(morpho_tag['features'].get('Aspect'))
        if aspect is None:
            if morpho_tag['features'].get('VerbForm') == 'Part':
                aspect = 'progressive'
            elif morpho_tag['features'].get('VerbForm') == 'Fin':
                aspect = 'imperfective'
        return aspect

    def convert(self, morpho_tag):
        return (self._get_verb_tense(morpho_tag),
                self._get_verb_person(morpho_tag),
                self._get_verb_number(morpho_tag),
                self._get_verb_mood(morpho_tag),
                self._get_verb_aspect(morpho_tag))
