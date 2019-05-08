import pymorphy2
from typing import Union
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

    def __init__(self, force_inflect=True):
        self.morph = pymorphy2.MorphAnalyzer()
        self.convertor = converters.converter('opencorpora-int', 'ud20')
        self.force_inflect = force_inflect

    def get_lemma_form(self, token: str, morpho_tag = None) -> str:
        """Return lemma-form of given token
        """
        parses = self.morph.parse(token)
        if morpho_tag is None:
            return parses[0].normal_form
        parses_in_ud2 = [self._transform_ud20_form(parse) for parse in parses]
        parses_in_ud2 = self._filter_and_sort_parses(parses_in_ud2, morpho_tag)
        if len(parses_in_ud2) > 0:
            return parses_in_ud2[0]['pymorphy'].normal_form
        return parses[0].normal_form

    def _transform_ud20_form(self, parse):
        ud20 = self.convertor(str(parse.tag)).split()
        ud20 = {'source_token': parse.word, 'pos_tag': ud20[0], 'features': self.__get_morpho_features(ud20[1])}
        return {'pymorphy': parse, 'ud20': ud20}

    def __get_morpho_features(self, features: str):
        if features == '_':
            return {}
        features = features.split('|')
        features = list(map(lambda x: tuple(x.split('=')), features))
        return dict(features)

    def _get_similarity_morpho_tag(self, first_morpho, second_morpho):
        if first_morpho['pos_tag'] != second_morpho['pos_tag']:
            return 0
        num_matches = len(set(first_morpho['features'].items()) & set(second_morpho['features'].items()))
        return num_matches + 1

    def _get_similarity_morpho_tag_binary(self, first_morpho, second_morpho):
        if first_morpho['pos_tag'] == second_morpho['pos_tag'] \
                and set(first_morpho['features'].items()) >= set(second_morpho['features'].items()):
                return 1
        return 0

    def extract_requirement_morpho_tags(self, morpho_tag):
        if morpho_tag['pos_tag'] in ["NOUN", "NUM"]:
            keys = ["Case", "Number"]
        elif morpho_tag['pos_tag'] in ["ADJ"]:
            keys = ["Case", "Gender", "Number"]
        elif morpho_tag['pos_tag'] in ["ADV"]:
            keys = ["Degree"]
        elif morpho_tag['pos_tag'] == "VERB":
            if morpho_tag['features'].get('Voice') == 'Act':
                keys = ["Mood", "Tense", "Aspect", "Person", "Gender", "Number", "VerbForm"]
            else:
                keys = ["Mood", "Tense", "Aspect", "Person", "Voice", "Gender", "Number", "VerbForm"]
        else:
            keys = []
        values = [morpho_tag['features'].get(key) for key in keys]
        morpho_tag['features'] = dict(filter(lambda x: bool(x[1]), zip(keys, values)))
        return morpho_tag

    def _filter_and_sort_parses(self, parses, morpho_tag, identicaly=False):
        if self.force_inflect:
            morpho_tag = self.extract_requirement_morpho_tags(morpho_tag)
        if identicaly:
            parses_with_sim = [(parse, self._get_similarity_morpho_tag_binary(morpho_tag, parse['ud20']))
                               for parse in parses]
        else:
            parses_with_sim = [(parse, self._get_similarity_morpho_tag(morpho_tag, parse['ud20']))
                               for parse in parses]
        filtered = list(filter(lambda x: x[1] > 0, parses_with_sim))
        filtered.sort(reverse=True, key=lambda x: x[1])
        return [i[0] for i in filtered]

    def inflect_token(self, token: str, morpho_tag) -> Union[str, None]:
        parses = self.morph.parse(token)
        parses = [self._transform_ud20_form(parse) for parse in parses]
        parses = self._filter_and_sort_parses(parses, morpho_tag)
        if not parses:
            return None
        lexemes = parses[0]['pymorphy'].lexeme
        lexemes = [self._transform_ud20_form(lexeme) for lexeme in lexemes]
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

    def get_lemma_form(self, token: str, morpho_tag):
        if morpho_tag['pos_tag'] == 'VERB':
            return self.get_lemma_verb(token)
        return self.lemmatizer.lemmatize(token)

    def get_lemma_verb(self, token: str):
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join([en.lemma(splited[0])] + splited[1:])
        return en.lemma(token)

    def pluralize(self, token, morpho_tag):
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join(splited[:-1] + en.pluralize(splited[-1],
                                                        self.to_en_pos(morpho_tag['pos_tag']),
                                                        self.classical_pluralize))
        return en.pluralize(token, self.to_en_pos(morpho_tag['pos_tag']), self.classical_pluralize)

    def singularize(self, token, morpho_tag):
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join(splited[:-1] + en.singularize(splited[-1], self.to_en_pos(morpho_tag['pos_tag'])))
        return en.pluralize(token, self.to_en_pos(morpho_tag['pos_tag']), self.classical_pluralize)

    def _tense_similarity(self, first_tense, second_tense):
        if first_tense[0] == second_tense[0] and\
           first_tense[-1] == second_tense[-1]:
            return sum([int(x == y) for x, y in zip(first_tense, second_tense)])
        return 0

    def _sort_and_filter_candidates(self, source, candidates):
        candidates = [(cand, self._tense_similarity(source, cand)) for cand in candidates]
        candidates = list(filter(lambda x: x[1] > 0, candidates))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cand[0] for cand in candidates]

    def _inflect_verb(self, token, morpho_tag):
        candidate_tenses = en.tenses(morpho_tag['source_token'])
        morpho_tense = self.verb_convertor_ud20_pattern.convert(morpho_tag)
        candidate_tenses = self._sort_and_filter_candidates(morpho_tense, candidate_tenses)
        tense_for_inflection = list(dropwhile(lambda cand: en.conjugate(token, cand) is None, candidate_tenses))
        lemma = self.get_lemma_verb(token)
        if tense_for_inflection is not []:
            splited = lemma.split('_')
            if len(splited) > 1:
                return "_".join([en.conjugate(splited[0], tense_for_inflection[0])] + splited[1:])
            else:
                return en.conjugate(lemma, tense_for_inflection[0])
        return None

    def inflect_token(self, token, morpho_tag):
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


class VerbConvertorUD20Patternen(object):

    def __init__(self):
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

    def convert(self, morpho_tag):
        return (self._get_verb_tense(morpho_tag),
                self._get_verb_person(morpho_tag),
                self._get_verb_number(morpho_tag),
                self._get_verb_mood(morpho_tag),
                self._get_verb_aspect(morpho_tag))
