import pymorphy2
from typing import Union
import pattern.en as en
from nltk.stem import WordNetLemmatizer
from itertools import dropwhile
from russian_tagsets import converters

class RuInflector:
    """Works only with noun verb adjective and adverb
    """

    def __init__(self, force_inflect=False):
        self.morph = pymorphy2.MorphAnalyzer()
        self.convertor = converters.converter('opencorpora-int', 'ud20')
        self.force_inflect = force_inflect

    def get_lemma_form(self, token: str, morpho_tag = None) -> str:
        """Return lemma-form of given token
        """
        parses = self.morph.parse(token)
        if morpho_tag is None:
            return parses[0].normal_form
        parses = [self._tranform_ud20_form(parse) for parse in parses]
        parses = self._filter_and_sort_parses(parses, morpho_tag)
        if len(parses) > 0:
            return parses[0]['pymorphy'].normal_form
        return self.morph.parse(token)[0].normal_form

    def _tranform_ud20_form(self, parse):
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
            keys = ["Mood", "Tense", "Aspect", "Person", "Voice", "Gender", "Number", "Verbform"]
        else:
            keys = []
        values = [morpho_tag['features'].get(key) for key in keys]
        morpho_tag['features'] = dict(zip(keys, values))
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
        parses = [self._tranform_ud20_form(parse) for parse in parses]
        parses = self._filter_and_sort_parses(parses, morpho_tag)
        if not parses:
            return None
        lexemes = parses[0]['pymorphy'].lexeme
        lexemes = [self._tranform_ud20_form(lexeme) for lexeme in lexemes]
        lexemes = self._filter_and_sort_parses(lexemes, morpho_tag, True)
        if not lexemes:
            return None
        return lexemes[0]['pymorphy'].word


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
        self.convertor_ud20_pattern = Convertor_UD20_pattern()

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
        morpho_tense = self.convertor_ud20_pattern.verb(morpho_tag)
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


class Convertor_UD20_pattern:

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

    def verb(self, morpho_tag):
        return (self._get_verb_tense(morpho_tag),
                self._get_verb_person(morpho_tag),
                self._get_verb_number(morpho_tag),
                self._get_verb_mood(morpho_tag),
                self._get_verb_aspect(morpho_tag))

if __name__ == '__main__':
    r = RuInflector()
    print(r.inflect_token('коричневый', {'source_token': 'синего',
                                         'pos_tag': 'ADJ',
                                         'features': {'Case': 'Gen', 'Degree': 'Pos', 'Gender': 'Masc', 'Number':'Sing'}}))
    print(r.inflect_token('скейтер', {'source_token': 'анимешник',
                                      'pos_tag': 'NOUN',
                                      'features': {'Animacy':'Inan','Case':'Nom','Gender':'Masc','Number':'Sing'}}))
    print(r.get_lemma_form('рыло', {'pos_tag': 'NOUN', 'features': {}}))
    print(r.get_lemma_form('рыло', {'pos_tag': 'VERB', 'features': {}}))