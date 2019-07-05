from abc import abstractmethod
from itertools import dropwhile
from itertools import repeat
from typing import List, Union, Optional

import pattern.en as en
from nltk.stem import WordNetLemmatizer
import pymorphy2
from russian_tagsets import converters

from deeppavlov.models.morpho_tagger.lemmatizer import UDPymorphyLemmatizer
from deeppavlov.models.morpho_tagger.common_tagger import make_full_UD_tag
from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag

class UD20_en_convertor(object):
    """This class is convert morpho tags, from UD2.0 format into format that applies in Pattern.en library

    Args:

    Attributes:
        tense: dict that maps names of tenses
        person: dict that maps names of person
        number: dict that maps names of number
        mood: dict that maps names of mood
        aspect: dict that maps names of aspect
    """

    def __init__(self):
        self.tense = {'Past': 'past', 'Pres': 'present', 'Fut': 'future'}
        self.person = {'1': 1, '2': 2, '3': 3}
        self.number = {'Sing': 'singular', 'Plur': 'plural'}
        self.mood = {'Ind': 'indicative', 'Imp': 'imperative', 'Cnd': 'conditional', 'Sub': 'subjunctive'}
        self.aspect = {'Imp': 'imperfective', 'Perf': 'perfective', 'Prog': 'progressive'}

    def _get_verb_tense(self, morpho_tag: dict) -> str:
        if morpho_tag['features'].get('VerbForm') == 'Inf':
            return 'infinitive'
        return self.tense.get(morpho_tag['features'].get('Tense'))

    def _get_verb_person(self, morpho_tag: dict) -> str:
        return self.person.get(morpho_tag['features'].get('Person'))

    def _get_verb_number(self, morpho_tag: dict) -> str:
        return self.number.get(morpho_tag['features'].get('Number'))

    def _get_verb_mood(self, morpho_tag: dict) -> str:
        return self.mood.get(morpho_tag['features'].get('Mood'))

    def _get_verb_aspect(self, morpho_tag: dict) -> str:
        aspect = self.aspect.get(morpho_tag['features'].get('Aspect'))
        if aspect is None:
            if morpho_tag['features'].get('VerbForm') == 'Part':
                aspect = 'progressive'
            elif morpho_tag['features'].get('VerbForm') == 'Fin':
                aspect = 'imperfective'
        return aspect

    def convert_verb_morpho_tag(self, morpho_tag: dict) -> dict:
        """Converts morpho tags from UD2.0 format into Pattern.en format
        Args:
            morpho_tag: dict that contains information about some word in UD2.0 format,
                        for instance {'source_token': '',
                                      'pos_tag': 'VERB',
                                      'features': {'Mood': 'Ind',
                                                   'Number': 'Sing',
                                                   'Person': '3',
                                                   'Tense': 'Pres',
                                                   'VerbForm': 'Fin'}
        Return:
             morpho_tag: dict that contains information about some word in Pattern.en format,
                         for instance {'tense': 'past',
                                       'person': 3,
                                       'number': 'singular',
                                       'mood': 'indicative',
                                       'aspect':'imperfective'}
        """
        result = {'tense': self._get_verb_tense(morpho_tag),
                  'person': self._get_verb_person(morpho_tag),
                  'number': self._get_verb_number(morpho_tag),
                  'mood': self._get_verb_mood(morpho_tag),
                  'aspect': self._get_verb_aspect(morpho_tag)}
        return {k: v for k, v in result.items() if v is not None}

class Inflector(object):
    """A basic class for inflectors. It must contain one method:
    * :meth: `inflect_token` for single word inflection in certain morphological form.
              It is an abstract method and should be reimplemented.
    """

    @abstractmethod
    def inflect_token(self, token: str, morpho_tag: dict):
        raise NotImplementedError("Your class should implement 'inflect_token' method")

    def get_morpho_tag_similarity(self, first_morpho_tag: dict, second_morpho_tag: dict) -> int:
        if first_morpho_tag['pos_tag'] == second_morpho_tag['pos_tag']:
            num_matches = len(set(first_morpho_tag['features'].items()) & set(second_morpho_tag['features'].items()))
            return num_matches + 1
        return 0

    def sort_and_filter_candidates_with_similarity(self,
                                                   target: dict,
                                                   candidates: List[dict],
                                                   func_similarity=None) -> List[dict]:
        """
        It sorts and filters morphological tag by similarity with target morphological tag
        Args:
            target: morphological tag to which we want to inflect
            candidates: list of morphological tags from which
                        algorithm chooses candidate that most similar with target
            func_similarity: function that evaluates similarity of two morphological tags
        Return:
            List of morphological tags that
            was sorted with similarity(descending order) and filtered(similarity is not equal with 0)
        """
        if not func_similarity:
            func_similarity = self.get_morpho_tag_similarity
        candidates = [(cand, func_similarity(target, cand)) for cand in candidates]
        candidates = list(filter(lambda x: x[1] > 0, candidates))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cand[0] for cand in candidates]



class EnInflector(Inflector):
    """Class that inflects word and defines lemma form of words (nouns, adjectives, verbs) according with
    morpho tag in UD2.0 format. Inflection bases on pattern.en library.
    Phrasal verbs should be linked with '_' symbol.
    Lemmatization bases on nltk.WordLemmatizer
    Args:
        classical_pluralize: arg from pattern.en.inflect.pluralize function
        sep_of_composite_word: separator that uses in composite words such as phrasal verbs or compound noun
    Attribute:
        classical_pluralize: arg from pattern.en.inflect.pluralize function
        lemmatizer: nltk.WordLemmatizer object
        ud20_en_convertor: UD20_en_convertor object,
                           it converts morpho tags of verb between UD2.0 and pattern.en formats
        to_wn_postag: dict that maps pos_tags into wordnet pos_tag format
        to_en_postag: dict that maps pos_tags into Pattern.en pos_tag format
        position_of_main_word: position of main word in composite words
        verb_form: possible verb form, that uses in Pattern.en library
    """

    def __init__(self, classical_pluralize: bool=True, sep_of_composite_word='_'):
        self.classical_pluralize = classical_pluralize
        self.sep_of_composite_word = sep_of_composite_word
        self.ud20_en_convertor = UD20_en_convertor()
        self.lemmatizer = WordNetLemmatizer()
        self.to_wn_postag = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
        self.to_en_postag = {'NOUN': en.NOUN, 'VERB': en.VERB, 'ADJ': en.ADJECTIVE, 'ADV': en.ADVERB}
        self.position_of_main_word = {'NOUN': -1, 'VERB': 0}
        self.verb_form = [
            {'tense': 'infinitive', 'person': None, 'number': None, 'mood': None, 'aspect': None},
            {'tense': 'past', 'person': 1, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': 1, 'number': 'singular', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': 2, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': 2, 'number': 'singular', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': 3, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': 3, 'number': 'singular', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': None, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': None, 'number': None, 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'past', 'person': None, 'number': None, 'mood': 'indicative', 'aspect': 'progressive'},
            {'tense': 'present', 'person': 1, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'present', 'person': 1, 'number': 'singular', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'present', 'person': 2, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'present', 'person': 2, 'number': 'singular', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'present', 'person': 3, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'present', 'person': 3, 'number': 'singular', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'present', 'person': None, 'number': 'plural', 'mood': 'indicative', 'aspect': 'imperfective'},
            {'tense': 'present', 'person': None, 'number': None, 'mood': 'indicative', 'aspect': 'progressive'}
        ]

    def apply_func_to_certain_part_in_phrasal_word(self, token, position, function, sep=None, *args, **kwargs):
        if not sep:
            sep = self.sep_of_composite_word
        splited = token.split(sep)
        if len(splited) == 1:
            return function(token, *args, **kwargs)
        splited[position] = function(splited[position], *args, **kwargs)
        return sep.join(splited)

    def pluralize(self, token: str, morpho_tag: dict) -> str:
        pos_tag = self.to_en_postag.get(morpho_tag.get('pos_tag'))
        features = morpho_tag.get('features')
        if features and features.get('Number') == 'Plur':
            return token
        plur = self.apply_func_to_certain_part_in_phrasal_word(token,
                                                               position=self.position_of_main_word.get(pos_tag, -1),
                                                               function=en.pluralize,
                                                               pos=pos_tag if pos_tag else self.to_en_postag['NOUN'],
                                                               classical=self.classical_pluralize)
        return plur

    def singularize(self, token: str, morpho_tag: dict) -> str:
        pos_tag = self.to_en_postag.get(morpho_tag.get('pos_tag'))
        features = morpho_tag.get('features')
        if features and features.get('Number') == 'Sing':
            return token
        sing = self.apply_func_to_certain_part_in_phrasal_word(token,
                                                               position=self.position_of_main_word.get(pos_tag, -1),
                                                               function=en.singularize,
                                                               pos=pos_tag if pos_tag else self.to_en_postag['NOUN'])
        return sing

    def _tense_similarity(self, first_tense, second_tense):
        if first_tense['tense'] == second_tense['tense'] and\
                first_tense['aspect'] == second_tense['aspect']:
            return len(set(first_tense.items()) & set(second_tense.items()))
        return 0

    def lemmatize(self, token: str, morpho_tag: dict={}) -> str:
        pos_tag = morpho_tag.get('pos_tag')
        wn_pos_tag = self.to_wn_postag[pos_tag] if pos_tag else self.to_wn_postag['NOUN']
        if pos_tag == 'NOUN' and\
                morpho_tag.get('features', {}).get('Number') == 'Sing':
            return token
        lemma = self.apply_func_to_certain_part_in_phrasal_word(token,
                                                                position=self.position_of_main_word.get(pos_tag, -1),
                                                                function=self.lemmatizer.lemmatize,
                                                                pos=wn_pos_tag)
        return lemma

    def _inflect_verb(self, token, morpho_tag):
        lemma = self.lemmatize(token, morpho_tag)
        morpho_tense = self.ud20_en_convertor.convert_verb_morpho_tag(morpho_tag)
        candidate_tenses = self.sort_and_filter_candidates_with_similarity(target=morpho_tense,
                                                                           candidates=self.verb_form,
                                                                           func_similarity=self._tense_similarity)
        tense_for_inflection = list(dropwhile(lambda cand: en.conjugate(lemma, **cand) is None, candidate_tenses))
        if len(tense_for_inflection) == 0:
            return None
        inflected = self.apply_func_to_certain_part_in_phrasal_word(lemma,
                                                                    position=self.position_of_main_word['VERB'],
                                                                    function=en.conjugate,
                                                                    **(tense_for_inflection[0]))
        return inflected

    def inflect_token(self, token, inflect_morpho_tag):
        """It inflects token in certain morpho-form
        Args:
            token: token that will be inflected
            inflect_morpho_tag: morpho tag in which token will be inflected
                                in UD2.0 format e.g. {'source_token': '', 'pos_tag': 'VERB', 'features': {}}
        """
        if inflect_morpho_tag['pos_tag'] in ['NOUN', 'PROPN']:
            if inflect_morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, {'pos_tag': 'NOUN', 'features': {}})
            elif inflect_morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, {'pos_tag': 'NOUN', 'features': {}})

        if inflect_morpho_tag['pos_tag'] in ['ADJ']:
            if inflect_morpho_tag['features'].get('Degree') == 'Cmp':
                token = en.inflect.comparative(token)
            elif inflect_morpho_tag['features'].get('Degree') == 'Sup':
                token = en.inflect.superlative(token)
            elif inflect_morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, {'pos_tag': 'ADJ', 'features': {}})
            elif inflect_morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, {'pos_tag': 'ADJ', 'features': {}})

        if inflect_morpho_tag['pos_tag'] in ['AUX', 'VERB']:
            token = self._inflect_verb(token, inflect_morpho_tag)
        return token

class RuInflector(Inflector):
    """

    """

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.convertor = converters.converter('opencorpora-int', 'ud20')
        self.lemmatizer = UDPymorphyLemmatizer()
        self.names_of_morpho_tags = {
            'NOUN': {'Case', 'Number', 'Gender'},
            'NUM': {'Case', 'Number', 'Gender'},
            'ADJ': {'Case', 'Gender', 'Number', 'Animacy', 'Variant'},
            'ADV': {'Degree'},
            'VERB': {'Mood', 'Tense', 'Aspect', 'Person', 'Voice', 'Gender', 'Number', 'VerbForm', 'Variant'}
        }
        self._reset_memo()

    def _reset_memo(self):
        self.memo = dict()

    def lemmatize(self, token, morpho_tag: Optional[dict]=None):
        if morpho_tag:
            morpho_tag_string = make_full_UD_tag(pos=morpho_tag.get('pos_tag'),
                                                 tag=morpho_tag.get('features'),
                                                 mode="dict")
        lemma = self.lemmatizer._lemmatize(token, morpho_tag_string if morpho_tag else None)
        return lemma

    def get_morpho_tag_similarity(self, first_morpho_tag: dict, second_morpho_tag: dict) -> int:
        if first_morpho_tag.get('pos_tag') == second_morpho_tag.get('pos_tag'):
            num_matches = len(set(first_morpho_tag['features'].items()) & set(second_morpho_tag['features'].items()))
            return num_matches + 1
        first_features = first_morpho_tag.get('features')
        second_features = second_morpho_tag.get('features')
        if first_morpho_tag and\
                second_morpho_tag and\
                (first_features.get('Degree') == 'Cmp' or second_features.get('Degree') == 'Cmp'):
            num_matches = len(set(first_morpho_tag['features'].items()) & set(second_morpho_tag['features'].items()))
            return num_matches + 1
        return 0

    def get_morpho_tag_binary_similarity(self, first_morpho_tag: dict, second_morpho_tag: dict) -> int:
        if first_morpho_tag['pos_tag'] == second_morpho_tag['pos_tag']:
            if set(first_morpho_tag['features'].items()) == set(second_morpho_tag['features'].items()):
                return 2
            if set(first_morpho_tag['features'].items()) > set(second_morpho_tag['features'].items()):
                return 1
        first_features = first_morpho_tag.get('features')
        second_features = second_morpho_tag.get('features')
        if first_morpho_tag and\
                second_morpho_tag and\
                first_features.get('Degree') == 'Cmp' and\
                second_features.get('Degree') == 'Cmp':
            if set(first_morpho_tag['features'].items()) == set(second_morpho_tag['features'].items()):
                return 2
            if set(first_morpho_tag['features'].items()) > set(second_morpho_tag['features'].items()):
                return 1
        return 0

    def extract_requirement_morpho_tags(self, morpho_tag):
        pos_tag = morpho_tag.get('pos_tag')
        morpho_features = morpho_tag.get('features')
        if not morpho_features:
            return morpho_tag
        keys = self.names_of_morpho_tags.get(pos_tag).copy()
        if pos_tag == 'ADJ' and\
                morpho_features.get('Number') == 'Plur':
            keys.discard('Gender')
        elif pos_tag == "ADV" and\
                morpho_features.get('Degree') == 'Pos':
            keys.discard('Degree')
        elif pos_tag == "VERB" and\
                morpho_features.get('Voice') == 'Act':
            keys.discard('Voice')
        values = [morpho_features.get(key) for key in keys]
        morpho_tag['features'] = dict(filter(lambda x: bool(x[1]), zip(keys, values)))
        return morpho_tag

    def correct_conversion_russian_tagsets(self, pos, morpho_tag, opencorpora_str):
        if morpho_tag.get('Variant') == 'Brev': # correction of russian_tagsets conversion
            morpho_tag['Variant'] = 'Short'
        if pos == 'DET' and\
                'ADJF' in opencorpora_str:
            pos = 'ADJ'
        return pos, morpho_tag

    def get_morpho_dict(self, opencorpora_parse, with_opencorpora: bool=False, with_word: bool=False):
        ud20 = self.convertor(str(opencorpora_parse.tag))
        pos, morpho_tag = make_pos_and_tag(ud20, sep=" ", return_mode="dict")
        pos, morpho_tag = self.correct_conversion_russian_tagsets(pos, morpho_tag, str(opencorpora_parse.tag))
        morpho_dict = {'pos_tag': pos,
                       'features': morpho_tag}
        if with_opencorpora:
            morpho_dict['opencorpora'] = opencorpora_parse
        if with_word:
            morpho_dict['word'] = opencorpora_parse.word
        return morpho_dict

    def inflect_token(self, token: str, morpho_tag: dict) -> Union[str, None]:
        morpho_tag = self.extract_requirement_morpho_tags(morpho_tag)
        morpho_tag_string = make_full_UD_tag(morpho_tag.get('pos_tag'), morpho_tag.get('features'), mode="dict")
        if (token, morpho_tag_string) in self.memo:
            return self.memo.get((token, morpho_tag_string))
        parses = self.morph.parse(token)
        parses = [self.get_morpho_dict(parse, with_opencorpora=True) for parse in parses]
        parses = self.sort_and_filter_candidates_with_similarity(target=morpho_tag,
                                                                 candidates=parses,
                                                                 func_similarity=self.get_morpho_tag_similarity)
        if not parses:
            return None
        lexemes = parses[0]['opencorpora'].lexeme
        lexemes = [self.get_morpho_dict(lexeme, with_word=True) for lexeme in lexemes]
        lexemes = [self.extract_requirement_morpho_tags(lexeme) for lexeme in lexemes]
        lexemes = self.sort_and_filter_candidates_with_similarity(target=morpho_tag,
                                                                  candidates=lexemes,
                                                                  func_similarity=self.get_morpho_tag_binary_similarity)
        if not lexemes:
            return None
        self.memo[(token, morpho_tag_string)] = lexemes[0]['word']
        return lexemes[0]['word']



