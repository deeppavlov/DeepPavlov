import pattern.en as en
from itertools import dropwhile

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
                                                        self.to_en_pos(morpho_tag['pos']),
                                                        self.classical_pluralize))
        return en.pluralize(token, self.to_en_pos(morpho_tag['pos']), self.classical_pluralize)

    def singularize(self, token, morpho_tag):
        """"""
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join(splited[:-1] + en.singularize(splited[-1], self.to_en_pos(morpho_tag['pos'])))
        return en.pluralize(token, self.to_en_pos(morpho_tag['pos']), self.classical_pluralize)

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
        if morpho_tag['pos'] in ['NOUN', 'PROPN']:
            if morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, morpho_tag)
            if morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, morpho_tag)
        if morpho_tag['pos'] in ['ADJ']:
            if morpho_tag['features'].get('Degree') == 'Cmp':
                token = en.inflect.comparative(token)
            if morpho_tag['features'].get('Degree') == 'Sup':
                token = en.inflect.superlative(token)
            if morpho_tag['features'].get('Number') == 'Sing':
                token = self.singularize(token, morpho_tag)
            if morpho_tag['features'].get('Number') == 'Plur':
                token = self.pluralize(token, morpho_tag)
        if morpho_tag['pos'] in ['VERB']:
            token = self._inflect_verb(token, morpho_tag)
        return token

if __name__ == '__main__':
    e = EnInflector()
    morpho_tag = {'source_token': 'recieved', 'pos': 'VERB', 'features': {'tense': 'Past', 'VerbForm': 'Part'}}
    print(e.inflect_token('get', morpho_tag))
    morpho_tag = {'source_token': 'recieved', 'pos': 'VERB', 'features': {'tense': 'Past', 'VerbForm': 'Fin', 'Mood': 'Ind'}}
    print(e.inflect_token('get', morpho_tag))
    morpho_tag = {'source_token': 'recieved', 'pos': 'VERB',
                  'features': {'tense': 'Past', 'VerbForm': 'Fin', 'Mood': 'Ind'}}
    print(e.inflect_token('caught_up', morpho_tag))
    morpho_tag = {'source_token': 'recieved', 'pos': 'VERB', 'features': {'tense': 'Past', 'VerbForm': 'Part'}}
    print(e.inflect_token('caught_up', morpho_tag))
    morpho_tag = {'source_token': 'recieved', 'pos': 'VERB',
                  'features': {'tense': 'Past', 'VerbForm': 'Fin', 'Mood': 'Ind'}}
    print(e.inflect_token('get_off', morpho_tag))
    morpho_tag = {'source_token': 'recieved', 'pos': 'VERB', 'features': {'tense': 'Past', 'VerbForm': 'Part'}}
    print(e.inflect_token('get_off', morpho_tag))