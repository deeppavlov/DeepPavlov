import pattern.en as en

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
        self.ud_to_en_person = {
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

    def apply_func_according_pos_tag(self, token, morpho_tag, func, *args, **kwargs):
        splited = token.split('_')
        if (len(splited) > 1) and (morpho_tag['pos'] == 'VERB'):
            return "_".join([func(splited[0], args, kwargs)].extend(splited[1:]))
        elif (len(splited) > 1) and (morpho_tag['pos'] in ['NOUN', 'PROPN']):
            return "_".join(splited[:-1].extend(func(splited[-1], args, kwargs)))
        else:
            return func(token, args, kwargs)

    def get_lemma_verb(self, token: str):
        """"""
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join([en.lemma(splited[0]].extend(splited[1:]))
        return en.lemma(token)

    def pluralize(self, token, morpho_tag):
        """"""
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join(splited[:-1].extend(en.pluralize(splited[-1],
                                                             self.to_en_pos(morpho_tag['pos']),
                                                             self.classical_pluralize)))
        return en.pluralize(token, self.to_en_pos(morpho_tag['pos']), self.classical_pluralize)

    def singularize(self, token, morpho_tag):
        """"""
        splited = token.split('_')
        if len(splited) > 1:
            return "_".join(splited[:-1].extend(en.singularize(splited[-1], self.to_en_pos(morpho_tag['pos']))))
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
        return self.ud_to_en_aspect.get(morpho_tag['features'].get('aspect'), None)

    def _transform_to_pattern_en_form(self, morpho_tag):
        return (self._get_verb_tense(morpho_tag),
                self._get_verb_person(morpho_tag),
                self._get_verb_number(morpho_tag),
                self._get_verb_mood(morpho_tag),
                self._get_verb_aspect(morpho_tag))

    def _tense_similarity(self, first_tense, second_tense):
        if first_tense[0] != second_tense[0]:
            return 0
        return sum([int(x == y) for x, y in zip(first_tense, second_tense)])

    def inflect_token(self, lemma, morpho_tag, force: bool = False):
        """It inflects token in certain morpho-form. Phrasal verbs should be linked with '_' symbol.
        Args:
            lemma: token that will be inflected
            morpho_tag: morpho_tags in {'pos_tag', 'plur', 'tense', 'comp', 'supr'} format
        Return:
            inflected token
        """
        if morpho_tag['pos'] in ['NOUN', 'PROPN']:
            if morpho_tag['features'].get('Number') == 'Sing':
                lemma = self.singularize(lemma, morpho_tag)
            if morpho_tag['features'].get('Number') == 'Plur':
                lemma = self.pluralize(lemma, morpho_tag)
        if morpho_tag['pos'] in ['ADJ']:
            if morpho_tag['features'].get('Degree') == 'Cmp':
                lemma = en.inflect.comparative(lemma)
            if morpho_tag['features'].get('Degree') == 'Sup':
                lemma = en.inflect.superlative(lemma)
            if morpho_tag['features'].get('Number') == 'Sing':
                lemma = self.singularize(lemma, morpho_tag)
            if morpho_tag['features'].get('Number') == 'Plur':
                lemma = self.pluralize(lemma, morpho_tag)
        if morpho_tag['pos'] in ['VERB']:
            lemma = en.conjugate(self.get_lemma_verb(lemma),
                                 tense = self._get_verb_tense(morpho_tag),
                                 person = self._get_verb_person(morpho_tag),
                                 number = self._get_verb_number(morpho_tag),
                                 mood = self._get_verb_mood(morpho_tag),
                                 aspect = self._get_verb_aspect(morpho_tag),
                                 )