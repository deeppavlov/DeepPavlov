from deeppavlov.models.augmentation.utils.inflection import *
import pytest

@pytest.fixture()
def resource_en_inflector():
    inflector = EnInflector()
    return inflector

@pytest.mark.parametrize('source_lemma', [('give_up', 'give_up'),
                                          ('had', 'have'),
                                          ('proved', 'prove'),
                                          ('got_out', 'get_out'),
                                          ('smile', 'smile'),
                                          ('been', 'be'),
                                          ('am', 'be'),
                                          ('begins', 'begin')])
def test_en_get_lemma_verb(resource_en_inflector, source_lemma):
    infl = resource_en_inflector
    source, lemma = source_lemma
    my_lemma = infl.lemmatize(source, {'pos_tag': 'VERB', 'features': {}})
    assert my_lemma == lemma

@pytest.mark.parametrize('source_lemma_morph', [('apples', 'apple', {'pos_tag': 'NOUN', 'features': {}}),
                                                ('hardest', 'hard', {'pos_tag': 'ADJ', 'features': {}}),
                                                ('hardly', 'hardly', {'pos_tag': 'ADV', 'features': {}}),
                                                ('phenomena', 'phenomenon', {'pos_tag': 'NOUN', 'features': {}}),
                                                ('os', 'os', {'pos_tag': 'NOUN', 'features': {'Number': 'Sing'}}),
                                                ('colder', 'cold', {'pos_tag': 'ADJ', 'features': {}})])
def test_en_get_lemma_form(resource_en_inflector, source_lemma_morph):
    infl = resource_en_inflector
    source, lemma, morph = source_lemma_morph
    assert infl.lemmatize(source, morph) == lemma

@pytest.mark.parametrize('source_plur_morph', [('apple', 'apples', {'pos_tag': 'NOUN', 'features': {'Number': 'Sing'}}),
                                         ('phenomenon', 'phenomena', {'pos_tag': 'NOUN', 'features': {'Number': 'Sing'}}),
                                         #('minus', 'minuses', {'pos_tag': 'NOUN'}),
                                         #('lotus', 'lotuses', {'pos_tag': 'NOUN'}),
                                         ('green_apple', 'green_apples', {'pos_tag': 'NOUN', 'features': {'Number': 'Sing'}}),
                                         ('fdsa_sky', 'fdsa_skies', {'pos_tag': 'NOUN', 'features': {'Number': 'Sing'}}),
                                         ('apples', 'apples', {'pos_tag': 'NOUN', 'features': {'Number': 'Plur'}})])
def test_en_pluralize(resource_en_inflector, source_plur_morph):
    infl = resource_en_inflector
    source, plur, morph = source_plur_morph
    assert infl.pluralize(source, morph) == plur

@pytest.mark.parametrize('source_sing_morph', [('apples', 'apple', {'pos_tag': 'NOUN', 'features': {'Number': 'Plur'}}),
                                         #('phenomena', 'phenomenon', {'pos_tag': 'NOUN', 'features': {'Number': 'Plur'}}),
                                         #('minuses', 'minus', {'pos_tag': 'NOUN'}),
                                         #('lotuses', 'lotus', {'pos_tag': 'NOUN'}),
                                         ('green_apples', 'green_apple', {'pos_tag': 'NOUN', 'features': {'Number': 'Plur'}}),
                                         ('fdsa_skies', 'fdsa_sky', {'pos_tag': 'NOUN', 'features': {'Number': 'Plur'}}),
                                         ('apple', 'apple', {'pos_tag': 'NOUN', 'features': {'Number': 'Sing'}})])
def test_en_singularize(resource_en_inflector, source_sing_morph):
    infl = resource_en_inflector
    source, sing, morph = source_sing_morph
    assert infl.singularize(source, morph) == sing

@pytest.fixture()
def resource_convertor():
    conv = UD20_en_convertor()
    return conv


@pytest.mark.parametrize('source_pattern', [({'source_token': '',
                                              'pos_tag': 'AUX',
                                              'features': {'Mood': 'Ind',
                                                           'Number': 'Sing',
                                                           'Person': '3',
                                                           'Tense': 'Past',
                                                           'VerbForm': 'Fin'}},
                                             {'tense': 'past', 'person': 3, 'number': 'singular', 'mood': 'indicative', 'aspect':'imperfective'}),
                                            ({'source_token': '',
                                              'pos_tag': 'AUX',
                                              'features': {'Mood': 'Ind',
                                                           'Tense': 'Past',
                                                           'VerbForm': 'Fin'}},
                                             {'tense': 'past', 'mood': 'indicative', 'aspect': 'imperfective'}),
                                            ({'source_token': '',
                                              'pos_tag': 'VERB',
                                              'features': {'Mood': 'Ind',
                                                           'Number': 'Sing',
                                                           'Person': '3',
                                                           'Tense': 'Pres',
                                                           'VerbForm': 'Fin'}},
                                             {'tense': 'present', 'person': 3, 'number': 'singular', 'mood': 'indicative', 'aspect': 'imperfective'})])
def test_convert_verb_ud20_to_pattern(resource_convertor, source_pattern):
    conv = resource_convertor
    source, pattern = source_pattern
    assert conv.convert_verb_morpho_tag(source) == pattern

@pytest.mark.parametrize('source_verb_morph', [('end', 'ends', {'source_token': '',
                                                                'pos_tag': 'VERB',
                                                                'features': {'Mood': 'Ind',
                                                                             'Number': 'Sing',
                                                                             'Person': '3',
                                                                             'Tense': 'Pres',
                                                                             'VerbForm': 'Fin'}}),
                                               ('be', 'been', {'source_token': '',
                                                               'pos_tag': 'VERB',
                                                               'features': {'Tense': 'Past',
                                                                            'VerbForm': 'Part'}}),
                                               ('force', 'forced', {'source_token': '',
                                                                    'pos_tag': 'VERB',
                                                                    'features': {'Tense': 'Past',
                                                                                 'VerbForm': 'Part',
                                                                                 'Voice': 'Pass'}}),
                                               ('get_in', 'got_in', {'source_token': '',
                                                                 'pos_tag': 'VERB',
                                                                 'features': {'Mood': 'Ind',
                                                                              'Tense': 'Past',
                                                                              'VerbForm': 'Fin'}})])
def test_en_verb_inflect(resource_en_inflector, source_verb_morph):
    infl = resource_en_inflector
    source, verb, morph = source_verb_morph
    assert infl.inflect_token(source, morph) == verb

@pytest.mark.parametrize('source_res_morph', [('book', 'books', {'source_token': '',
                                                                 'pos_tag': 'NOUN',
                                                                 'features': {'Number': 'Plur'}}),
                                              ('interesting', 'more interesting', {'source_token': '',
                                                                                   'pos_tag': 'ADJ',
                                                                                   'features': {'Degree': 'Cmp'}}),
                                              ('interestingly', 'interestingly', {'source_token': '',
                                                                                'pos_tag': 'ADV',
                                                                                'features': {}}),
                                              ('Potter_book', 'Potter_books', {'source_token': '',
                                                                               'pos_tag': 'NOUN',
                                                                               'features': {'Number': 'Plur'}})])
def test_en_inflection(resource_en_inflector, source_res_morph):
    infl = resource_en_inflector
    source, res, morph = source_res_morph
    assert infl.inflect_token(source, morph) == res
