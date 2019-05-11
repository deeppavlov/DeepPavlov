import pytest
from pymorphy2 import MorphAnalyzer
from deeppavlov.models.augmentation.utils.inflector import *

@pytest.fixture()
def resource_ru_inflector():
    inflector = RuInflector(force_inflect=False)
    return inflector

@pytest.fixture()
def resource_ru_inflector_force():
    inflector = RuInflector(force_inflect=True)
    return inflector

@pytest.fixture()
def resource_pymorpho():
    m = MorphAnalyzer()
    return m

@pytest.mark.parametrize("source_lemma", [('бегу', 'бег'),
                                          ('прыгнул', 'прыгнуть'),
                                          ('синих', 'синий'),
                                          ('капсулы', 'капсула'),
                                          ('быстрее', 'быстрый'),
                                          ('первее', 'первый'),
                                          ('томно', 'томно')])
def test_ru_lemma_form_without_morpho_tag(resource_ru_inflector,
                                          resource_ru_inflector_force,
                                          source_lemma):
    force_infl = resource_ru_inflector_force
    infl = resource_ru_inflector
    source, lemma = source_lemma
    assert force_infl.get_lemma_form(source) == lemma
    assert infl.get_lemma_form(source) == lemma

@pytest.mark.parametrize("source_lemma_morpho", [('рыло', 'рыть', {'source_token': '',
                                                                   'pos_tag': 'VERB',
                                                                   'features': {}}),
                                                 ('рыло', 'рыло', {'source_token': '',
                                                                   'pos_tag': 'NOUN',
                                                                   'features': {}}),
                                                 ('стекло', 'стекло', {'source_token': '',
                                                                       'pos_tag': 'NOUN',
                                                                       'features': {}}),
                                                 ('стекло', 'стечь', {'source_token': '',
                                                                       'pos_tag': 'VERB',
                                                                       'features': {}}),
                                                 ('жарко', 'жарко', {'source_token': '',
                                                                     'pos_tag': 'ADV',
                                                                     'features': {}}),
                                                 ('жарко', 'жаркий', {'source_token': '',
                                                                     'pos_tag': 'ADJ',
                                                                     'features': {}})
                                                 ])
def test_ru_lemma_form_morpho_tag(resource_ru_inflector,
                                  resource_ru_inflector_force,
                                  source_lemma_morpho):
    force_infl = resource_ru_inflector_force
    infl = resource_ru_inflector
    source, lemma, morpho = source_lemma_morpho
    assert force_infl.get_lemma_form(source, morpho) == lemma
    assert infl.get_lemma_form(source, morpho) == lemma

def test_transform_uf20_form(resource_ru_inflector, resource_pymorpho):
    infl = resource_ru_inflector
    morph = resource_pymorpho
    parse = morph.parse('синих')[0]
    res = infl._pymorphy_to_ud20(parse)
    assert res == {'pymorphy': parse,
                   'ud20': {'source_token': 'синих',
                            'pos_tag': 'ADJ',
                            'features': {'Number': 'Plur', 'Case': 'Gen'}}}


@pytest.mark.parametrize("morphos_score", [({'source_token': '', 'pos_tag': 'VERB', 'features': {}},
                                            {'source_token': '', 'pos_tag': 'NOUN', 'features': {}},
                                            0),
                                           ({'source_token': '', 'pos_tag': 'ADJ', 'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            {'source_token': '', 'pos_tag': 'ADJ', 'features': {'Number': 'Sing', 'Case': 'Gen'}},
                                            2)])
def test_get_similarity_morpho_tag(resource_ru_inflector, morphos_score):
    f, s, score = morphos_score
    infl = resource_ru_inflector
    assert infl._get_similarity_morpho_tag(f, s) == score

@pytest.mark.parametrize("morphos_score", [({'source_token': '', 'pos_tag': 'VERB', 'features': {}},
                                            {'source_token': '', 'pos_tag': 'NOUN', 'features': {}},
                                            0),
                                           ({'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            {'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            1),
                                           ({'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            {'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen', 'a': 'b'}},
                                            0),
                                            ({'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen', 'a': 'b'}},
                                            {'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            1)
                                           ])
def test_get_similarity_morpho_tag_binary(resource_ru_inflector, morphos_score):
    f, s, score = morphos_score
    infl = resource_ru_inflector
    assert infl._get_similarity_morpho_tag_binary(f, s) == score

@pytest.mark.parametrize('param', [('рыть', 'рыли', {'source_token': '',
                                                     'pos_tag': 'VERB',
                                                     'features': {'VerbForm': 'Fin',
                                                                     'Tense': 'Past',
                                                                     'Number': 'Plur',
                                                                     'Mood': 'Ind',
                                                                     'Aspect': 'Imp',
                                                                     'Voice': 'Act'}}),
                                   ('коричневый', 'коричневых', {'source_token': '',
                                                                 'pos_tag': 'ADJ',
                                                                 'features': {
                                                                     'Number': 'Plur',
                                                                     'Case': 'Gen',
                                                                     'Degree': 'Pos'
                                                                 }}),
                                   ('стать', 'стало', {'source_token': '',
                                                     'pos_tag': 'VERB',
                                                     'features': {'VerbForm': 'Fin',
                                                                  'Tense': 'Past',
                                                                  'Number': 'Sing',
                                                                  'Mood': 'Ind',
                                                                  'Gender': 'Neut',
                                                                  'Aspect': 'Perf',
                                                                  'Voice': 'Mid'}}),
                                   ('порассказывать', 'порассказывайте', {'source_token': '',
                                                                          'pos_tag': 'VERB',
                                                                          'features': {'VerbForm': 'Fin',
                                                                                       'Voice': 'Act',
                                                                                       'Tense': 'Fut',
                                                                                       'Person': '2',
                                                                                       'Number': 'Plur',
                                                                                       'Mood': 'Imp',
                                                                                       'Aspect': 'Imp'

                                                                          }}),
                                   ('малоизвестный', 'малоизвестные', {'source_token': '',
                                                                       'pos_tag': 'ADJ',
                                                                       'features': {'Animacy': 'Inan',
                                                                                    'Case': 'Acc',
                                                                                    'Degree': 'Pos',
                                                                                    'Number': 'Plur'

                                                                       }}),
                                   ('быстро', 'быстро', {'source_token': '',
                                                         'pos_tag': 'ADV',
                                                         'features': {'Degree': 'Pos'}}),
                                   ('желаться', 'желается', {'source_token': '',
                                                           'pos_tag': 'VERB',
                                                           'features': {'Aspect': 'Imp',
                                                                        'Mood': 'Ind',
                                                                        'Number': 'Sing',
                                                                        'Person': '3',
                                                                        'Tense': 'Pres',
                                                                        'VerbForm': 'Fin',
                                                                        'Voice': 'Mid'
                                                           }}),
                                   ('быстрый', 'быстрее', {'source_token': '',
                                                           'pos_tag': 'ADV',
                                                           'features': {'Degree': 'Cmp'}}),


                                   ])
def test_ru_inflect_token(resource_ru_inflector, resource_ru_inflector_force, param):
    source, res, morpho = param
    infl = resource_ru_inflector
    force_infl = resource_ru_inflector_force
    if infl.inflect_token(source, morpho) is not None:
        assert infl.inflect_token(source, morpho) == res
    assert force_infl.inflect_token(source, morpho) == res

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
    assert infl.get_lemma_verb(source) == lemma

@pytest.mark.parametrize('source_lemma_morph', [('apples', 'apple', {'pos_tag': 'NOUN', 'features': {}}),
                                                ('hardest', 'hard', {'pos_tag': 'ADJ', 'features': {}}),
                                                ('hardly', 'hardly', {'pos_tag': 'ADV', 'features': {}}),
                                                ('phenomena', 'phenomenon', {'pos_tag': 'NOUN', 'features': {}}),
                                                ('os', 'os', {'pos_tag': 'NOUN', 'features': {'Number': 'Sing'}}),
                                                ('colder', 'cold', {'pos_tag': 'ADJ', 'features': {}})])
def test_en_get_lemma_form(resource_en_inflector, source_lemma_morph):
    infl = resource_en_inflector
    source, lemma, morph = source_lemma_morph
    assert infl.get_lemma_form(source, morph) == lemma

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
    conv = VerbConvertorUD20Patternen()
    return conv

@pytest.mark.parametrize('source_pattern', [({'source_token': '',
                                              'pos_tag': 'AUX',
                                              'features': {'Mood': 'Ind',
                                                           'Number': 'Sing',
                                                           'Person': '3',
                                                           'Tense': 'Past',
                                                           'VerbForm': 'Fin'}},
                                             ('past', 3, 'singular', 'indicative', 'imperfective')),
                                            ({'source_token': '',
                                              'pos_tag': 'AUX',
                                              'features': {'Mood': 'Ind',
                                                           'Tense': 'Past',
                                                           'VerbForm': 'Fin'}},
                                             ('past', None, None, 'indicative', 'imperfective')),
                                            ({'source_token': '',
                                              'pos_tag': 'VERB',
                                              'features': {'Mood': 'Ind',
                                                           'Number': 'Sing',
                                                           'Person': '3',
                                                           'Tense': 'Pres',
                                                           'VerbForm': 'Fin'}},
                                             ('present', 3, 'singular', 'indicative', 'imperfective'))])
def test_convert_verb_ud20_to_pattern(resource_convertor, source_pattern):
    conv = resource_convertor
    source, pattern = source_pattern
    assert conv.convert(source) == pattern

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

