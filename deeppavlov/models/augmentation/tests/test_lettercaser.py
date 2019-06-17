import pytest
from deeppavlov.models.augmentation.utils.lettercaser import *


@pytest.fixture()
def resource_lettercaser():
    lett = Lettercaser()
    return lett


@pytest.fixture()
def resource_lettercaser_notdefault():
    lett = Lettercaser(cases={"lower": lambda x: x.lower(),
                                "capitalize": lambda x: x.capitalize(),
                                "upper": lambda x: x.upper(),
                                "spec": lambda x: x[:-1].lower()+x[-1].upper()},
                       default_case=lambda x: x.upper())
    return lett


@pytest.mark.parametrize(('tokens', 'res'), [(['Abs', 'bsd', 'VVV', 'vVVv', 'cCC', 'CccC'],
                                              ['capitalize', 'lower', 'upper', None, None, None]),
                                             (['dFv', 'dsa', 'CcC'],
                                              [None, 'lower', None])])
def test_get_cases(resource_lettercaser, tokens, res):
    lett = resource_lettercaser
    assert [lett.determine_lettercase(i) for i in tokens] == res


@pytest.mark.parametrize(('tokens', 'cases', 'res'), [(['abs', 'BSD', 'VvV', 'vvVv', 'cCc', 'cccc'],
                                                       ['capitalize', 'lower', 'upper', None, None, None],
                                                       ['Abs', 'bsd', 'VVV', 'vvvv', 'ccc', 'cccc'])])
def test_rest_cases(resource_lettercaser, tokens, cases, res):
    lett = resource_lettercaser
    assert [lett.put_in_lettercase(t, c) for t, c in zip(tokens, cases)] == res


@pytest.mark.parametrize(('tokens', 'res'), [(['Abs', 'bsd', 'VVV', 'vVVv', 'ccC', 'CccC'],
                                              ['capitalize', 'lower', 'upper', None, 'spec', None]),
                                             ])
def test_get_cases_extra(resource_lettercaser_notdefault, tokens, res):
    lett = resource_lettercaser_notdefault
    assert [lett.determine_lettercase(i) for i in tokens] == res


@pytest.mark.parametrize(('tokens', 'cases', 'res'), [(['abs', 'BSD', 'VvV', 'vvVv', 'cCc', 'cccc'],
                                                       ['capitalize', 'lower', 'upper', None, 'spec', None],
                                                       ['Abs', 'bsd', 'VVV', 'VVVV', 'ccC', 'CCCC'])])
def test_rest_cases_extra(resource_lettercaser_notdefault, tokens, cases, res):
    lett = resource_lettercaser_notdefault
    assert [lett.put_in_lettercase(t, c) for t, c in zip(tokens, cases)] == res