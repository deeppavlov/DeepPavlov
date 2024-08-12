import io
import json
import logging
import os
import shutil
import signal
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from struct import unpack
from time import sleep
from typing import Optional, Union
from urllib.parse import urljoin

import pexpect
import pexpect.popen_spawn
import pytest
import requests

import deeppavlov
from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config, parse_value_with_config
from deeppavlov.core.common.aliases import ALIASES
from deeppavlov.core.data.utils import get_all_elems_from_json
from deeppavlov.download import deep_download
from deeppavlov.utils.server import get_server_params
from deeppavlov.utils.socket import encode

tests_dir = Path(__file__).parent
test_configs_path = tests_dir / "deeppavlov" / "configs"
src_dir = Path(deeppavlov.__path__[0]) / "configs"
test_src_dir = tests_dir / "test_configs"
download_path = tests_dir / "download"

cache_dir: Optional[Path] = None
if not os.getenv('DP_PYTEST_NO_CACHE'):
    cache_dir = tests_dir / 'download_cache'

SKIP_TF = os.getenv('SKIP_TF', False)

api_port = os.getenv('DP_PYTEST_API_PORT')
if api_port is not None:
    api_port = int(api_port)

TEST_MODES = ['IP',  # test_inferring_pretrained_model
              'TI',  # test_consecutive_training_and_inferring
              ]

ALL_MODES = ('IP', 'TI')

ONE_ARGUMENT_INFER_CHECK = ('Dummy text', None)
TWO_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', None)
FOUR_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', 'Dummy text', 'Dummy_text', None)

LIST_ARGUMENTS_INFER_CHECK = (['Dummy text', 'Dummy text'], ['Dummy text', 'Dummy text'], None)

RECORD_ARGUMENTS_INFER_CHECK = ("Index", "Dummy query text", "Dummy passage text", "Dummy entity", 1, None)

# Mapping from model name to config-model_dir-ispretrained and corresponding queries-response list.
PARAMS = {
    "relation_extraction": {
        ("relation_extraction/re_docred.json", "relation_extraction", ('IP',)):
            [
                (
                    [["Barack", "Obama", "is", "married", "to", "Michelle", "Obama", ",", "born", "Michelle",
                      "Robinson", "."]],
                    [[[(0, 2)], [(5, 7), (9, 11)]]],
                    [["PER", "PER"]],
                    (
                        'P26',
                        'spouse'
                    )
                )
            ],
        ("relation_extraction/re_rured.json", "relation_extraction", ('IP',)):
            [
                (
                    [["Илон", "Маск", "живет", "в", "Сиэттле", "."]],
                    [[[(0, 2)], [(4, 6)]]],
                    [["PERSON", "CITY"]],
                    (
                        'P495',
                        'страна происхождения'
                    )
                ),
            ]
    },
    "faq": {
        ("faq/fasttext_logreg.json", "fasttext_logreg", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],  # TODO: add ru test
    },
    "spelling_correction": {
        ("spelling_correction/brillmoore_wikitypos_en.json", "error_model", ALL_MODES):
            [
                ("helllo", ("hello",)),
                ("datha", ("data",))
            ],
        ("spelling_correction/levenshtein_corrector_ru.json", "error_model", ('IP',)):
            [
                ("преветствую", ("приветствую",)),
                ("Я джва года хочу такую игру", ("я два года хочу такую игру",))
            ]
    },
    "classifiers": {
        ("classifiers/paraphraser_rubert.json", "classifiers", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/insults_kaggle_bert.json", "classifiers", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_bert.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_twitter.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_sst_conv_bert.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/glue/glue_mrpc_roberta.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/glue/glue_stsb_roberta.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/glue/glue_mnli_roberta.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/glue/glue_rte_roberta_mnli.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/glue/glue_cola_roberta.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/glue/glue_qnli_roberta.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/glue/glue_qqp_roberta.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/glue/glue_sst2_roberta.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/glue/glue_wnli_roberta.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/superglue/superglue_copa_roberta.json", "classifiers", ('TI',)): [LIST_ARGUMENTS_INFER_CHECK],
        ("classifiers/superglue/superglue_boolq_roberta_mnli.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/superglue/superglue_record_roberta.json", "classifiers", ('TI',)): [RECORD_ARGUMENTS_INFER_CHECK],
        ("classifiers/superglue/superglue_wic_bert.json", "classifiers", ('TI',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/topics_distilbert_base_uncased.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/few_shot_roberta.json", "classifiers", ('IP',)): [
            ('Dummy text', ['Dummy text Dummy text', 'Dummy class'], ('Dummy class',))
        ]
    },
    "distil": {
        ("classifiers/paraphraser_convers_distilrubert_2L.json", "distil", ('IP')): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/paraphraser_convers_distilrubert_6L.json", "distil", ('IP')): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/rusentiment_convers_distilrubert_2L.json", "distil", ('IP')): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_convers_distilrubert_6L.json", "distil", ('IP')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_rus_convers_distilrubert_2L.json", "distil", ('IP')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_rus_convers_distilrubert_6L.json", "distil", ('IP')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_case_agnostic_mdistilbert.json", "distil", ('IP')): [ONE_ARGUMENT_INFER_CHECK],
        ("squad/squad_ru_convers_distilrubert_2L.json", "distil", ('IP')): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_ru_convers_distilrubert_6L.json", "distil", ('IP')): [TWO_ARGUMENTS_INFER_CHECK]
    },
    "russian_super_glue": {
        ("russian_super_glue/russian_superglue_lidirus_rubert.json", "russian_super_glue", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_danetqa_rubert.json", "russian_super_glue", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_terra_rubert.json", "russian_super_glue", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_rcb_rubert.json", "russian_super_glue", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_russe_rubert.json", "russian_super_glue", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_rwsd_rubert.json", "russian_super_glue", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_muserc_rubert.json", "russian_super_glue", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_parus_rubert.json", "russian_super_glue", ('IP',)): [LIST_ARGUMENTS_INFER_CHECK],
        ("russian_super_glue/russian_superglue_rucos_rubert.json", "russian_super_glue", ('IP',)): [RECORD_ARGUMENTS_INFER_CHECK]
    },
    "multitask":{
        ("multitask/multitask_example.json", "multitask", ALL_MODES): [
            ('Dummy text',) + (('Dummy text', 'Dummy text'),) * 3 + ('Dummy text',) + (None,)],
        ("multitask/mt_glue.json", "multitask", ALL_MODES): [
            ('Dummy text',) * 2 + (('Dummy text', 'Dummy text'),) * 6 + (None,)]
    },
    "entity_extraction": {
        ("entity_extraction/entity_detection_en.json", "entity_extraction", ('IP',)):
            [
                ("Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.",
                 (['forrest gump', 'robert zemeckis', 'eric roth'],
                  [(0, 12), (48, 63), (79, 88)],
                  [[0, 1], [10, 11], [15, 16]],
                  ['WORK_OF_ART', 'PERSON', 'PERSON'],
                  [(0, 89)],
                  ['Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.'],
                  [0.8798, 0.9986, 0.9985]))
            ],
        ("entity_extraction/entity_detection_ru.json", "entity_extraction", ('IP',)):
            [
                ("Москва — столица России, центр Центрального федерального округа и центр Московской области.",
                 (['москва', 'россии', 'центрального федерального округа', 'московской области'],
                  [(0, 6), (17, 23), (31, 63), (72, 90)],
                  [[0], [3], [6, 7, 8], [11, 12]],
                  ['CITY', 'COUNTRY', 'LOC', 'LOC'],
                  [(0, 91)],
                  ['Москва — столица России, центр Центрального федерального округа и центр Московской области.'],
                  [0.8359, 0.938, 0.9917, 0.9803]))
            ],
        ("entity_extraction/entity_extraction_en.json", "entity_extraction", ('IP',)):
            [
                ("Forrest Gump is a comedy-drama film directed by Robert Zemeckis and written by Eric Roth.",
                 (['forrest gump', 'robert zemeckis', 'eric roth'],
                  ['WORK_OF_ART', 'PERSON', 'PERSON'],
                  [(0, 12), (48, 63), (79, 88)],
                  [['Q134773', 'Q552213', 'Q12016774'], ['Q187364', 'Q36951156'],
                   ['Q942932', 'Q89320386', 'Q89909683']],
                  [[[1.1, 110, 1.0], [1.1, 13, 0.73], [1.1, 8, 0.04]], [[1.1, 73, 1.0], [0.5, 52, 0.29]],
                   [[1.1, 37, 0.95], [1.1, 2, 0.35], [0.67, 2, 0.35]]],
                  [['Forrest Gump', 'Forrest Gump (novel)', ''], ['Robert Zemeckis', 'Welcome to Marwen'],
                   ['Eric Roth', '', '']],
                  [['Forrest Gump', 'Forrest Gump', 'Forrest Gump'], ['Robert Zemeckis', 'Welcome to Marwen'],
                   ['Eric Roth', 'Eric Roth', 'Eric W Roth']]))
            ],
        ("entity_extraction/entity_extraction_ru.json", "entity_extraction", ('IP',)):
            [
                ("Москва — столица России, центр Центрального федерального округа и центр Московской области.",
                 (['москва', 'россии', 'центрального федерального округа', 'московской области'],
                  ['CITY', 'COUNTRY', 'LOC', 'LOC'],
                  [(0, 6), (17, 23), (31, 63), (72, 90)],
                  [['Q649', 'Q1023006', 'Q2380475'], ['Q159', 'Q2184', 'Q139319'], ['Q190778', 'Q4504288', 'Q27557290'],
                   ['Q1697', 'Q4303932', 'Q24565285']],
                  [[[1.1, 200, 1.0], [1.0, 20, 0.0], [1.0, 18, 0.0]],
                   [[1.1, 200, 1.0], [1.0, 58, 1.0], [1.0, 29, 0.85]],
                   [[1.1, 200, 1.0], [0.67, 3, 0.92], [0.67, 3, 0.89]],
                   [[0.9, 200, 1.0], [0.9, 6, 0.83], [0.61, 8, 0.03]]],
                  [['Москва', 'Москоу (Канзас)', 'Москоу (Теннесси)'],
                   ['Россия', 'Российская Советская Федеративная Социалистическая Республика',
                    'Российская республика'],
                   ['Центральный федеральный округ', 'Центральный округ (Краснодар)', ''],
                   ['Московская область', 'Московская область (1917—1918)',
                    'Мостовский (Волгоградская область)']],
                  [['Москва', 'Москоу', 'Москоу'],
                   ['Россия', 'Российская Советская Федеративная Социалистическая Республика',
                    'Российская республика'],
                   ['Центральный федеральный округ', 'Центральный округ (Краснодар)', 'Центральный округ (Братск)'],
                   ['Московская область', 'Московская область', 'Мостовский']]))
            ]
    },
    "ner": {
        ("ner/ner_bert_base.json", "ner_bert_base", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_conll2003_bert.json", "ner_conll2003_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes_bert.json", "ner_ontonotes_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes_bert_mult.json", "ner_ontonotes_bert_mult", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_rus_bert.json", "ner_rus_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_collection3_bert.json", "ner_collection3_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_conll2003_deberta_crf.json", "ner_conll2003_deberta_crf", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes_deberta_crf.json", "ner_ontonotes_deberta_crf", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
    },
    "sentence_segmentation": {
        ("sentence_segmentation/sentseg_dailydialog_bert.json", "sentseg_dailydialog_bert", ('IP', 'TI')): [
            (["hey", "alexa", "how", "are", "you"], None)]
    },
    "kbqa": {
        ("kbqa/kbqa_cq_en.json", "kbqa", ('IP',)):
            [
                ("What is the currency of Sweden?",
                 ("Swedish krona", ["Q122922"], ["SELECT ?answer WHERE { wd:Q34 wdt:P38 ?answer. }"])),
                ("Where was Napoleon Bonaparte born?",
                 ("Ajaccio", ["Q40104"], ["SELECT ?answer WHERE { wd:Q517 wdt:P19 ?answer. }"])),
                ("When did the Korean War end?",
                 ("27 July 1953", ["+1953-07-27^^T"], ["SELECT ?answer WHERE { wd:Q8663 wdt:P582 ?answer. }"])),
                ("   ", ("Not Found", [], []))
            ],            
        ("kbqa/kbqa_cq_ru.json", "kbqa", ('IP',)):
            [
                ("Кто такой Оксимирон?",
                 ("российский рэп-исполнитель", ['российский рэп-исполнитель"@ru'],
                  ["SELECT ?answer WHERE { wd:Q4046107 wdt:P0 ?answer. }"])),
                ("Кто написал «Евгений Онегин»?",
                 ("Александр Сергеевич Пушкин", ["Q7200"], ["SELECT ?answer WHERE { wd:Q50948 wdt:P50 ?answer. }"])),
                ("абв", ("Not Found", [], []))
            ]
    },
    "ranking": {
        ("ranking/ranking_ubuntu_v2_torch_bert_uncased.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "doc_retrieval": {
        ("doc_retrieval/en_ranker_tfidf_wiki_test.json", "doc_retrieval", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("doc_retrieval/ru_ranker_tfidf_wiki_test.json", "doc_retrieval", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("doc_retrieval/en_ranker_pop_wiki_test.json", "doc_retrieval", ('TI',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "squad": {
        ("squad/squad_ru_bert.json", "squad_ru_bert", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_bert.json", "squad_bert", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK]
    },
    "odqa": {
        ("odqa/en_odqa_infer_wiki.json", "odqa", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("odqa/ru_odqa_infer_wiki.json", "odqa", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("odqa/en_odqa_pop_infer_wiki.json", "odqa", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "morpho_tagger": {
        ("morpho_syntax_parser/morpho_ru_syntagrus_bert.json", "morpho_tagger_bert", ('IP', 'TI')):
            [ONE_ARGUMENT_INFER_CHECK]
    },
    "syntax_tagger": {
        ("morpho_syntax_parser/syntax_ru_syntagrus_bert.json", "syntax_ru_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("morpho_syntax_parser/ru_syntagrus_joint_parsing.json", "syntax_ru_bert", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
    },
}

MARKS = {"gpu_only": ["squad"], "slow": ["error_model", "squad"]}  # marks defined in pytest.ini

TEST_GRID = []
for model in PARAMS.keys():
    for conf_file, model_dir, mode in PARAMS[model].keys():
        marks = []
        for mark in MARKS.keys():
            if model in MARKS[mark]:
                marks.append(eval("pytest.mark." + mark))
        grid_unit = pytest.param(model, conf_file, model_dir, mode, marks=marks)
        TEST_GRID.append(grid_unit)


def _override_with_test_values(item: Union[dict, list]) -> None:
    if isinstance(item, dict):
        keys = [k for k in item.keys() if k.startswith('pytest_')]
        for k in keys:
            item[k[len('pytest_'):]] = item.pop(k)
        item = item.values()

    for child in item:
        if isinstance(child, (dict, list)):
            _override_with_test_values(child)


def download_config(config_path):
    src_file = src_dir / config_path
    if not src_file.is_file():
        src_file = test_src_dir / config_path

    if not src_file.is_file():
        raise RuntimeError('No config file {}'.format(config_path))

    with src_file.open(encoding='utf8') as fin:
        config: dict = json.load(fin)

    # Download referenced config files
    config_references = get_all_elems_from_json(parse_config(config), 'config_path')
    for config_ref in config_references:
        splitted = config_ref.split("/")
        first_subdir_index = splitted.index("configs") + 1
        m_name = config_ref.split('/')[first_subdir_index]
        config_ref = '/'.join(config_ref.split('/')[first_subdir_index:])

        test_configs_path.joinpath(m_name).mkdir(exist_ok=True)
        if not test_configs_path.joinpath(config_ref).exists():
            download_config(config_ref)

    # Update config for testing
    config.setdefault('train', {}).setdefault('pytest_epochs', 1)
    config['train'].setdefault('pytest_max_batches', 2)
    config['train'].setdefault('pytest_max_test_batches', 2)
    _override_with_test_values(config)

    config_path = test_configs_path / config_path
    config_path.parent.mkdir(exist_ok=True, parents=True)
    with config_path.open("w", encoding='utf8') as fout:
        json.dump(config, fout)


def install_config(config_path):
    logfile = io.BytesIO(b'')
    p = pexpect.popen_spawn.PopenSpawn(sys.executable + " -m deeppavlov install " + str(config_path), timeout=None,
                                       logfile=logfile)
    p.readlines()
    if p.wait() != 0:
        raise RuntimeError('Installing process of {} returned non-zero exit code: \n{}'
                           .format(config_path, logfile.getvalue().decode()))


def setup_module():
    shutil.rmtree(str(test_configs_path), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)
    test_configs_path.mkdir(parents=True)

    for m_name, conf_dict in PARAMS.items():
        test_configs_path.joinpath(m_name).mkdir(exist_ok=True, parents=True)
        for (config_path, _, _), _ in conf_dict.items():
            download_config(config_path)

    os.environ['DP_ROOT_PATH'] = str(download_path)
    os.environ['DP_CONFIGS_PATH'] = str(test_configs_path)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['DP_CACHE_DIR'] = str(cache_dir.resolve())


def teardown_module():
    shutil.rmtree(str(test_configs_path.parent), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)

    if cache_dir:
        shutil.rmtree(str(cache_dir), ignore_errors=True)


def _infer(config, inputs, download=False):
    chainer = build_model(config, download=download)
    if inputs:
        prediction = chainer(*inputs)
        if len(chainer.out_params) == 1:
            prediction = [prediction]
    else:
        prediction = []
    return prediction


@pytest.mark.parametrize("model,conf_file,model_dir,mode", TEST_GRID, scope='class')
class TestQuickStart(object):
    @staticmethod
    def infer(config_path, qr_list=None, check_outputs=True):

        *inputs, expected_outputs = zip(*qr_list) if qr_list else ([],)
        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_infer, config_path, inputs)
        outputs = list(zip(*f.result()))

        if check_outputs:
            errors = ';'.join([f'expected `{expected}` got `{output}`'
                               for output, expected in zip(outputs, expected_outputs)
                               if expected is not None and expected != output])
            if errors:
                raise RuntimeError(f'Unexpected results for {config_path}: {errors}')

    @staticmethod
    def infer_api(config_path, qr_list):
        *inputs, expected_outputs = zip(*qr_list)
        server_params = get_server_params(config_path)

        url_base = 'http://{}:{}'.format(server_params['host'], api_port or server_params['port'])
        url = urljoin(url_base.replace('http://0.0.0.0:', 'http://127.0.0.1:'), server_params['model_endpoint'])

        post_headers = {'Accept': 'application/json'}

        logfile = io.BytesIO(b'')
        args = [sys.executable, "-m", "deeppavlov", "riseapi", str(config_path)]
        if api_port:
            args += ['-p', str(api_port)]
        p = pexpect.popen_spawn.PopenSpawn(' '.join(args),
                                           timeout=None, logfile=logfile)
        try:
            p.expect(url_base)

            get_url = urljoin(url_base.replace('http://0.0.0.0:', 'http://127.0.0.1:'), '/api')
            get_response = requests.get(get_url)
            response_code = get_response.status_code
            assert response_code == 200, f"GET /api request returned error code {response_code} with {config_path}"

            model_args_names = get_response.json()['in']
            post_payload = dict(zip(model_args_names, inputs))
            # TODO: remove this if from here and socket
            if 'docred' in str(config_path) or 'rured' in str(config_path):
                post_payload = {k: v[0] for k, v in post_payload.items()}
            post_response = requests.post(url, json=post_payload, headers=post_headers)
            response_code = post_response.status_code
            assert response_code == 200, f"POST request returned error code {response_code} with {config_path}"

        except pexpect.exceptions.EOF:
            raise RuntimeError('Got unexpected EOF: \n{}'.format(logfile.getvalue().decode()))

        finally:
            p.kill(signal.SIGTERM)
            p.wait()
            # if p.wait() != 0:
            #     raise RuntimeError('Error in shutting down API server: \n{}'.format(logfile.getvalue().decode()))

    @staticmethod
    def infer_socket(config_path, socket_type):
        socket_params = get_server_params(config_path)
        model_args_names = socket_params['model_args_names']

        host = socket_params['host']
        host = host.replace('0.0.0.0', '127.0.0.1')
        port = api_port or socket_params['port']

        socket_payload = {}
        for arg_name in model_args_names:
            arg_value = ' '.join(['qwerty'] * 10)
            socket_payload[arg_name] = [arg_value]

        if 'parus' in str(config_path):
            socket_payload = {k: [v] for k, v in socket_payload.items()}

        logfile = io.BytesIO(b'')
        args = [sys.executable, "-m", "deeppavlov", "risesocket", str(config_path), '--socket-type', socket_type]
        if socket_type == 'TCP':
            args += ['-p', str(port)]
            address_family = socket.AF_INET
            connect_arg = (host, port)
        else:
            address_family = socket.AF_UNIX
            connect_arg = socket_params['unix_socket_file']
        p = pexpect.popen_spawn.PopenSpawn(' '.join(args),
                                           timeout=None, logfile=logfile)
        try:
            p.expect(socket_params['socket_launch_message'])
            with socket.socket(address_family, socket.SOCK_STREAM) as s:
                try:
                    s.connect(connect_arg)
                except ConnectionRefusedError:
                    sleep(1)
                    s.connect(connect_arg)
                s.sendall(encode(socket_payload))
                s.settimeout(120)
                header = s.recv(4)
                body_len = unpack('<I', header)[0]
                data = bytearray()
                while len(data) < body_len:
                    chunk = s.recv(body_len - len(data))
                    if not chunk:
                        raise ValueError(f'header does not match body\nheader: {body_len}\nbody length: {len(data)}'
                                         f'data: {data}')
                    data.extend(chunk)
            try:
                resp = json.loads(data)
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Can't decode model response {data}")
            assert resp['status'] == 'OK', f"{socket_type} socket request returned status: {resp['status']}" \
                                           f" with {config_path}\n{logfile.getvalue().decode()}"

        except pexpect.exceptions.EOF:
            raise RuntimeError(f'Got unexpected EOF: \n{logfile.getvalue().decode()}')

        except json.JSONDecodeError:
            raise ValueError(f'Got JSON not serializable response from model: "{data}"\n{logfile.getvalue().decode()}')

        finally:
            p.kill(signal.SIGTERM)
            p.wait()

    def test_inferring_pretrained_model(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            config_file_path = str(test_configs_path.joinpath(conf_file))
            install_config(config_file_path)
            deep_download(config_file_path)

            self.infer(test_configs_path / conf_file, PARAMS[model][(conf_file, model_dir, mode)])
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_inferring_pretrained_model_api(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            self.infer_api(test_configs_path / conf_file, PARAMS[model][(conf_file, model_dir, mode)])
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_inferring_pretrained_model_socket(self, model, conf_file, model_dir, mode):
        pytest.skip(f"Disabled")
        if 'IP' in mode:
            self.infer_socket(test_configs_path / conf_file, 'TCP')

            if 'TI' not in mode:
                shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip(f"Unsupported mode: {mode}")


    def test_consecutive_training_and_inferring(self, model, conf_file, model_dir, mode):
        if 'TI' in mode:
            c = test_configs_path / conf_file
            model_path = download_path / model_dir

            if 'IP' not in mode:
                config_path = str(test_configs_path.joinpath(conf_file))
                install_config(config_path)
                deep_download(config_path)
            shutil.rmtree(str(model_path), ignore_errors=True)

            logfile = io.BytesIO(b'')
            p = pexpect.popen_spawn.PopenSpawn(sys.executable + " -m deeppavlov train " + str(c), timeout=None,
                                               logfile=logfile)
            p.readlines()
            if p.wait() != 0:
                raise RuntimeError('Training process of {} returned non-zero exit code: \n{}'
                                   .format(model_dir, logfile.getvalue().decode()))
            self.infer(c, PARAMS[model][(conf_file, model_dir, mode)], check_outputs=False)

            shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))


def test_crossvalidation():
    model_dir = 'faq'
    conf_file = 'faq/fasttext_logreg.json'

    download_config(conf_file)

    c = test_configs_path / conf_file
    model_path = download_path / model_dir

    install_config(c)
    deep_download(c)
    shutil.rmtree(str(model_path), ignore_errors=True)

    logfile = io.BytesIO(b'')
    p = pexpect.popen_spawn.PopenSpawn(sys.executable + f" -m deeppavlov crossval {c} --folds 2",
                                       timeout=None, logfile=logfile)
    p.readlines()
    if p.wait() != 0:
        raise RuntimeError('Training process of {} returned non-zero exit code: \n{}'
                           .format(model_dir, logfile.getvalue().decode()))

    shutil.rmtree(str(download_path), ignore_errors=True)


def test_hashes_existence():
    all_configs = list(src_dir.glob('**/*.json')) + list(test_src_dir.glob('**/*.json'))
    url_root = 'http://files.deeppavlov.ai/'
    downloads_urls = set()
    for config in all_configs:
        config = json.loads(config.read_text(encoding='utf-8'))
        # TODO: replace with get downloads from config
        # TODO: download only headers
        # TODO: make requests in async mode
        config_urls = {d if isinstance(d, str) else d['url'] for d in config.get('metadata', {}).get('download', [])}
        downloads_urls |= {parse_value_with_config(url, config) for url in config_urls}
    downloads_urls = [url + '.md5' for url in downloads_urls if url.startswith(url_root)]
    messages = []

    logging.getLogger("urllib3").setLevel(logging.WARNING)

    for url in downloads_urls:
        status = requests.get(url).status_code
        if status != 200:
            messages.append(f'got status_code {status} for {url}')
    if messages:
        raise RuntimeError('\n'.join(messages))


def test_aliases():
    configs = list(src_dir.glob('**/*.json'))
    config_names = [c.stem for c in configs]

    assert len(config_names) == len(set(config_names)), 'Some model names are duplicated'

    aliases_in_configs = set(ALIASES.keys()) & set(config_names)
    assert aliases_in_configs == set(), f'Following model(s) marked as deprecated but still present in configs list: ' \
                                        f'{", ".join(aliases_in_configs)}.'

    alias_targets_not_in_configs = set(ALIASES.values()) - set(config_names)
    assert alias_targets_not_in_configs == set(), f'Following model(s) marked as alias targets but there is no such ' \
                                                  f'config in the library: {", ".join(alias_targets_not_in_configs)}'
