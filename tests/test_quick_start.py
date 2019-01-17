import io
import json
import logging
import os
import pickle
import signal
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Union

import pytest
import pexpect
import pexpect.popen_spawn
import requests
from urllib.parse import urljoin

import deeppavlov
from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.download import deep_download
from deeppavlov.core.data.utils import get_all_elems_from_json
from deeppavlov.core.common.paths import get_settings_path
from utils.server_utils.server import get_server_params, SERVER_CONFIG_FILENAME


tests_dir = Path(__file__).parent
test_configs_path = tests_dir / "deeppavlov" / "configs"
src_dir = Path(deeppavlov.__path__[0]) / "configs"
test_src_dir = tests_dir / "test_configs"
download_path = tests_dir / "download"

cache_dir: Path = None
if not os.getenv('DP_PYTEST_NO_CACHE'):
    cache_dir = tests_dir / 'download_cache'

api_port = os.getenv('DP_PYTEST_API_PORT')

TEST_MODES = ['IP',  # test_interacting_pretrained_model
              'TI',  # test_consecutive_training_and_interacting
              ]

ALL_MODES = ('IP', 'TI')

ONE_ARGUMENT_INFER_CHECK = ('Dummy text', None)
TWO_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', None)
FOUR_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', 'Dummy text', 'Dummy_text', None)

# Mapping from model name to config-model_dir-ispretrained and corresponding queries-response list.
PARAMS = {
    "ecommerce_skill": {
        ("ecommerce_skill/bleu_retrieve.json", "ecommerce_skill_bleu", ALL_MODES): [('Dummy text', '[]', '{}', None)],
        ("ecommerce_skill/tfidf_retrieve.json", "ecommerce_skill_tfidf", ALL_MODES): [('Dummy text', '[]', '{}', None)]
    },
    "faq": {
        ("faq/tfidf_logreg_en_faq.json", "faq_tfidf_logreg_en", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("faq/tfidf_autofaq.json", "faq_tfidf_cos", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("faq/tfidf_logreg_autofaq.json", "faq_tfidf_logreg", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("faq/fasttext_avg_autofaq.json", "faq_fasttext_avg", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("faq/fasttext_tfidf_autofaq.json", "faq_fasttext_tfidf", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK]
    },
    "spelling_correction": {
        ("spelling_correction/brillmoore_wikitypos_en.json", "error_model", ALL_MODES):
            [
                ("helllo", "hello"),
                ("datha", "data")
            ],
        ("spelling_correction/brillmoore_kartaslov_ru.json", "error_model", ('IP',)):
            [
                ("преведствую", "приветствую"),
                ("я джва года дду эту игру", "я два года жду эту игру")
            ],
        ("spelling_correction/levenshtein_corrector_ru.json", "error_model", ('IP',)):
            [
                ("преветствую", "приветствую"),
                ("Я джва года хочу такую игру", "я два года хочу такую игру")
            ]
    },
    "go_bot": {
        ("go_bot/gobot_dstc2.json", "gobot_dstc2", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("go_bot/gobot_dstc2_best.json", "gobot_dstc2_best", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("go_bot/gobot_dstc2_minimal.json", "gobot_dstc2_minimal", ('TI',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "classifiers": {
        ("classifiers/intents_dstc2.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_dstc2_big.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/insults_kaggle.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_twitter.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_twitter_preproc.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/topic_ag_news.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_cnn.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_elmo.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/yahoo_convers_vs_info.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "snips": {
        ("classifiers/intents_snips.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_big.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_bigru.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_bilstm.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_bilstm_bilstm.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_bilstm_cnn.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_bilstm_proj_layer.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_bilstm_self_add_attention.json", "classifiers", ('TI',)):
            [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_bilstm_self_mult_attention.json", "classifiers", ('TI',)):
            [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_cnn_bilstm.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_sklearn.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_snips_tfidf_weighted.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "sample": {
        ("classifiers/intents_sample_csv.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_sample_json.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "ner": {
        ("ner/ner_conll2003.json", "ner_conll2003", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_dstc2.json", "slotfill_dstc2", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes.json", "ner_ontonotes", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_few_shot_ru_simulate.json", "ner_fs", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_rus.json", "ner_rus", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/slotfill_dstc2.json", "slotfill_dstc2", ('IP',)):
            [
                ("chinese food", "{'food': 'chinese'}"),
                ("in the west part", "{'area': 'west'}"),
                ("moderate price range", "{'pricerange': 'moderate'}")
            ]
    },
    "elmo_embedder": {
        ("elmo_embedder/elmo_ru-news.json", "elmo_embedder_ru-news", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
    },
    "elmo_model": {
        ("elmo/elmo-1b-benchmark_test.json", "elmo-1b-benchmark_test", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
    },

    "ranking": {
        ("ranking/ranking_insurance_test.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_insurance_interact_test.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_test.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_interact_test.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt_test.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt_interact_test.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_test.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_interact_test.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_pretrain.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_tune.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_tune_interact.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_elmo.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_elmo_interact.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp_test.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp_bilstm_interact_test.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp_bilstm_test.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp_interact_test.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "doc_retrieval": {
        ("doc_retrieval/en_ranker_tfidf_wiki_test.json", "doc_retrieval", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("doc_retrieval/ru_ranker_tfidf_wiki_test.json", "doc_retrieval", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("doc_retrieval/en_ranker_pop_wiki_test.json", "doc_retrieval", ('TI',)): [
            ONE_ARGUMENT_INFER_CHECK]
    },
    "squad": {
        ("squad/squad.json", "squad_model", ALL_MODES): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_ru.json", "squad_model_ru", ALL_MODES): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/multi_squad_noans.json", "multi_squad_noans", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK]
    },
    "seq2seq_go_bot": {
        ("seq2seq_go_bot/bot_kvret_train.json", "seq2seq_go_bot", ('TI',)):
        [
           ("will it snow on tuesday?",
            "f78cf0f9-7d1e-47e9-aa45-33f9942c94be",
            "",
            "",
            "",
            None)
        ],
        ("seq2seq_go_bot/bot_kvret.json", "seq2seq_go_bot", ('IP',)):
        [
           ("will it snow on tuesday?",
            "f78cf0f9-7d1e-47e9-aa45-33f9942c94be",
            None)
        ]
    },
    "odqa": {
        ("odqa/en_odqa_infer_wiki_test.json", "odqa", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("odqa/ru_odqa_infer_wiki_test.json", "odqa", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("odqa/en_odqa_pop_infer_wiki_test.json", "odqa", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "morpho_tagger": {
        ("morpho_tagger/UD2.0/morpho_en.json", "morpho_tagger_en", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("morpho_tagger/UD2.0/morpho_ru_syntagrus_pymorphy.json", "morpho_tagger_pymorphy", ALL_MODES):
            [ONE_ARGUMENT_INFER_CHECK],
        ("morpho_tagger/UD2.0/morpho_ru_syntagrus.json", "morpho_tagger_pymorphy", ALL_MODES):
            [ONE_ARGUMENT_INFER_CHECK]
    }
}

MARKS = {"gpu_only": ["squad"], "slow": ["error_model", "go_bot", "squad"]}  # marks defined in pytest.ini

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
        m_name = config_ref.split('/')[-2]
        config_ref = '/'.join(config_ref.split('/')[-2:])

        test_configs_path.joinpath(m_name).mkdir(exist_ok=True)
        if not test_configs_path.joinpath(config_ref).exists():
            download_config(config_ref)

    # Update config for testing
    config.setdefault('train', {}).setdefault('pytest_epochs', 1)
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


def _serialize(config):
    chainer = build_model(config, download=True)
    return chainer.serialize()


def _deserialize(config, raw_bytes, examples):
    chainer = build_model(config, serialized=raw_bytes)
    for *query, expected_response in examples:
        query = [[q] for q in query]
        actual_response = chainer(*query)
        if expected_response is not None:
            if actual_response is not None and len(actual_response) > 0:
                actual_response = actual_response[0]
            assert expected_response == str(actual_response), \
                f"Error in interacting with {model_dir} ({conf_file}): {query}"


@pytest.mark.parametrize("model,conf_file,model_dir,mode", TEST_GRID, scope='class')
class TestQuickStart(object):
    @staticmethod
    def interact(config_path, model_directory, qr_list=None):
        qr_list = qr_list or []
        logfile = io.BytesIO(b'')
        p = pexpect.popen_spawn.PopenSpawn(' '.join([sys.executable, "-m", "deeppavlov", "interact", str(config_path)]),
                                           timeout=None, logfile=logfile)
        try:
            for *query, expected_response in qr_list:  # works until the first failed query
                for q in query:
                    p.expect("::")
                    p.sendline(q)

                p.expect(">> ")
                if expected_response is not None:
                    actual_response = p.readline().decode().strip()
                    assert expected_response == actual_response,\
                        f"Error in interacting with {model_directory} ({config_path}): {query}"

            p.expect("::")
            p.sendline("quit")
            p.readlines()
            if p.wait() != 0:
                raise RuntimeError('Error in quitting from deep.py: \n{}'.format(logfile.getvalue().decode()))
        except pexpect.exceptions.EOF:
            raise RuntimeError('Got unexpected EOF: \n{}'.format(logfile.getvalue().decode()))

    @staticmethod
    def interact_api(config_path):
        server_conf_file = get_settings_path() / SERVER_CONFIG_FILENAME

        server_params = get_server_params(server_conf_file, config_path)
        model_args_names = server_params['model_args_names']

        url_base = 'http://{}:{}/'.format(server_params['host'], api_port or server_params['port'])
        url = urljoin(url_base.replace('http://0.0.0.0:', 'http://127.0.0.1:'), server_params['model_endpoint'])

        post_headers = {'Accept': 'application/json'}

        post_payload = {}
        for arg_name in model_args_names:
            arg_value = str(' '.join(['qwerty'] * 10))
            post_payload[arg_name] = [arg_value]

        logfile = io.BytesIO(b'')
        args = [sys.executable, "-m", "deeppavlov", "riseapi", str(config_path)]
        if api_port:
            args += ['-p', str(api_port)]
        p = pexpect.popen_spawn.PopenSpawn(' '.join(args),
                                           timeout=None, logfile=logfile)
        try:
            p.expect(url_base)
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

    def test_interacting_pretrained_model(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            config_file_path = str(test_configs_path.joinpath(conf_file))
            install_config(config_file_path)
            deep_download(config_file_path)

            self.interact(test_configs_path / conf_file, model_dir, PARAMS[model][(conf_file, model_dir, mode)])
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_interacting_pretrained_model_api(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            self.interact_api(test_configs_path / conf_file)

            if 'TI' not in mode:
                shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_serialization(self, model, conf_file, model_dir, mode):
        if 'IP' not in mode:
            return pytest.skip("Unsupported mode: {}".format(mode))

        config_file_path = test_configs_path / conf_file

        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_serialize, config_file_path)
        raw_bytes = f.result()

        serialized: list = pickle.loads(raw_bytes)
        if not any(serialized):
            pytest.skip("Serialization not supported: {}".format(conf_file))
            return
        serialized.clear()

        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_deserialize, config_file_path, raw_bytes, PARAMS[model][(conf_file, model_dir, mode)])

        exc = f.exception()
        if exc is not None:
            raise exc

    def test_consecutive_training_and_interacting(self, model, conf_file, model_dir, mode):
        if 'TI' in mode:
            c = test_configs_path / conf_file
            model_path = download_path / model_dir

            if 'IP' not in mode:
                config_path = str(test_configs_path.joinpath(conf_file))
                install_config(config_path)
                deep_download(config_path)
            shutil.rmtree(str(model_path),  ignore_errors=True)

            logfile = io.BytesIO(b'')
            p = pexpect.popen_spawn.PopenSpawn(sys.executable + " -m deeppavlov train " + str(c), timeout=None,
                                               logfile=logfile)
            p.readlines()
            if p.wait() != 0:
                raise RuntimeError('Training process of {} returned non-zero exit code: \n{}'
                                   .format(model_dir, logfile.getvalue().decode()))
            self.interact(c, model_dir)

            shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))


def test_crossvalidation():
    model_dir = 'faq'
    conf_file = 'cv/cv_tfidf_autofaq.json'

    download_config(conf_file)

    c = test_configs_path / conf_file
    model_path = download_path / model_dir

    install_config(c)
    deep_download(c)
    shutil.rmtree(str(model_path),  ignore_errors=True)

    logfile = io.BytesIO(b'')
    p = pexpect.popen_spawn.PopenSpawn(sys.executable + f" -m deeppavlov crossval {c} --folds 2",
                                       timeout=None, logfile=logfile)
    p.readlines()
    if p.wait() != 0:
        raise RuntimeError('Training process of {} returned non-zero exit code: \n{}'
                           .format(model_dir, logfile.getvalue().decode()))

    shutil.rmtree(str(download_path), ignore_errors=True)


def test_param_search():
    model_dir = 'faq'
    conf_file = 'paramsearch/tfidf_logreg_autofaq_psearch.json'

    download_config(conf_file)

    c = test_configs_path / conf_file
    model_path = download_path / model_dir

    install_config(c)
    deep_download(c)

    shutil.rmtree(str(model_path),  ignore_errors=True)

    logfile = io.BytesIO(b'')
    p = pexpect.popen_spawn.PopenSpawn(sys.executable + f" -m deeppavlov.paramsearch {c} --folds 2",
                                       timeout=None, logfile=logfile)
    p.readlines()
    if p.wait() != 0:
        raise RuntimeError('Training process of {} returned non-zero exit code: \n{}'
                           .format(model_dir, logfile.getvalue().decode()))

    shutil.rmtree(str(download_path), ignore_errors=True)


def test_evolving():
    model_dir = 'evolution'
    conf_file = 'evolution/evolve_intents_snips.json'
    download_config(conf_file)

    c = test_configs_path / conf_file
    model_path = download_path / model_dir

    install_config(c)
    deep_download(c)

    shutil.rmtree(str(model_path), ignore_errors=True)

    logfile = io.BytesIO(b'')
    p = pexpect.popen_spawn.PopenSpawn(sys.executable + f" -m deeppavlov.evolve {c} --iterations 1 --p_size 1",
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
        downloads_urls |= {d if isinstance(d, str) else d['url'] for d in
                           config.get('metadata', {}).get('download', [])}
    downloads_urls = [url + '.md5' for url in downloads_urls if url.startswith(url_root)]
    messages = []

    logging.getLogger("urllib3").setLevel(logging.WARNING)

    for url in downloads_urls:
        status = requests.get(url).status_code
        if status != 200:
            messages.append(f'got status_code {status} for {url}')
    if messages:
        raise RuntimeError('\n'.join(messages))
