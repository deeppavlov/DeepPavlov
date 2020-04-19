import io
import json
import logging
import os
import pickle
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
from deeppavlov.core.commands.utils import parse_config
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

api_port = os.getenv('DP_PYTEST_API_PORT')
if api_port is not None:
    api_port = int(api_port)

TEST_MODES = ['IP',  # test_inferring_pretrained_model
              'TI',  # test_consecutive_training_and_inferring
              'SR',  # test_serialization
              ]

ALL_MODES = ('IP', 'TI', 'SR')

ONE_ARGUMENT_INFER_CHECK = ('Dummy text', None)
TWO_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', None)
FOUR_ARGUMENTS_INFER_CHECK = ('Dummy text', 'Dummy text', 'Dummy text', 'Dummy_text', None)

# Mapping from model name to config-model_dir-ispretrained and corresponding queries-response list.
PARAMS = {
    "ecommerce_skill": {
        ("ecommerce_skill/bleu_retrieve.json", "ecommerce_skill_bleu", ALL_MODES): [('Dummy text', [], {}, None)],
        ("ecommerce_skill/tfidf_retrieve.json", "ecommerce_skill_tfidf", ALL_MODES): [('Dummy text', [], {}, None)]
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
        ("go_bot/gobot_dstc2_minimal.json", "gobot_dstc2_minimal", ('TI',)): [([{"text": "the weather is clooudy and gloooomy"}], None)]
    },
    "classifiers": {
        ("classifiers/paraphraser_bert.json", "classifiers", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/paraphraser_rubert.json", "classifiers", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK],
        ("classifiers/insults_kaggle_bert.json", "classifiers", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/insults_kaggle_conv_bert.json", "classifiers", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_bert.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_dstc2_bert.json", "classifiers", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_dstc2.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/intents_dstc2_big.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/insults_kaggle.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_twitter.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_twitter_bert_emb.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_twitter_preproc.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/topic_ag_news.json", "classifiers", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_cnn.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_elmo_twitter_cnn.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/rusentiment_bigru_superconv.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/yahoo_convers_vs_info.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/ru_obscenity_classifier.json", "classifiers", ('IP',)):
            [
                ("Ну и сука же она", True),
                ("я два года жду эту игру", False)
            ],
        ("classifiers/sentiment_sst_conv_bert.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_sst_multi_bert.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_yelp_conv_bert.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_yelp_multi_bert.json", "classifiers", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_imdb_bert.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("classifiers/sentiment_imdb_conv_bert.json", "classifiers", ('TI',)): [ONE_ARGUMENT_INFER_CHECK]
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
        ("ner/conll2003_m1.json", "conll2003_m1", ('IP', 'TI')): [
            (["Peter", "Blackburn"], ["NNP", "NNP"], None)],
        ("ner/vlsp2016_full.json", "vlsp2016_full", ('IP', 'TI')): [
            (["Hương", "tự_tin"], ["NNP", "V"], ["B-NP", "B-VP"], None)],
        ("ner/ner_conll2003_bert.json", "ner_conll2003_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes_bert.json", "ner_ontonotes_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes_bert_mult.json", "ner_ontonotes_bert_mult", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_rus_bert.json", "ner_rus_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_conll2003.json", "ner_conll2003", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_dstc2.json", "slotfill_dstc2", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes.json", "ner_ontonotes", ALL_MODES): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_ontonotes_bert_emb.json", "ner_ontonotes_bert_emb", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_few_shot_ru_simulate.json", "ner_fs", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/ner_rus.json", "ner_rus", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ner/slotfill_dstc2.json", "slotfill_dstc2", ('IP',)):
            [
                ("chinese food", {'food': 'chinese'}),
                ("in the west part", {'area': 'west'}),
                ("moderate price range", {'pricerange': 'moderate'})
            ]
    },
    "sentence_segmentation": {
        ("sentence_segmentation/sentseg_dailydialog.json", "sentseg_dailydialog", ('IP', 'TI')): [
            (["hey", "alexa", "how", "are", "you"], None)]
    },
    "kbqa": {
        ("kbqa/kbqa_rus.json", "kbqa", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "elmo_embedder": {
        ("embedder/elmo_ru_news.json", "embedder_ru_news", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
    },
    "elmo_model": {
        ("elmo/elmo_1b_benchmark_test.json", "elmo_1b_benchmark_test", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
    },
    "ranking": {
        ("ranking/ranking_insurance.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_insurance_interact.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_interact.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt_interact.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_interact.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_pretrain.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_tune.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_tune_interact.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_paraphraser_elmo.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_elmo_interact.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp_bilstm_interact.json", "ranking",
         ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp_bilstm.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/paraphrase_ident_qqp_interact.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_bert_uncased.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_bert_sep.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_bert_sep_interact.json", "ranking", ('IP',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v1_mt_word2vec_smn.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v1_mt_word2vec_dam.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v1_mt_word2vec_dam_transformer.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt_word2vec_smn.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt_word2vec_dam.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt_word2vec_dam_transformer.json", "ranking", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("ranking/ranking_ubuntu_v2_mt_word2vec_dam_transformer.json", "ranking", ('IP',)):
            [(' & & & & & & & & bonhoeffer  whar drives do you want to mount what &  i have an ext3 usb drive  '
              '& look with fdisk -l & hello there & fdisk is all you need',
              None)]
    },
    "doc_retrieval": {
        ("doc_retrieval/en_ranker_tfidf_wiki_test.json", "doc_retrieval", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("doc_retrieval/ru_ranker_tfidf_wiki_test.json", "doc_retrieval", ('TI',)): [ONE_ARGUMENT_INFER_CHECK],
        ("doc_retrieval/en_ranker_pop_wiki_test.json", "doc_retrieval", ('TI',)): [
            ONE_ARGUMENT_INFER_CHECK]
    },
    "squad": {
        ("squad/squad_ru_bert.json", "squad_ru_bert", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_ru_bert_infer.json", "squad_ru_bert_infer", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_ru_rubert.json", "squad_ru_rubert", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_ru_rubert_infer.json", "squad_ru_rubert_infer", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_bert.json", "squad_bert", ('IP', 'TI')): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_bert_infer.json", "squad_bert_infer", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad.json", "squad_model", ALL_MODES): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_ru.json", "squad_model_ru", ALL_MODES): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/multi_squad_noans.json", "multi_squad_noans", ('IP',)): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_zh_bert_mult.json", "squad_zh_bert_mult", ALL_MODES): [TWO_ARGUMENTS_INFER_CHECK],
        ("squad/squad_zh_bert_zh.json", "squad_zh_bert_zh", ALL_MODES): [TWO_ARGUMENTS_INFER_CHECK]
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
        ("morpho_tagger/UD2.0/morpho_en.json", "morpho_en", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("morpho_tagger/UD2.0/morpho_ru_syntagrus_pymorphy_lemmatize.json", "morpho_tagger_pymorphy", ('IP', 'TI')):
            [ONE_ARGUMENT_INFER_CHECK],
        ("morpho_tagger/BERT/morpho_ru_syntagrus_bert.json", "morpho_tagger_bert", ('IP', 'TI')):
            [ONE_ARGUMENT_INFER_CHECK]
    },
    "syntax_tagger": {
        ("syntax/syntax_ru_syntagrus_bert.json", "syntax_ru_bert", ('IP', 'TI')): [ONE_ARGUMENT_INFER_CHECK],
        ("syntax/ru_syntagrus_joint_parsing.json", "syntax_ru_bert", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
    },
    "nemo": {
        ("nemo/tts2asr_test.json", "nemo", ('IP',)): [ONE_ARGUMENT_INFER_CHECK]
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


def _serialize(config):
    chainer = build_model(config, download=True)
    return chainer.serialize()


def _infer(config, inputs, download=False):
    chainer = build_model(config, download=download)
    return chainer(*inputs) if inputs else []


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
    def infer(config_path, qr_list=None, check_outputs=True):

        *inputs, expected_outputs = zip(*qr_list) if qr_list else ([],)
        with ProcessPoolExecutor(max_workers=1) as executor:
            f = executor.submit(_infer, config_path, inputs)
        outputs = f.result()

        if check_outputs:
            errors = ';'.join([f'expected `{expected}` got `{output}`'
                               for output, expected in zip(outputs, expected_outputs)
                               if expected is not None and expected != output])
            if errors:
                raise RuntimeError(f'Unexpected results for {config_path}: {errors}')

    @staticmethod
    def infer_api(config_path):
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

            model_args_names = get_response.json()
            post_payload = dict()
            for arg_name in model_args_names:
                arg_value = ' '.join(['qwerty'] * 10)
                post_payload[arg_name] = [arg_value]

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
                s.settimeout(60)
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
            self.infer_api(test_configs_path / conf_file)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_inferring_pretrained_model_socket(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            self.infer_socket(test_configs_path / conf_file, 'TCP')

            if 'TI' not in mode:
                shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip(f"Unsupported mode: {mode}")

    def test_serialization(self, model, conf_file, model_dir, mode):
        if 'SR' not in mode:
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
    conf_file = 'cv/cv_tfidf_autofaq.json'

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


def test_param_search():
    model_dir = 'faq'
    conf_file = 'paramsearch/tfidf_logreg_autofaq_psearch.json'

    download_config(conf_file)

    c = test_configs_path / conf_file
    model_path = download_path / model_dir

    install_config(c)
    deep_download(c)

    shutil.rmtree(str(model_path), ignore_errors=True)

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
