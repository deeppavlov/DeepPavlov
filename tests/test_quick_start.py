import io
import json
from pathlib import Path
import shutil

import pytest
import pexpect

from deeppavlov.download import deep_download

tests_dir = Path(__file__, '..').resolve()
test_configs_path = tests_dir / "deeppavlov" / "configs"
download_path = tests_dir / "download"

TEST_MODES = ['IP',  # test_interacting_pretrained_model
              'TI',  # test_consecutive_training_and_interacting
              'DE'  # test_downloaded_model_existence
              ]

ALL_MODES = ('DE', 'IP', 'TI')

# Mapping from model name to config-model_dir-ispretrained and corresponding queries-response list.
PARAMS = {"error_model": {("error_model/brillmoore_wikitypos_en.json", "error_model", ALL_MODES):
                              [
                                  ("helllo", "hello"),
                                  ("datha", "data")
                              ],
                          ("error_model/brillmoore_kartaslov_ru.json", "error_model", ALL_MODES): []},
          "go_bot": {("go_bot/gobot_dstc2.json", "gobot_dstc2", ALL_MODES): [],
                     ("go_bot/gobot_dstc2_best.json", "gobot_dstc2_best", ALL_MODES): [],
                     ("go_bot/gobot_dstc2_minimal.json", "gobot_dstc2_minimal", ('TI',)): [],
                     ("go_bot/gobot_dstc2_all.json", "gobot_dstc2_all", ('TI',)): []},
          "intents": {
              ("intents/intents_dstc2.json", "intents", ALL_MODES):  [],
              ("intents/intents_snips_bigru.json", "intents", ('TI')): [],
              ("intents/intents_snips_bilstm.json", "intents", ('TI')): [],
              ("intents/intents_snips_bilstm_bilstm.json", "intents", ('TI')): [],
              ("intents/intents_snips_bilstm_cnn.json", "intents", ('TI')): [],
              ("intents/intents_snips_bilstm_self_add_attention.json", "intents", ('TI')): [],
              ("intents/intents_snips_bilstm_self_mult_attention.json", "intents", ('TI')): [],
              ("intents/intents_snips_cnn_bilstm.json", "intents", ('TI')): []

          },
          "snips": {("intents/intents_snips.json", "intents", ('TI',)): []},
          "sample": {("intents/intents_sample_csv.json", "intents", ('TI',)): [],
                    ("intents/intents_sample_json.json", "intents", ('TI',)): []},
          "ner": {("ner/ner_conll2003.json", "ner_conll2003", ALL_MODES): [],
                  ("ner/ner_dstc2.json", "slotfill_dstc2", ALL_MODES): [],
                  ("ner/ner_ontonotes.json", "ner_ontonotes_senna", ('DE', 'IP')): [],
                  ("ner/ner_rus.json", "ner_rus", ('DE', 'IP')): [],
                  ("ner/slotfill_dstc2.json", "slotfill_dstc2", ALL_MODES):
                      [
                          ("chinese food", "{'food': 'chinese'}"),
                          ("in the west part", "{'area': 'west'}"),
                          ("moderate price range", "{'pricerange': 'moderate'}")
                      ]
                  },
          "ranking": {("ranking/insurance_config.json", "ranking", ALL_MODES): []},
          "squad": {("squad/squad.json", "squad_model", ALL_MODES): [],
                    ("squad/squad_ru.json", "squad_model_ru", ALL_MODES): []},
          "seq2seq_go_bot": {("seq2seq_go_bot/bot_kvret.json", "seq2seq_go_bot", ALL_MODES): []},
          "odqa": {("odqa/ranker_test.json", "odqa", ALL_MODES): [],
                   ("odqa/odqa_infer_test.json", "odqa", ('DE', 'IP')): []}
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


def setup_module():
    src_dir = tests_dir.parent / "deeppavlov" / "configs"
    test_src_dir = tests_dir / "test_configs"
    shutil.rmtree(str(test_configs_path), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)
    test_configs_path.mkdir(parents=True)

    for m_name, conf_dict in PARAMS.items():
        test_configs_path.joinpath(m_name).mkdir()
        for (conf_file, _, _), _ in conf_dict.items():
            src_file = src_dir / conf_file
            if not src_file.is_file():
                src_file = test_src_dir / conf_file

            if not src_file.is_file():
                raise RuntimeError('Unexisting config file {}'.format(conf_file))

            with src_file.open() as fin:
                config = json.load(fin)
            if config.get("train"):
                config["train"]["epochs"] = 1
                for pytest_key in [k for k in config["train"] if k.startswith('pytest_')]:
                    config["train"][pytest_key[len('pytest_'):]] = config["train"].pop(pytest_key)
            config["deeppavlov_root"] = str(download_path)
            with (test_configs_path / conf_file).open("w") as fout:
                json.dump(config, fout)


def teardown_module():
    shutil.rmtree(str(test_configs_path.parent), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)


@pytest.mark.parametrize("model,conf_file,model_dir,mode", TEST_GRID, scope='class')
class TestQuickStart(object):

    @staticmethod
    def interact(conf_file, model_dir, qr_list=None):
        qr_list = qr_list or []
        logfile = io.BytesIO(b'')
        p = pexpect.spawn("python3", ["-m", "deeppavlov.deep", "interact", str(conf_file)], timeout=None,
                          logfile=logfile)
        try:
            for *query, expected_response in qr_list:  # works until the first failed query
                for q in query:
                    p.expect("::")
                    p.sendline(q)

                p.expect(">> ")
                actual_response = p.readline().decode().strip()
                assert expected_response == actual_response, f"Error in interacting with {model_dir} ({conf_file}): {query}"
            p.expect("::")
            p.sendline("quit")
            if p.expect(pexpect.EOF) != 0:
                logfile.seek(0)
                raise RuntimeError('Error in quitting from deep.py: \n{}'
                                   .format(''.join((line.decode() for line in logfile.readlines()))))
        except pexpect.exceptions.EOF:
            logfile.seek(0)
            raise RuntimeError('Got unexpected EOF: \n{}'
                               .format(''.join((line.decode() for line in logfile.readlines()))))

    def test_interacting_pretrained_model(self, model, conf_file, model_dir, mode):
        if 'IP' in mode:
            config_file_path = str(test_configs_path.joinpath(conf_file))
            deep_download(['-test', '-c', config_file_path])

            self.interact(test_configs_path / conf_file, model_dir, PARAMS[model][(conf_file, model_dir, mode)])

            if 'TI' not in mode:
                shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))

    def test_consecutive_training_and_interacting(self, model, conf_file, model_dir, mode):
        if 'TI' in mode:
            c = test_configs_path / conf_file
            model_path = download_path / model_dir

            if 'IP' not in mode:
                config_path = str(test_configs_path.joinpath(conf_file))
                deep_download(['-test', '-c', config_path])
            shutil.rmtree(str(model_path),  ignore_errors=True)

            logfile = io.BytesIO(b'')
            _, exitstatus = pexpect.run("python3 -m deeppavlov.deep train " + str(c), timeout=None, withexitstatus=True,
                                        logfile = logfile)
            if exitstatus != 0:
                logfile.seek(0)
                raise RuntimeError('Training process of {} returned non-zero exit code: \n{}'
                                   .format(model_dir, ''.join((line.decode() for line in logfile.readlines()))))
            self.interact(c, model_dir)

            shutil.rmtree(str(download_path), ignore_errors=True)
        else:
            pytest.skip("Unsupported mode: {}".format(mode))
