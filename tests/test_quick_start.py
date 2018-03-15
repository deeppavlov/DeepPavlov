import pytest
from pathlib import Path
import pexpect
import json
import shutil


tests_dir = Path(__file__, '..').resolve()
test_configs_path = tests_dir / "configs"
download_path = tests_dir / "download"


# Mapping from model name to config-model_dir and corresponding query-response pairs.
PARAMS = {"error_model": {("configs/error_model/brillmoore_wikitypos_en.json", "error_model", True):
                              [
                                  ("helllo", "hello"),
                                  ("datha", "data")
                              ],
                          ("configs/error_model/brillmoore_kartaslov_ru.json", "error_model", True):
                              [

                              ]
                          },
          "go_bot": {("configs/go_bot/gobot_dstc2.json", "go_bot", True):
                         [

                         ],
                     # ("configs/go_bot/gobot_dstc2_minimal.json", "go_bot_minimal"):
                     #     [
                     #
                     #     ]
                     },
          "intents": {("configs/intents/intents_dstc2.json", "intents", True):  []},
          "snips": {("configs/intents/intents_snips.json", "intents", False): []},
          "sample": {("configs/intents/intents_sample_csv.json", "intents", False): [],
                    ("configs/intents/intents_sample_json.json", "intents", False): []},
          "ner": {("configs/ner/ner_conll2003.json", "ner_conll2003_model", True):
                      [
                          # ("Albert Einstein and Erwin Schrodinger", "['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER']"),
                          # ("Antananarivo is the capital of Madagascar", "['B-LOC', 'O', 'O', 'O', 'O', 'B-LOC']"),
                          # ("UN launches new global data collection tool to help reduce disaster",
                          #  "['B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']")
                      ],
                  ("configs/ner/ner_dstc2.json", "ner", True):
                      [
                          # ("chinese food", "['B-food', 'O']"),
                          # ("in the west part", "['O', 'O', 'B-area', 'O']"),
                          # ("moderate price range", "['B-pricerange', 'O', 'O']")
                      ],
                  ("configs/ner/slotfill_dstc2.json", "ner", True):
                      [
                          ("chinese food", "{'food': 'chinese'}"),
                          ("in the west part", "{'area': 'west'}"),
                          ("moderate price range", "{'pricerange': 'moderate'}")
                      ]
                }
        }


def setup_module():
    src_dir = tests_dir.parent / "deeppavlov"

    shutil.rmtree(str(test_configs_path), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)
    test_configs_path.mkdir()

    for m_name, conf_dict in PARAMS.items():
        test_configs_path.joinpath(m_name).mkdir()
        for (conf_file, _, _), _ in conf_dict.items():
            with (src_dir / conf_file).open() as fin:
                config = json.load(fin)
            try:
                config["train"]["epochs"] = 1
            except KeyError:
                pass
            config["deeppavlov_root"] = str(download_path)
            with (tests_dir / conf_file).open("w") as fout:
                json.dump(config, fout)


def teardown_module():
    shutil.rmtree(str(test_configs_path))
    shutil.rmtree(str(download_path))


def download(full=None):
    cmd = "python3 -m deeppavlov.download -test"
    if full:
        cmd += " -all"
    pexpect.run(cmd, timeout=None)


@pytest.mark.parametrize("model,conf_file,model_dir,d", [(m, c, md, d) for m in PARAMS.keys() for c, md, d in PARAMS[m].keys()])
class TestQuickStart(object):

    @staticmethod
    def interact(conf_file, model_dir, qr_list=None):
        qr_list = qr_list or []
        p = pexpect.spawn("python3", ["-m", "deeppavlov.deep", "interact", str(conf_file)], timeout=None)
        for (query, expected_response) in qr_list:  # works until the first failed query
            p.expect(":: ")
            p.sendline(query)
            p.expect(">> ")
            actual_response = p.readline().decode().strip()
            assert expected_response == actual_response, f"Error in interacting with {model_dir} ({conf_file}): {query}"
        p.expect(":: ")
        p.sendline("quit")
        assert p.expect(pexpect.EOF) == 0, f"Error in quitting from deep.py ({conf_file})"

    def test_downloaded_model_existence(self, model, conf_file, model_dir, d):
        if d:
            if not download_path.exists():
                download()
            assert download_path.joinpath(model_dir).exists(), f"{model_dir} was not downloaded"

    def test_interacting_pretrained_model(self, model, conf_file, model_dir, d):
        if d:
            self.interact(tests_dir / conf_file, model_dir, PARAMS[model][(conf_file, model_dir, d)])

    def test_consecutive_training_and_interacting(self, model, conf_file, model_dir, d):
        c = tests_dir / conf_file
        model_path = download_path / model_dir
        shutil.rmtree(str(model_path),  ignore_errors=True)
        _, exitstatus = pexpect.run("python3 -m deeppavlov.deep train " + str(c), timeout=None, withexitstatus=True)
        assert exitstatus == 0, f"Training process of {model_dir} returned non-zero exit code"
        self.interact(c, model_dir)
