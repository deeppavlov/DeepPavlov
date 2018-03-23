import pytest
from pathlib import Path
import pexpect
import json
import shutil


tests_dir = Path(__file__, '..').resolve()
test_configs_path = tests_dir / "configs"
download_path = tests_dir / "download"


# Mapping from model name to config-model_dir-ispretrained and corresponding queries-response list.
PARAMS = {"error_model": {("configs/error_model/brillmoore_wikitypos_en.json", "error_model", True):
                              [
                                  ("helllo", "hello"),
                                  ("datha", "data")
                              ],
                          ("configs/error_model/brillmoore_kartaslov_ru.json", "error_model", True): []},
          "go_bot": {("configs/go_bot/gobot_dstc2.json", "go_bot", True): []},
          "intents": {("configs/intents/intents_dstc2.json", "intents", True):  []},
          "snips": {("configs/intents/intents_snips.json", "intents", False): []},
          "sample": {("configs/intents/intents_sample_csv.json", "intents", False): [],
                    ("configs/intents/intents_sample_json.json", "intents", False): []},
          "ner": {("configs/ner/ner_conll2003.json", "ner_conll2003", True): [],
                  ("configs/ner/ner_dstc2.json", "ner", True): [],
                  ("configs/ner/ner_ontonotes_emb.json", "ner_ontonotes", True): [],
                  ("configs/ner/slotfill_dstc2.json", "ner", True):
                      [
                          ("chinese food", "{'food': 'chinese'}"),
                          ("in the west part", "{'area': 'west'}"),
                          ("moderate price range", "{'pricerange': 'moderate'}")
                      ]
                  },
          "ranking": {("configs/ranking/insurance_config.json", "ranking", True): []},
          "squad": {("configs/squad/squad.json", "squad_model", True): []}
          }

MARKS = {"gpu_only": ["squad"], "slow": ["error_model", "go_bot", "squad"]}  # marks defined in pytest.ini

TEST_GRID = []
for model in PARAMS.keys():
    for conf_file, model_dir, ispretrained in PARAMS[model].keys():
        marks = []
        for mark in MARKS.keys():
            if model in MARKS[mark]:
                marks.append(eval("pytest.mark." + mark))
        grid_unit = pytest.param(model, conf_file, model_dir, ispretrained, marks=marks)
        TEST_GRID.append(grid_unit)


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
            if config.get("train"):
                config["train"]["epochs"] = 1
                for pytest_key in [k for k in config["train"] if k.startswith('pytest_')]:
                    config["train"][pytest_key[len('pytest_'):]] = config["train"].pop(pytest_key)
            config["deeppavlov_root"] = str(download_path)
            with (tests_dir / conf_file).open("w") as fout:
                json.dump(config, fout)


def teardown_module():
    shutil.rmtree(str(test_configs_path), ignore_errors=True)
    shutil.rmtree(str(download_path), ignore_errors=True)


def download(full=None):
    cmd = "python3 -m deeppavlov.download -test"
    if full:
        cmd += " -all"
    pexpect.run(cmd, timeout=None)


@pytest.mark.parametrize("model,conf_file,model_dir,ispretrained", TEST_GRID)
class TestQuickStart(object):

    @staticmethod
    def interact(conf_file, model_dir, qr_list=None):
        qr_list = qr_list or []
        p = pexpect.spawn("python3", ["-m", "deeppavlov.deep", "interact", str(conf_file)], timeout=None)
        for *query, expected_response in qr_list:  # works until the first failed query
            for q in query:
                p.expect("::")
                p.sendline(q)
            p.expect(">> ")
            actual_response = p.readline().decode().strip()
            assert expected_response == actual_response, f"Error in interacting with {model_dir} ({conf_file}): {query}"
        p.expect("::")
        p.sendline("quit")
        assert p.expect(pexpect.EOF) == 0, f"Error in quitting from deep.py ({conf_file})"

    @pytest.mark.skipif("not ispretrained")
    def test_downloaded_model_existence(self, model, conf_file, model_dir, ispretrained):
        if not download_path.exists():
            download()
        assert download_path.joinpath(model_dir).exists(), f"{model_dir} was not downloaded"

    @pytest.mark.skipif("not ispretrained")
    def test_interacting_pretrained_model(self, model, conf_file, model_dir, ispretrained):
        self.interact(tests_dir / conf_file, model_dir, PARAMS[model][(conf_file, model_dir, ispretrained)])

    def test_consecutive_training_and_interacting(self, model, conf_file, model_dir, ispretrained):
        c = tests_dir / conf_file
        model_path = download_path / model_dir
        shutil.rmtree(str(model_path),  ignore_errors=True)
        _, exitstatus = pexpect.run("python3 -m deeppavlov.deep train " + str(c), timeout=None, withexitstatus=True)
        assert exitstatus == 0, f"Training process of {model_dir} returned non-zero exit code"
        self.interact(c, model_dir)
