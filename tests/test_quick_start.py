# this test is designed to run from 'tests' folder
import pytest
from pathlib import Path
import json
import subprocess as sp
import shutil


N2C = {"error_model": ("configs/error_model/config_en.json",
                       "configs/error_model/config_ru.json"),
       "go_bot": ("configs/go_bot/config.json",
                  "configs/go_bot/config_all.json",
                  "configs/go_bot/config_minimal.json"),
       "intents": ("configs/intents/config_dstc2_infer.json",
                   "configs/intents/config_dstc2_train.json"),
       "ner": ("configs/ner/ner_conll2003_train.json",
               "configs/ner/ner_dstc2_train.json",
               "configs/ner/slot_config_train.json")}

TEST_QUERY = "In the center of the city, near the hotel"


def setup_module():
    src_dir = (Path() / "../deeppavlov").resolve()
    tests_dir = Path().resolve()
    (tests_dir / "configs").mkdir()
    for m_name, conf_files in N2C.items():
        (tests_dir / "configs" / m_name).mkdir()
        for conf_file in conf_files:
            with (src_dir / conf_file).open() as fin:
                config = json.load(fin)
            try:
                config["train"]["epochs"] = 1
            except KeyError:
                pass
            with (tests_dir / conf_file).open("w") as fout:
                json.dump(config, fout)


def teardown_module():
    shutil.rmtree("configs")


def download(full=None):
    cmd = ["python", "-m", "deeppavlov.download"]
    if full:
        cmd.append("-all")
    sp.run(cmd)


@pytest.mark.parametrize("model", [k for k, v in N2C.items()])
class TestQuickStart(object):

    @staticmethod
    def interact(config, query):
        p = sp.Popen(["python", "-m", "deeppavlov.deep", "interact", config],
                     stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        out, _ = p.communicate(f"{query}".encode())
        return out

    def test_downloaded_model_exist(self, model):
        if not (Path() / "../download").resolve().exists():
            download()
        assert (Path() / "../download/" / model).exists(), f"{model} was not downloaded"

    def test_interact_pretrained_model(self, model):
        for c in N2C[model]:
            assert self.interact(c, TEST_QUERY), f"Error in interacting with pretrained {model}: {c}"

    def test_consecutive_training_and_interacting(self, model):
        for c in N2C[model]:
            shutil.rmtree("../download/" + model)
            p = sp.run(["python", "-m", "deeppavlov.deep", "train", c])
            assert p.returncode == 0, f"Training process of {model} with {c} returned non-zero exit code"
            assert self.interact(c, TEST_QUERY), f"Error in interacting with 1-epoch trained {model}: {c}"
