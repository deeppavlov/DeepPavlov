import pytest
from pathlib import Path
import json
import subprocess as sp
import shutil


tests_dir = Path(__file__, '..').resolve()

# Mapping from model names to configs and corresponding Query-Response pairs
MCQR = {"error_model": {"configs/error_model/brillmoore_wikitypos_en.json": ("error_model", "", ""),
                        "configs/error_model/brillmoore_kartaslov_ru.json": ("error_model", "", "")
                        },
        "go_bot": {"configs/go_bot/gobot_dstc2.json": ("go_bot", "", ""),
                   # "configs/go_bot/config_all.json":
                   #     ("go_bot_all", "", ""),
                   "configs/go_bot/gobot_dstc2_minimal.json": ("go_bot_minimal", "", "")
                   },
        "intents": {"configs/intents/intents_dstc2.json": ("intents", "", "")
                    },
        "ner": {"configs/ner/ner_conll2003.json": ("ner_conll2003_model", "", ""),
                "configs/ner/ner_dstc2.json": ("ner_dstc2_model", "", ""),
                "configs/ner/slotfill_dstc2.json": ("ner", "", "")
                }
        }


def setup_module():
    src_dir = tests_dir.parent / 'deeppavlov'
    test_configs_path = tests_dir / 'configs'

    shutil.rmtree(str(test_configs_path), ignore_errors=True)
    test_configs_path.mkdir()

    for m_name, conf_files in MCQR.items():
        test_configs_path.joinpath(m_name).mkdir()
        for conf_file, qr in conf_files.items():
            with (src_dir / conf_file).open() as fin:
                config = json.load(fin)
            try:
                config["train"]["epochs"] = 1
            except KeyError:
                pass
            with (tests_dir / conf_file).open("w") as fout:
                json.dump(config, fout)


def teardown_module():
    test_configs_path = tests_dir / 'configs'
    shutil.rmtree(str(test_configs_path))


def download(full=None):
    cmd = ["python3", "-m", "deeppavlov.download"]
    if full:
        cmd.append("-all")
    sp.run(cmd)


@pytest.mark.parametrize("model", [k for k, v in MCQR.items()])
class TestQuickStart(object):

    @staticmethod
    def interact(config, query="exit"):
        p = sp.Popen(["python3", "-m", "deeppavlov.deep", "interact", config],
                     stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        out, _ = p.communicate(f"{query}".encode())
        return out

    def test_downloaded_model_exist(self, model):
        if not tests_dir.parent.joinpath('download').exists():
            download()
        assert tests_dir.parent.joinpath('download', model).exists(), f"{model} was not downloaded"

    def test_interact_pretrained_model(self, model):
        for c, fqr in MCQR[model].items():
            c = tests_dir / c
            assert self.interact(c), f"Error in interacting with pretrained {model}: {c}"

    def test_consecutive_training_and_interacting(self, model):
        for c, fqr in MCQR[model].items():
            c = tests_dir / c
            model_path = tests_dir.parent / 'download' / fqr[0]
            shutil.rmtree(str(model_path),  ignore_errors=True)
            p = sp.run(["python3", "-m", "deeppavlov.deep", "train", c])
            assert p.returncode == 0, f"Training process of {model} with {c} returned non-zero exit code"
            assert self.interact(c), f"Error in interacting with 1-epoch trained {model}: {c}"
