import pytest
import subprocess as sp
import os
import shutil


N2C = {"go_bot": ("../deeppavlov/configs/go_bot/config.json",
                  "../deeppavlov/configs/go_bot/config_train.json"),
       "intents": ("../deeppavlov/configs/intents/config_dstc2_infer.json",
                   "../deeppavlov/configs/intents/config_dstc2_train.json"),
       "ner": ("../deeppavlov/configs/ner/ner_dstc2_train.json",
               "../deeppavlov/configs/ner/ner_dstc2_train.json"),
       "error_model": ("../deeppavlov/configs/error_model/config_en.json",
                       "../deeppavlov/configs/error_model/config_en.json")}


def download(full=None):
    cmd = ["python", "-m", "deeppavlov.download"]
    if full:
        cmd.append("-all")
    sp.run(cmd)


@pytest.mark.parametrize("model", [k for k, v in N2C.items()])
class TestQuickStart(object):

    @staticmethod
    def interact(query, mdl):
        p = sp.Popen(["python", "-m", "deeppavlov.deep", "interact", N2C[mdl][0]],
                     stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        out, _ = p.communicate(f"{query}".encode())
        return out

    def test_downloaded_model_exist(self, model):
        if not os.path.exists("../download/"):
            download()
        assert os.path.exists("../download/" + model)

    def test_interact_pretrained_model(self, model):
        assert self.interact("In the center of the city, near the hotel", model)

    def test_model_training(self, model):
        shutil.rmtree("../download/" + model)
        p = sp.run(["python", "-m", "deeppavlov.deep", "train", N2C[model][1]])
        assert p.returncode == 0

    def test_interact_trained_model(self, model):
        assert self.interact("To the south of the center", model)
