import sys
import getopt

from deeppavlov.core.commands.train import train_model_from_config

from deeppavlov.models.morpho_tagger.common import predict_with_model

SHORT_OPTS = "Tp"

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    mode = "train"
    for opt, val in opts:
        if opt in ["-T"]:
            mode = "test"
        elif opt in ["-p"]:
            mode = "predict"
        else:
            raise ValueError("Unknown option {}".format(opt))
    config_path = args[0]
    if mode == "predict":
        predict_with_model(config_path)
    elif mode == "test":
        train_model_from_config(config_path, is_trained=True)
    elif mode == "train":
        train_model_from_config(config_path, is_trained=False)