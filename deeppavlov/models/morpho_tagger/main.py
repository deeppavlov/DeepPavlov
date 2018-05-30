import sys

from deeppavlov.models.morpho_tagger.common import predict_with_model

if __name__ == "__main__":
    config_path = sys.argv[1]
    predict_with_model(config_path)