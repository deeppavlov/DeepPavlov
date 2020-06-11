import argparse

from deeppavlov.core.common.file import find_config
from deeppavlov.download import deep_download
from deeppavlov.models.morpho_tagger.common import predict_with_model

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="path to file with prediction configuration")
parser.add_argument("-d", "--download", action="store_true", help="download model components")
parser.add_argument("-b", "--batch-size", dest="batch_size", default=16, help="inference batch size", type=int)
parser.add_argument("-f", "--input-file", dest="file_path", default=None, help="path to the input file", type=str)
parser.add_argument("-i", "--input-format", dest="input_format", default="ud",
                    help="input format ('text' for untokenized text, 'ud' or 'vertical'", type=str)
parser.add_argument("-o", "--output-format", dest="output_format", default="basic",
                    help="input format ('basic', 'ud' or 'conllu' (the last two mean the same)", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    config_path = find_config(args.config_path)
    if args.download:
        deep_download(config_path)
    answer = predict_with_model(config_path, infile=args.file_path, input_format=args.input_format,
                                batch_size=args.batch_size, output_format=args.output_format)
    for elem in answer:
        print(elem)
