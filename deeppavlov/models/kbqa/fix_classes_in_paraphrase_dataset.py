import argparse
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Path to XML file with paraphrases dataset"
    )
    parser.add_argument(
        "output",
        help="Path to XML file where changed dataset will be saved"
    )
    args = parser.parse_args()
    return args


def reduce_class_by_1(match_obj):
    return match_obj.group(1) + str(int(match_obj.group(2)) - 1) + match_obj.group(3)


def main():
    args = get_args()
    with open(args.input) as in_f, open(args.output, 'w') as out_f:
        in_text = in_f.read()
        out_text = re.sub(r'(<value name="class">)([\-0-9]+)(</value>)', reduce_class_by_1, in_text)
        out_f.write(out_text)


if __name__ == '__main__':
    main()
