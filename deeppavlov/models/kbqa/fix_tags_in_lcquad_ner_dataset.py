import argparse
import pickle


def get_args():
    parser = argparse.ArgumentParser(
        description="Transforms -TAG markup to BIO"
    )
    parser.add_argument(
        "input", 
        help="Path to pickle file with original dataset"
    )
    parser.add_argument(
        "output",
        help="Path to pickle file where fixed dataset "
             "will be saved with fixed markup."
    )
    parser.add_argument(
        "mapping",
        help="A mapping showing how tags are transformed. "
             "The mapping has to have format "
             "{\"E\": \"ENTITY\", \"T\": \"TYPE\"}. "
             "For such mapping transformation will be "
             "['O-TAG', 'E-TAG', 'E-TAG', 'O-TAG', 'T-TAG'] -> "
             "['O', 'B-ENTITY', 'I-TYPE', 'O', 'O']."
    )        
    args = parser.parse_args()
    args.mapping = eval(args.mapping)
    return args


def fix_tags(tags, mapping):
    new_tags = []
    prev_tag = 'O-TAG'
    for tag in tags:
        if tag == 'O-TAG':
            new_tags.append('O')
        else:
            prefix = tag.split('-')[0]
            if prev_tag != tag:
                new_tags.append('B-' + mapping[prefix])
            else:
                new_tags.append('I-' + mapping[prefix])
        prev_tag = tag
    return new_tags


def main():
    args = get_args()
    with open(args.input, 'rb') as in_f:
        content = pickle.load(in_f)
    result = {}
    for data_type, data in content.items():
        result[data_type] = []
        for phrase, tags in data:
            new_tags = fix_tags(tags, args.mapping)
            result[data_type].append((phrase, new_tags))
    with open(args.output, 'wb') as out_f:
        pickle.dump(result, out_f)


if __name__ == '__main__':
    main()
