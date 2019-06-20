from ner_label import *
import pandas as pd
import pysnooper
from tqdm import tqdm
from sklearn.utils import shuffle
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--d',
                    dest='data_path',
                    type=str,
                    required=True)
parser.add_argument('--o',
                    dest='output_path',
                    type=str,
                    required=True)
args = parser.parse_args()

# check if output is there
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

# @pysnooper.snoop()
def devide(length, scale):
    sum = scale[0]+scale[1]+scale[2]
    per = [x/sum for x in scale]
    range1 = int(per[0] * length)
    range2 = int(per[1] * length) + range1
    return range1, range2


def data_devide(df,scale):
    df = shuffle(df)
    length = len(df)
    range1, range2 = devide(length, scale)
    train = df[0:range1]
    test = df[range1:range2]
    validation = df[range2:length]
    return train, test, validation


def process(df, name):
    name = args.output_path + '/' + name + '.txt'
    print("Processing set : %s" % name)
    with open(name, 'w') as f:
        for index in tqdm(range(len(df))):
            line = df.iloc[index]
            out = tag(line)
            for element in out:
                print(element[0], element[1], file=f)
            print('', file=f)


df = pd.read_csv(args.data_path)
scale = (8, 1, 1)

train, test, validation = data_devide(df,scale)

# print(train.head())
# print(test.head())
# print(validation.head())
# print(len(train), len(test), len(validation))

process(train, 'train')
process(test, 'test')
process(validation, 'validation')
