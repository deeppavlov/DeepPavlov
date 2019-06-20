from ner_label import *
import pandas as pd
import pysnooper

# @pysnooper.snoop()
def devide(length, scale):
    sum = scale[0]+scale[1]+scale[2]
    per = [x/sum for x in scale]
    range1 = int(per[0] * length)
    range2 = int(per[1] * length) + range1
    return range1,range2

def data_devide(origin_df,scale):
    df = origin_df['商品描述']
    length = len(df)
    range1, range2 = devide(length, scale)
    print(range1,range2)
    train = df[0:range1]
    test = df[range1+1:range2]
    validation = df[range2+1:length]
    return train, test, validation

def process(df, name):
    list = df.tolist()
    name = './' + name + '.txt'
    with open(name, 'a') as f:
        for line in list:
            out = tag(line)
            for element in out:
                print(element[0],element[1],element[2], file =f)
            print('', file = f)


path = './test_data.csv'
df = pd.read_csv(path)
scale = (8, 1, 1)

train, test, validation = data_devide(df,scale)

print(train.head())
print(test.head())
print(validation.head())
print(len(train), len(test), len(validation))

process(train, 'train')
process(test, 'test')
process(validation, 'validation')
