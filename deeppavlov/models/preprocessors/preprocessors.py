import re


def simple_prep(data: list) -> list:

    f = [x.lower() for x in data]
    f = [x.replace("\\n", " ") for x in f]
    f = [x.replace("\\t", " ") for x in f]
    f = [x.replace("\\xa0", " ") for x in f]
    f = [x.replace("\\xc2", " ") for x in f]

    f = [re.sub('!!+', ' !! ', x) for x in f]
    f = [re.sub('!', ' ! ', x) for x in f]
    f = [re.sub('! !', '!!', x) for x in f]

    f = [re.sub('\?\?+', ' ?? ', x) for x in f]
    f = [re.sub('\?', ' ? ', x) for x in f]
    f = [re.sub('\? \?', '??', x) for x in f]

    f = [re.sub('\?!+', ' ?! ', x) for x in f]

    f = [re.sub('\.\.+', '..', x) for x in f]
    f = [re.sub('\.', ' . ', x) for x in f]
    f = [re.sub('\.  \.', '..', x) for x in f]

    f = [re.sub(',', ' , ', x) for x in f]
    f = [re.sub(':', ' : ', x) for x in f]
    f = [re.sub(';', ' ; ', x) for x in f]
    f = [re.sub('\%', ' % ', x) for x in f]

    f = [x.replace("won't", "will not") for x in f]
    f = [x.replace("can't", "cannot") for x in f]
    f = [x.replace("i'm", "i am") for x in f]
    f = [x.replace(" im ", " i am ") for x in f]
    f = [x.replace("you're ", "you are") for x in f]
    f = [x.replace("'re", " are") for x in f]
    f = [x.replace("ain't", "is not") for x in f]
    f = [x.replace("'ll", " will") for x in f]
    f = [x.replace("'t", " not") for x in f]
    f = [x.replace("'ve", " have") for x in f]
    f = [x.replace("'s", " is") for x in f]
    f = [x.replace("'re", " are") for x in f]
    f = [x.replace("'d", " would") for x in f]

    f = [re.sub("ies( |$)", "y ", x) for x in f]
    f = [re.sub("s( |$)", " ", x) for x in f]
    f = [re.sub("ing( |$)", " ", x) for x in f]
    f = [x.replace("tard ", " ") for x in f]

    f = [re.sub(" [*$%&#@][*$%&#@]+", " xexp ", x) for x in f]
    f = [re.sub(" [0-9]+ ", " DD ", x) for x in f]
    f = [re.sub("<\S*>", "", x) for x in f]
    f = [re.sub('\s+', ' ', x) for x in f]
    return f


PREPROCESSORS = {"simple_prep": simple_prep}

