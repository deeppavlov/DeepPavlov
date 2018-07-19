[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Pipeline Manager
### Hi, everybody
This is a small manual on the use of the functionality to automatically iterate on our pipeline in DeepPavlov.
###In what cases can this be useful ?
Specific example. We consider the problem of classification, for example intents. You have 10 models that can give some result, a few tokenizers, a typo, a lemmatizer, ELMo, fasttext, and a lot more. And on the good side, you would try all the models, preferably with different combinations of vectorization and preprocessing. After that, to form a report on the experiments. To see which model works better, which is cooler than ELMo or fasttext, etc.