[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Morphological Tagging

It is an implementation of neural morphological tagger from
[Heigold et al., 2017. An extensive empirical evaluation of character-based morphological tagging for 14 languages](http://www.aclweb.org/anthology/E17-1048).
We distribute the model trained on ru_syntagrus corpus of [Universal Dependencies project](www.universaldependencies.org).
If you want to use it from scratch, do the following:

1. Download data
```python -m deeppavlov.deep download deeppavlov/configs/morpho_tagger/UD2.0/ru_syntagrus/morpho_ru_syntagrus_train.json```
To perform all downloads in runtime you can also run all subsequent commands with `-d` key, 
2. To apply a pre-trained ru_syntagrus model to ru_syntagrus test data, run
```python models/morpho_tagger/main.py train configs/morpho_tagger/UD2.0/ru_syntagrus/morpho_ru_syntagrus_predict.json```.
from `deeppavlov` directory of the repository. A subdirectory``results`` will be created and predictions will be written to the file
```ud_ru_syntagrus_test.res``` in it.
3. To evaluate ru_syntagrus model on ru_syntagrus test subset, run
```python -m deeppavlov.deep evaluate configs/UD2.0/morpho_tagger/ru_syntagrus/morpho_ru_syntagrus_train.json```.
4. To retrain model on ru_syntagrus dataset, run
```python -m deeppavlov.deep train configs/UD2.0/morpho_tagger/ru_syntagrus/morpho_ru_syntagrus_train.json```.
Be careful, one epoch takes 8-60 minutes depending on your GPU.
5. To tag Russian sentences from stdin, run
```python -m deeppavlov.deep configs/morpho_tagger/UD2.0/morpho_tagger/ru_syntagrus/morpho_ru_syntagrus_predict.json```.

Read [detailed readme](readme_morpho_basic.md) for advanced usage options.
