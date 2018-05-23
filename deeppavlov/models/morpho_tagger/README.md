It is an implementation of neural morphological tagger from
[Heigold et al., 2017. An extensive empirical evaluation of character-based morphological tagging for 14 languages](http://www.aclweb.org/anthology/E17-1048).
We distribute the model trained on ru_syntagrus corpus of [Universal Dependencies project](www.universaldependencies.org).
If you want to use it from scratch, do the following:

1. Download the archive from 
<http://lnsigo.mipt.ru/export/deeppavlov_data/morpho_tagger.tar.gz> to
 [../../../download](../../../download).
2. To apply a pre-trained ru_syntagrus model to ru_syntagrus test data, run
``python models/morpho_tagger/main.py -p configs/UD2.0/morpho_tagger/ru_syntagrus/morpho_ru_syntagrus_predict.json``.
3. To test ru_syntagrus model on ru_syntagrus test subset, run
``python models/morpho_tagger/main.py -T configs/UD2.0/morpho_tagger/ru_syntagrus/morpho_ru_syntagrus_test.json``.
4. To retrain model on ru_syntagrus dataset, run
``python models/morpho_tagger/main.py configs/UD2.0/morpho_tagger/ru_syntagrus/morpho_ru_syntagrus_train.json``.
Be careful, one epoch takes 8-60 minutes depending on your GPU.
5. To tag Russian sentences from stdin, run
``python deep.py configs/morpho_tagger/UD2.0/morpho_tagger/ru_syntagrus/morpho_ru_syntagrus_interact.json``.

Read [detailed readme](readme_morpho_basic.md) for advanced usage options.
