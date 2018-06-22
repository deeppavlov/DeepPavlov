[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Parameters evolution for DeepPavlov models

This repository contains implementation of parameters evolution for DeepPavlov models.



If one prefers to run evolution on some provided by DeepPavlov dataset,
firstly, download embeddings and datasets running the following command providing
corresponding name of the config file (see above):

```
cd deeppavlov
python deep.py download configs/intents/intents_snips.json
```

To evolve model of interest run the following command providing corresponding name of the config file (see above):
```
cd deeppavlov
python evolve.py interact configs/evolution/evolve_intents_snips.json
```

