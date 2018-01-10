# Automatic spelling correction component

Automatic spelling correction component is based on [An Improved Error Model for Noisy Channel Spelling Correction](http://www.aclweb.org/anthology/P00-1037)
by Eric Brill and Rober C. Moore and uses statistics based error model and an ARPA language model to correct spelling errors.

## Usage

!how to config  
!input and output format

```python
import json
import sys

from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.utils import set_usr_dir

if len(sys.argv < 2):
    print()

CONFIG_PATH = 'deeppavlov/models/spellers/error_model/config_ru_custom_vocab.json'
usr_dir = set_usr_dir(CONFIG_PATH)

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

model = build_model_from_config(config)
with open(sys.argv(1), 'w') as out_file:
    for line in sys.stdin:
        out_file.write(model.infer(line) + '\n')
        out_file.flush()
```

## Training

#### Error model

!how to train an error model:  
!write your own dataset_reader, run train

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.utils import set_usr_dir

MODEL_CONFIG_PATH = 'deeppavlov/models/spellers/error_model/config_ru_custom_vocab.json'
usr_dir = set_usr_dir(MODEL_CONFIG_PATH)
train_model_from_config(MODEL_CONFIG_PATH)
```

#### Language model

!how to train and connect a language model, add a link to kenlm

[A russian language model](http://lnsigo.mipt.ru/export/lang_models/ru_wiyalen_no_punkt.arpa.binary.gz)

## Ways to improve

!rewrite some things on Cython  
!adding and removing spaces  
!implement [Discriminative Reranking for Spelling Correction](http://www.aclweb.org/anthology/Y06-1009)
by Yang Zhang, Pilian He, Wei Xiang and Mu Li