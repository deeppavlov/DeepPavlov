[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![tensorflow 1.4](https://img.shields.io/badge/tensorflow-1.4-green.svg)


# Sequence-To-Sequence Dialogue Bot for goal-oriented task 
Dialogue agent is able to handle multi-domain data. 
The model is end-to-end differentiable and does not need to explicitly model dialogue state or belief trackers.

We also used a new Stanford NLP Group released dataset of 3,031 dialogues [[2]](#references) that are grounded through 
underlying knowledge bases and span three distinct tasks in the in-car personal assistant space: 
calendar scheduling, weather information retrieval, and point-of-interest navigation.

Here is a simple example of interaction with a trained dialogue bot (can be downloaded with [`deeppavlov/download.py`](../../download.py)):

```
x_text:: Where's the nearest parking garage?
>> the nearest is dish parking at dish_parking_address. is that okay?
x_text:: Yes, please set directions via a route that avoids all heavy traffic if possible.
>> it looks like there is a road block being reported on the route but i will still find the quickest route to dish_parking_address.
x_text:: Thanks so much for your help.
>> you're welcome. have a good day. end_of_dialogue

x_text:: Will it be warm in Camarillo over the next 2 days?
>> no, it's not gon na be warm in camarillo over the next 2 days
x_text:: Thank you car!
>> you're welcome! end_of_dialogue

x_text:: What day is my lab appointment, and who all will be there?
>> your lab appointment is on lab_appointment_date at lab_appointment_time with lab_appointment_party.
x_text:: Thanks!
>> you're welcome.
```

## Usage

#### Requirements

To use a seq2seq_go_bot model you should have a pretrained goal-oriented bot model

* config [`deeppavlov/configs/seq2seq_go_bot/bot_kvret_infer.json`](../../configs/seq2seq_go_bot/bot_kvret_infer.json)
is recommended to be used in inference mode
   
* config [`deeppavlov/configs/seq2seq_go_bot/bot_kvret.json`](../../configs/seq2seq_go_bot/bot_kvret.json)
is recommended to be used in train mode
   
#### Config parameters:

* `name` always equals to `"seq2seq_go_bot"`
* `source_vocab` — vocabulary of tokens from context (source) utterances
* `target_vocab` — vocabulary of tokens from response (target) utterances
* `start_of_sequence_token` — token corresponding to the start of sequence during decoding
* `end_of_sequence_token` — token corresponding to the end of sequence during decoding
* `bow_encoder` — one of bag-of-words encoders from [`deeppavlov.models.encoders.bow`](../../models/encoders/bow.py) module
   * `name` — encoder name
   * other arguments specific to your encoder
* `debug` — whether to display debug output (defaults to `false`) _(optional)_
* `network` — reccurent network that handles encoder-decoder mechanism
   * `name` equals to `"seq2seq_go_bot_nn"`
   * `learning_rate` — learning rate during training
   * `target_start_of_sequence_index` — index of `start_of_sequence_token` in decoder vocabulary
   * `target_end_of_sequence_index` — index of `end_of_sequence_token` in decoder vocabulary
   * `source_vocab_size` — size of encoder token vocabulary (size of `source_vocab` is recommended)
   * `target_vocab_size` — size of decoder token vocabulary (size of `target_vocab` is recommended)
   * `hidden_size` — LSTM hidden state size equal for encoder and decoder

#### Usage example
* To infer from a pretrained model with config path equal to `path/to/config.json`:

```python
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.file import read_json

CONFIG_PATH = 'path/to/config.json'
model = build_model_from_config(read_json(CONFIG_PATH))

utterance = ""
while utterance != 'exit':
    print(">> " + model([utterance])[0])
    utterance = input(':: ')
```

* To interact via command line use [`deeppavlov/deep.py`](../../deep.py) script:

```bash
cd deeppavlov
python3 deep.py interact path/to/config.json
```

# References
[1] [A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)

[2] [Mihail Eric, Lakshmi Krishnan, Francois Charette, and Christopher D. Manning, Key-Value Retrieval Networks for Task-Oriented Dialogue – 2017](https://arxiv.org/abs/1705.05414.pdf)
