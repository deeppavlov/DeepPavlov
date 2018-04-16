# Question Answering Model for SQuAD dataset

## Task definition
Question Answering on SQuAD dataset is a task to find an answer on question in a given context (e.g, paragraph from Wikipedia), where the answer to each
question is a segment of the context:

Context:
> In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of precipitation include drizzle, rain, sleet, snow, graupel and hail... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals **within a cloud**. Short, intense periods of rain in scattered locations are called “showers”.


Question:
> Where do water droplets collide with ice crystals to form precipitation?


Answer:
> within a cloud

Datasets, which follow this task format:
* Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) (EN)
* [SDSJ Task B](https://www.sdsj.ru/ru/contest.html) (RU)

## Model
Question Answering Model is based on R-Net, proposed by Microsoft Research Asia (["R-NET: Machine Reading Comprehension with Self-matching Networks"](https://www.microsoft.com/en-us/research/publication/mrc/)) and its [implementation](https://github.com/HKUST-KnowComp/R-Net) by Wenxuan Zhou.
  
## Configuration
Default config could be found at `deeppavlov/configs/squad/squad.json`
### Config components
* **squad_dataset_reader** - downloads and reads SQuAD dataset
  * data_path - path to save dataset
* **squad_iterator** - create batches from SQuAD dataset
* **squad_preprocessor** - preprocesses context, question by cleaning data and tokenizing
  * in: **context_raw**, **question_raw** - not processed contexts and questions
  * out: 
    * **context** - processed context (cleaned unicode symbols and quoting)
    * **context_tokens** - tokenized context
    * **context_chars** - tokenized context split on chars
    * **c_r2p** - mapping from raw context to processed context
    * **c_p2r** - mapping from processed context to raw context
    * **question** - processed question
    * **question_tokens** - tokenized question
    * **question_chars** - tokenized question split on chars
    * **spans** - mapping from word indices to position in text
  * context_limit - maximum length of context in words
  * question_limit - maximum length of question in words
  * char_limit - maximum number of chars in token
* **squad_ans_preprocessor** - preprocesses answer
  * in:
    * **ans_raw** - not processed answer
    * **ans_raw_start** - start position of not processed answer in context
    * **c_r2p**
    * **spans**
  * out: 
    * **ans** - processed answer
    * **ans_start** - start position of processed answer
    * **ans_end** - end position of processed answer
* **squad_vocab_embedder** - builds vocabulary and embedding matrix
  * in:
    * **context_tokens**
    * **question_tokens**
  * out
    * **context_tokens_idxs**
    * **question_tokens_idxs**
  * fit_on: **context_tokens** and **question_tokens**
  * level - token or char
  * emb_folder - path to store pretrained embeddings
  * emb_url - url to donwload embeddings
  * save_path - path to save vocabulary and embedding matrix
  * load_path - path to load vocabulary and embedding matrix
  * context_limit - maximum length of context in words
  * question_limit - maximum length of question in words
  * char_limit - maximum number of chars in token
* squad_model - model to find answer on question in context  
  * in: **context_tokens_idxs**, **context_chars_idxs**, **question_tokens_idxs**, **question_chars_idxs**
  * in_y: **ans_start**, **ans_end**
  * out:
    * **ans_start_predicted** - start position of predicted answer
    * **ans_end_predicted** - end position of predicted answer
  * word_emb - pretrained word embeddings
  * char_emb - pretrained char embeddings
  * context_limit - maximum length of context in words
  * question_limit - maximum length of question in words
  * char_limit - maximum number of chars in token
  * train_char_emb - update char_emb during training or not
  * char_hidden_size - size of word embedding built on characters
  * encoder_hidden_size - hidden size of encoder cells 
  * attention_hidden_size - hidden size to use to compute attention
  * learning_rate
  * min_learning_rate - minimal lr for lr decay
  * learning_rate_patience - patience for lr decay
  * keep_prob - dropout keep probability
  * grad_clip - gradient clipping value
  * weight_decay - weight_decay rate for exponential moving average
  * save_path
  * load_path
* squad_ans_postprocessor - extracts predicted answer from context
 * in: **ans_start_predicted**, **ans_end_predicted**, **context_raw**, **c_p2r**, **spans**
 * out: 
   * **ans_predicted** - text of predicted answer in raw context
   * **ans_start_predicted** - start position of predicted answer in raw context
   * **ans_end_predicted** - end position of predicted answer in raw context

## Running model
**Tensorflow-1.4.0 with GPU support is required** to run this model.
## Training
**Warning**: training with default config requires about 10Gb on  GPU. Run following command to train the model:  
```bash
python -m deeppavlov.deep train deeppavlov/configs/squad/squad.json
```
## Interact mode
Interact mode provides command line interface to already trained model.

To run model in interact mode run the following command:
 ```bash
python -m deeppavlov.deep interact deeppavlov/configs/squad/squad.json
```
Model will ask you to type in context and question.

## Pretrained models:
### SQuAD
Pretrained model is available and can be downloaded: [http://lnsigo.mipt.ru/export/deeppavlov_data/squad_model_1.1.tar.gz](http://lnsigo.mipt.ru/export/deeppavlov_data/squad_model_1.1.tar.gz)

It achieves ~80 F-1 score and ~71 EM on dev set. Results of the most recent solutions could be found on [SQuAD Leadearboad](https://rajpurkar.github.io/SQuAD-explorer/).
### SDSJ Task B
Pretrained model is available and can be downloaded: [http://lnsigo.mipt.ru/export/deeppavlov_data/squad_model_ru.tar.gz](http://lnsigo.mipt.ru/export/deeppavlov_data/squad_model_ru.tar.gz)

It achieves ~80 F-1 score and ~60 EM on dev set.