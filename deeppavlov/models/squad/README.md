# Question Answering Model for SQuAD dataset

## Task definition
Question Answering on SQuAD dataset is a task to find an answer on question in a given context (e.g, paragraph from Wikipedia), where the answer to each
question is a segment of the context:

Context:
```markdown
In meteorology, precipitation is any product of the condensation of atmospheric 
water vapor that falls under gravity. The main forms of precipitation include drizzle, 
rain, sleet, snow, graupel and hail... Precipitation forms as smaller droplets coalesce 
via collision with other rain drops or ice crystals **within a cloud**. Short, intense
periods of rain in scattered locations are called “showers”.
```

Question:
```text
Where do water droplets collide with ice crystals to form precipitation?
```

Answer:
```text
within a cloud
```
Datasets, which follow this task format:
* Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) (EN)
* [SDSJ Task B](https://www.sdsj.ru/ru/contest.html) (RU)

## Model
Question Answering Model is based on R-Net, proposed by Microsoft Research Asia ("R-NET: Machine Reading Comprehension with Self-matching Networks" [[link]](https://www.microsoft.com/en-us/research/publication/mrc/)) and its realization by Wenxuan Zhou [[link]](https://github.com/HKUST-KnowComp/R-Net).
  
## Configuration

## Running model
**Tensorflow-1.4.0 with GPU support is required** to run this model.
## Training
**Warning**: training with default config requires about 10Gb on  GPU. Run following command to train the model:  
```bash
python -m deeppavlov.deep train deeppavlov/configs/squad/squad.json
```
## Interact mode
Interact mode provides command line interface to trained model:

To run model in interact mode run the following command:
 ```bash
python -m deeppavlov.deep interact deeppavlov/configs/squad/squad.json
```
Model will ask you to type in context and question.


## Training on SDSJ Task B
If you want to train this model on SDSJ Task B then you should follow these steps:
* Convert data to SQuAD format
* Use Russian [word](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec) and [character](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize-char.vec) embeddings
