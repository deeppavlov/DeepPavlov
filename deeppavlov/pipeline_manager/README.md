[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Pipeline Manager
### Hi, everybody
This is a small manual on the use of the functionality to automatically iterate on our pipeline in DeepPavlov.
###In what cases can this be useful ?
Specific example. We consider the problem of classification, for example intents. You have 10 models that can give some 
result, a few tokenizers, a typo, a lemmatizer, ELMo, fasttext, and a lot more. And on the good side, you would try all
the models, preferably with different combinations of vectorization and preprocessing. After that, to form a report on 
the experiments. To see which model works better, which is cooler than ELMo or fasttext, etc.

The following functionality allows you to automate, to some extent, the above tasks, and make your life easier
~~(but it is not accurate)~~.

### Description of main class
"pipeline_manager" is a special class that takes the path to the config file, and a number of additional parameters that
specify its mode of operation. In the config file is set to a "pattern search" pipeline, on the basis of which a special
generator inside the class, combining different components specified in the pattern search, gives a set of configs in 
the standard for DeepPavlov-a form. Further, each such config is run separately, trained and tested using the library 
functionality. As training and testing on our pipeline of configi on our pipeline and intermediate results are logged 
in a separate file. When the algorithm is finished, a new folder called experiments will appear in the Downloads folder,
which will contain logs, model weights, as well as an excel file displaying all the executed paylines with the 
description of the components, and the achieved values on the metrics.

```
# The class itself is located on the following path
from deeppavlov.pipeline_manager.pipeline_manager import PipelineManager

# And the class instance is created as follows
manager = PipelineManager(config_path, exp_name, mode, root, hyper_search, sample_num, target_metric)

# The experiment starts as follows
manager.run()
```
**config_path** - str: The path to the config file, which is set to the pattern of iterating through the lines, 
training parameters, etc. this is a required parameter with no default value

**exp_name** - str: The name of the experiment, required for the formation of logs and report, is a required parameter 
without a value by default

**mode** - str: [train, evaluate] the mode of operation of the pipeline manager, you may how to train data, and test 
the already trained models train used by default

**root** - str: The path to the folder in which the report will be generated, and create the folder experiments, 
value './experiments/ ' is the default

**hyper_search** - str: [grid, random] a trigger that specifies what type of search grid is used by default

**sample_num** - int: If hyper_search=random, sample_num indicates the number of generated examples. The default value 
is 10.

**target_metric** - str: The name of the metric on the basis of which the results will be sorted when the report is 
generated. The default value is None, in this case the target metric is taken from the first of those names that are 
specified in the config file. If the specified metric is not present in DeepPavlov, an exception is thrown.

### Creating a config file
The config, the path to which the pipeline manager class enters, differs from the" standard "config, which you need to 
write to run a certain model in DeepPavlov, only by the way it describes the content of the" pipe " field in the 
changer. Here is a short version of how the "pipe" field is set in the changer":
```
{"chainer": {"in": ["x"], "in_y": ["y"],
             "pipe": [
                       {"id": "classes_vocab",
                        ...},
                       {"id": "my_embedder",
                        ...},
                       {"in": ["x"],
                        "name": "str_lower",
                        "out": ["x"]},
                       {"id": "my_tokenizer",
                         ...},
                       {"in": ["x"], "in_y": ["y"], "out": ["y_labels", "y_probas_dict"],
                        "main": True,
                        "name": "intent_model",
                        ...},
                     ],
             "out": ["y_labels", "y_probas_dict"]}}
```
As you can see in the example, we set the pipe field as a list of dictionaries that explicitly describe the sequence of 
components and their parameters. While we need to set up a pattern to automatically start many different paylines, we 
need to set up a pattern to iterate through different components. To do this, each element of the list in the pipe field
is transformed into a separate list of dictionaries, in which there is an enumeration of the components with the 
parameters that are required for the search. Even if there is only one component at a certain position in the pipeline,
it should be wrapped in a list. Here is an example:
```
{"chainer": {"in": ["x"], "in_y": ["y"],
    "pipe": [
      [{"id": "classes_vocab",
        ...}],
      [{"id": "my_embedder",
        ...}],
      [{"name": "str_lower", ...}, None],
      [{"name": "nltk_tokenizer",
        ...},
       {"name": "lazy_tokenizer",
        ...}],
      [{"name": "intent_model",
        "model_name": "cnn_model",
        ...},
       {"name": "intent_model",
        "model_name": "bilstm_model",
        ...}]
       ],
    "out": ["y_labels", "y_probas_dict"]}
}
```
As you can see now we have two tokens in the config, and two models CN and bi-LSTM.Also note that the third item in 
addition to the dictionary also contains None:
```
"pipe": [...
         [{"name": "str_lower", ...}, None],
         ...]
```
This means that this element may not be present in the pipeline at all. As a result, you will create a pipeline that 
does not contain this component. In this case, it is easy to calculate that 8 different paylines will be launched in 
the experiment.
That's basically it. In order to set the search pattern you need to take your config that you have already used before 
and add additional components and models to the "pipe" chainer-a that you want to try. The main thing is not to forget 
that each element in the new "pipe" is a separate list. Then you can run the experiment from the command line by typing:
```
python -m deeppavlov sort_out  <path to the config file>  -e  <experiment name>
```
When finished, the downloads folder will contain the experiments folder, where all your data, reports, checkpoints for 
individual experiments will be saved, sorted by date and experiment names.

There may be such a situation that within one config you will not be able to describe all the desired options of the 
lines, because of the incompatibility of individual components. In this case, you can create two configs, and run them 
one by one, specifying the same name of the experiment, in this case, all the logs and reports automatically connect.

### Hyperparameter search

Also, the pipeline manager may carry out selection of hyperparameters as the base model, and individual component if 
required. Currently supports "random" or "grid" search. When searching for an optimal hyperparameters, the number of 
different on our pipeline, and therefore the execution time of algorithm may increase significantly, it must be borne 
in mind.

### Grid search

### Random search
