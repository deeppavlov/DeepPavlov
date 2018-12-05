Pipeline Manager
================



The :class:`~pipeline_manager.pipeline_manager.PipelineManager` implements the functions of automatic experiment
management. The class accepts a config in the input in which the structure of the experiments is described, and
additional parameters, which are class attributes. Based on this information, a list of deeppavlov configs is
created. Experiments can be run sequentially or in parallel, both on video cards and on the processor.
A special class is responsible for describing and logging experiments, their execution time and results.
After passing all the experiments based on the logs, a small report is created in the form of a xlsx table,
and histogram with metrics info. When you start the experiment, you can also search for optimal hyperparameters,
"grid" and "random" search is available.

Running a large number of experiments, especially with large neural models, may take a large amount of time, so a
special test was added to check the correctness of the joints of individual blocks in all pipelines, or another
errors. During the test, all pipelines are trained on a small piece of the original dataset, if the test passed
without errors, you can not worry about the experiment, and then a normal experiments is automatically started.
The test starts automatically, nothing else needs to be done, but it can also be turned off. In this case, the
experiment will start immediately. Test supports multiprocessing.

Also you can save checkpoints for all pipelines, or only the best.