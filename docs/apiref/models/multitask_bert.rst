deeppavlov.models.multitask_bert
================================

.. autoclass:: deeppavlov.dataset_readers.multitask_reader.MultiTaskReader    

.. autoclass:: deeppavlov.dataset_iterators.multitask_iterator.MultiTaskIterator

    .. automethod:: gen_batches

    .. automethod:: get_instances

.. autoclass:: deeppavlov.models.multitask_bert.multitask_bert.MultiTaskBert

    .. automethod:: train_on_batch

    .. automethod:: __call__

    .. automethod:: call

.. autoclass:: deeppavlov.models.multitask_bert.multitask_bert.MTBertTask

    .. automethod:: build

    .. automethod:: _init_graph

    .. automethod:: get_train_op

    .. automethod:: train_on_batch

    .. automethod:: get_sess_run_infer_args

    .. automethod:: get_sess_run_train_args

    .. automethod:: post_process_preds

.. autoclass:: deeppavlov.models.multitask_bert.multitask_bert.MTBertSequenceTaggingTask

    .. automethod:: get_sess_run_infer_args

    .. automethod:: get_sess_run_train_args

    .. automethod:: post_process_preds

.. autoclass:: deeppavlov.models.multitask_bert.multitask_bert.MTBertClassificationTask

    .. automethod:: get_sess_run_infer_args

    .. automethod:: get_sess_run_train_args

    .. automethod:: post_process_preds

.. autoclass:: deeppavlov.models.multitask_bert.multitask_bert.MTBertReUser

    .. automethod:: __call__

.. autoclass:: deeppavlov.models.multitask_bert.multitask_bert.InputSplitter

    .. automethod:: __call__
