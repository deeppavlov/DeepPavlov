from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.log import get_logger
log = get_logger(__name__)

_refs = {}


class Pipeline:
    def __init__(self, dataset_iterator, metrics, pipe=None, batch_size=32, metric_optimization='maximize',
                 validation_patience=5, val_every_n_epochs=3, log_every_n_batches=0, log_every_n_epochs=1,
                 validate_best=True, test_best=True, in_x=None, in_y=None, out=None):
        # Type of metrics write
        # metrics = [('ner_f1', ner_f1), ('accuracy', per_item_accuracy)]

        # main attributes
        self.pipe = pipe or []
        self.dataset_iterator = dataset_iterator
        # TODO maybe fix
        self.metrics = metrics

        # inputs and outputs
        if isinstance(in_x, str):
            in_x = [in_x]
        if isinstance(in_y, str):
            in_y = [in_y]
        if isinstance(out, str):
            out = [out]
        self.in_x = in_x or ['x']
        self.in_y = in_y or ['y']
        self.out = out or self.in_x

        # train params
        self.batch_size = batch_size
        self.metric_optimization = metric_optimization
        self.validation_patience = validation_patience
        self.val_every_n_epochs = val_every_n_epochs
        self.log_every_n_batches = log_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.validate_best = validate_best
        self.test_best = test_best

        # memory
        self.op_mem = dict()
        self.res_mem = dict()

    def fill_components(self):
        assert len(self.pipe) == 0

        x, y = self.dataset_iterator.get_instances('train')

        for component in self.pipe:
            if isinstance(component, tuple):
                op = component[0]
                att = component[1]
                # TODO надо присвоить указанным компонентам нужные значения
                for key, val in att.items():
                    if hasattr(op, key):
                        getattr(op, )

            else:
                if callable(component):
                    component = component()
                else:
                    pass

            if hasattr(component, 'fit_on'):
                component: Estimator
                component.fit(x, y)
                component.save()

            if hasattr(component, 'fit_on_batch'):
                component: Estimator
                component.fit_batches(self.dataset_iterator, self.batch_size)
                component.save()

            else:
                self.res_mem[component.out] = component(x, y)

            self.op_mem[component.name] = component

        return self
