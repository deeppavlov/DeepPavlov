"""Registry for models. Create your models by subclassing one of the abstract model classes (RBModel
, SModel, TModel) and register it. You can assign a code name to the model in the decorator function
parentheses or leave them blank, in the last case the class name will be assigned automatically.
The name should repeat itself in your pipeline json configuration.

Example:
     @registry.register_model('my_model')
     class MyModel(TModel)

Note that you should import _REGISTRY variable and all your custom models in the entry point of
your training/inference script.
"""

from typing import Type, List

from deeppavlov.core.common.errors import ConfigError

REGISTRY = {}


def register(name: str = None) -> Type:
    """Register model. If name is not passed, the model class name is converted to snake-case."""

    def decorate(model_cls: Type, reg_name: str = None) -> Type:
        model_name = reg_name or short_name(model_cls)
        global REGISTRY
        if model_name in REGISTRY:
            raise ConfigError('{} name is already registered'.format(model_name))
        REGISTRY[model_name] = model_cls
        return model_cls

    return lambda model_cls_name: decorate(model_cls_name, name)


def short_name(cls: Type) -> str:
    return cls.__name__.split('.')[-1]


def model(name: str) -> type:
    if name not in REGISTRY:
        raise ConfigError("Model {} is not registered.".format(name))
    return REGISTRY[name]


def list_models() -> List:
    return list(REGISTRY)
