import os

from typing import Type, Callable

from deeppavlov.core.common.errors import ConfigError


class abstract_attribute(object):
    def __get__(self, obj, t: Type):
        for cls in type.__mro__:
            for name, value in cls.__dict__.items():
                if value is self:
                    this_obj = obj if obj else t
                    raise NotImplementedError(
                        '{} does not have the attribute {} '
                        '(abstract from class {}'.format(this_obj,
                                                         name,
                                                         cls.__name__))
            raise NotImplementedError('{} does not set the abstract attribute <unknown>'
                                      .format(t.__name__))


def check_attr_true(attr: str):
    def _check_attr_true(f: Callable):
        def wrapped(self, *args, **kwargs):
            if getattr(self, attr):
                return f(self, *args, **kwargs)
            else:
                print("'{0}' is False, doing nothing."
                      " Set {0} to True in json config "
                      "if you'd like the {1} to proceed.".format(attr, f))

        return wrapped

    return _check_attr_true


def run_alt_meth_if_no_path(alt_f: Callable, attr: str):
    def _run_alt_meth(f):
        def wrapped(self, *args, **kwargs):
            if self.model_path_.exists():
                if self.model_path_.is_file() or (
                            self.model_path_.is_dir() and os.listdir(str(self.model_path_))):
                    try:
                        return f(self, *args, **kwargs)
                    except ConfigError:
                        print('There are no needed model files')
            setattr(self, attr, True)
            print(
                "Attribute '{0}' is set to False, though the path doesn't exist or there"
                " is no ser data at the given path.\nCan't do {1}()."
                " Instead will do {2}()".format(attr, str(f).split()[1], str(alt_f).split()[1]))
            return alt_f(self, *args, **kwargs)

        return wrapped

    return _run_alt_meth


def check_path_exists(path_type='file'):
    def _chek_path_exists(f: Callable):
        def wrapped(self, *args, **kwargs):
            if path_type == 'file':
                if self.model_path_.exists():
                    return f(self, *args, **kwargs)
            elif path_type == 'dir':
                if self.model_path_.parent.exists():
                    return f(self, *args, **kwargs)
            raise FileNotFoundError(
                "{}.model_path doesn't exist. Check if there is a pretrained model."
                "If there is no a pretrained model, you might want to set 'train_now' to true "
                "in the model json config and run training first.".format(self.__class__.__name__))

        return wrapped

    return _chek_path_exists
