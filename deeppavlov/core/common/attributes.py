from typing import Type, Callable


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
        def wrapped(self, *args):
            if getattr(self, attr):
                return f(self, *args)
            else:
                print("'{0}' is False, doing nothing."
                      " Set {0} to True in json config "
                      "if you'd like the {1} to proceed.".format(attr, f))

        return wrapped

    return _check_attr_true


def run_alt_meth_if_no_path(alt_f: Callable, attr: str):
    def _run_alt_meth(f):
        def wrapped(self, *args):
            if self.model_path_.exists():
                return f(self, *args)
            else:
                setattr(self, attr, True)
                # TODO somehow pass the wrapped function name
                print("'{0}' is set to False, though the path doesn't exist. Can't do {1}. "
                      "Ignoring False, setting '{0}' to True. Proceeding anyway.".format(attr, f))
                return alt_f(self, *args)

        return wrapped

    return _run_alt_meth


def check_path_exists(path_type='file'):
    def _check_path_exists(f: Callable):
        def wrapped(self, *args, **kwargs):
            if path_type == 'file':
                if self.model_path_.exists():
                    return f(self, *args, **kwargs)
            elif path_type == 'dir':
                if self.model_path_.parent.exists():
                    return f(self, *args, **kwargs)
            raise FileNotFoundError(
                "{}.model_path doesn't exist. Check if there is a pretrained model."
                "If there is no pretrained model, you might want to set 'train_now' to true "
                "in the model json config and run training first.".format(
                    self.__class__.__name__))

        return wrapped

    return _check_path_exists
