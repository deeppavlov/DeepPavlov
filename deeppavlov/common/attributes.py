from typing import Type


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
                    # we did not find a match, should be rare, but prepare for it
            raise NotImplementedError('{} does not set the abstract attribute <unknown>'
                                      .format(t.__name__))
