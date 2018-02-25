from deeppavlov.core.models.component import Component


class Chainer(Component):
    def __init__(self, in_params, out_params, *args, **kwargs):
        self.pipe = []
        self.in_params = in_params
        self.out_params = out_params

    def append(self, in_params, out_params, component):
        self.pipe.append((in_params, out_params, component))

    def __call__(self, *args, to_return=None, **kwargs):
        if to_return is None:
            to_return = self.out_params

        args = list(args)
        mem = {}
        for k in self.in_params:
            try:
                mem[k] = kwargs.pop(k)
            except KeyError:
                mem[k] = args.pop(0)

        for in_params, out_params, component in self.pipe:
            res = component(*[mem[k] for k in in_params])
            if len(out_params) == 1:
                mem[out_params[0]] = res
            else:
                for k, v in zip(out_params, res):
                    mem[k] = v

        res = [mem[k] for k in to_return]
        if len(res) == 1:
            return res[0]
        return res
