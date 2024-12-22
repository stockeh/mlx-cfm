import importlib
import mlx.nn as nn
import mlx.core as mx

from mlx.utils import tree_flatten, tree_map

from typing import List, Type, Union


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_activation(activation_f: str) -> Type:
    package_name = "mlx.nn.layers.activations"
    module = importlib.import_module(package_name)

    activations = [getattr(module, attr) for attr in dir(module)]
    activations = [
        cls
        for cls in activations
        if isinstance(cls, type) and issubclass(cls, nn.Module)
    ]
    names = [cls.__name__.lower() for cls in activations]

    try:
        index = names.index(activation_f.lower())
        return activations[index]
    except ValueError:
        raise NotImplementedError(
            f"get_activation: {activation_f=} is not yet implemented."
        )


class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @property
    def num_params(self):
        return sum(x.size for k, x in tree_flatten(self.parameters()))

    @property
    def shapes(self):
        return tree_map(lambda x: x.shape, self.parameters())

    def summary(self):
        print(self)
        print(f"Number of parameters: {self.num_params}")

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError("Subclass must implement this method")


class MLP(Base):
    def __init__(
        self,
        n_inputs: int,
        n_hiddens_list: Union[List, int],
        n_outputs: int,
        activation_f: str = "selu",
        time_varying=False,
    ):
        super().__init__()

        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]

        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)

        activation = get_activation(activation_f)
        self.time_varying = time_varying
        self.layers = []
        ni = n_inputs + (1 if time_varying else 0)
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                self.layers.append(nn.Linear(ni, n_units))
                self.layers.append(activation())
                ni = n_units
        self.layers.append(nn.Linear(ni, n_outputs))

    def __call__(self, t, x, repeat=True):
        if repeat:
            t = mx.repeat(t, x.shape[0])
        if self.time_varying:
            x = mx.concatenate([x, t[:, None]], axis=-1)

        for l in self.layers:
            x = l(x)
        return x
