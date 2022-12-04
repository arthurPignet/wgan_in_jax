import haiku as hk
import chex
import jax

from typing import *
 

class GeneratorNetwork(hk.Module):

    def __init__(self, data_dim: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.data_dim = data_dim

    def __call__(self, s: chex.Array) -> chex.Array:
        """
        :param s: state
        :return: Q(s,a)
        """
        
        h = hk.Linear(400)(s)
        h = jax.nn.relu(h)
        h = hk.Linear(400)(h)
        h = jax.nn.relu(h)

        h = hk.Linear(300)(h)
        h = jax.nn.relu(h)

        return hk.Linear(self.data_dim, w_init=hk.initializers.RandomUniform(-3e-3, 3e-3))(h)

class CriticNetwork(hk.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    def __call__(self, s: chex.Array) -> chex.Array:
        """
        :param s: state
        :param a: action
        :param is_training: True if training mode, for BatchNorm
        :return: Q(s,a)
        """

        h = s
        h = hk.Linear(400)(h)
        h = jax.nn.relu(h)

        h = hk.Linear(300)(h)
        h = jax.nn.relu(h)

        return hk.Linear(1, w_init=hk.initializers.RandomUniform(-3e-3, 3e-3))(h)

