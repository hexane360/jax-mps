from jax import numpy as jnp
from jax import random

from .util import OperationTestConfig


def make_donation_op_configs():
    with OperationTestConfig.module_name("donation"):
        yield from [
            # Same shape and dtype as the output: lowered via `tf.aliasing_output`.
            OperationTestConfig(
                lambda a, b: a + b,
                lambda key: random.normal(key, (5,)),
                lambda key: random.normal(key, (5,)),
                donate_argnums=(0,),
                name="aliasing_output",
            ),
            # Same size but different shape: lowered via `jax.buffer_donor`.
            OperationTestConfig(
                lambda a, b: (a + b).reshape(6),
                lambda key: random.normal(key, (2, 3)),
                lambda key: random.normal(key, (2, 3)),
                donate_argnums=(0,),
                name="buffer_donor",
            ),
            # Reclaiming an input feeding several ops must not corrupt the result.
            OperationTestConfig(
                lambda p: p * 2.0 + p * 3.0 + jnp.sin(p),
                lambda key: random.normal(key, (5,)),
                donate_argnums=(0,),
                name="multiple_consumers",
            ),
        ]
