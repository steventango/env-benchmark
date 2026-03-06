import jax
import jax.numpy as jnp
from foragax.env import Biome
from foragax.env import ForagaxEnv as _ForagaxEnv
from foragax.objects import DefaultForagaxObject

_WALL = DefaultForagaxObject(
    name="wall",
    blocking=True,
    collectable=False,
    color=(128, 128, 128),
    reward=0.0,
)
_FLOWER = DefaultForagaxObject(
    name="flower",
    blocking=False,
    collectable=True,
    color=(75, 181, 67),
    reward=1.0,
    regen_delay=(9, 11),
)
_THORNS = DefaultForagaxObject(
    name="thorns",
    blocking=False,
    collectable=True,
    color=(220, 50, 50),
    reward=-1.0,
    regen_delay=(9, 11),
)

_SIZE = (1_000, 1_000)


class ForagaxEnv:
    def __init__(self):
        self.env = _ForagaxEnv(
            size=_SIZE,
            aperture_size=(11, 11),
            observation_type="object",
            objects=(_WALL, _FLOWER, _THORNS),
            biomes=(
                # Single biome covering the entire world:
                # 0% wall, 10% flower, 10% thorns (matches forager.py)
                Biome(
                    start=(0, 0),
                    stop=_SIZE,
                    object_frequencies=(0.0, 0.1, 0.1),
                ),
            ),
        )
        self.env_params = self.env.default_params
        self.env_state = None
        self.key = None

        # Define the body function ONCE during initialization
        # to maintain a stable Python id() for the JAX compilation cache
        def _scan_body(carry, action):
            key, state = carry
            key, key_step = jax.random.split(key)
            _, state, _reward, _done, _info = self.env.step(
                key_step, state, action, self.env_params
            )
            return (key, state), None

        self._scan_body = _scan_body

    def reset(self, seed: int = 0):
        self.key = jax.random.key(seed)
        self.key, key_reset = jax.random.split(self.key)
        _, self.env_state = self.env.reset(key_reset, self.env_params)
        # block_until_ready ensures JIT compilation cost is included in timing
        self.env_state = jax.block_until_ready(self.env_state)

    def step(self, action):
        self.key, key_step = jax.random.split(self.key)
        action = jnp.int32(action)
        _, self.env_state, _reward, _done, _info = self.env.step(
            key_step, self.env_state, action, self.env_params
        )
        # block until JAX dispatch completes so timing is accurate
        self.env_state = jax.block_until_ready(self.env_state)

    def scan_step(self, actions):
        """Run multiple steps using jax.lax.scan. actions: int array of shape (n,)"""
        (self.key, self.env_state), _ = jax.lax.scan(
            self._scan_body, (self.key, self.env_state), actions
        )
        self.env_state = jax.block_until_ready(self.env_state)
