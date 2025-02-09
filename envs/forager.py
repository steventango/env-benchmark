from forager.Env import ForagerEnv, ForagerConfig
from forager.objects import Wall, Flower, Thorns

config = ForagerConfig(
    size=1_000, # equivalently: (1_000, 1_000)

    # tell the env what types of objects you expect to see
    object_types={
        'wall': Wall,
        'flower': Flower,
        'thorns': Thorns,
    },

    # 'objects' mode is a (ap_x, ap_y, #objects) binary tensor
    # 'colors' mode is a (ap_x, ap_y, 3) uint8 tensor
    observation_mode='objects',

    # controls how far the agent can see around itself
    # should always be odd---this way the agent is centered
    # need not be square
    aperture=(65, 65),
)


class Env:
    def __init__(self):
        self.env = None

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed):
        config.seed = seed
        self.env = ForagerEnv(config)
        self.env.generate_objects(0.1, 'flower')
        self.env.generate_objects(0.1, 'thorns')
        return self.env.start()
