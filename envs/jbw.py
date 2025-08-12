from math import log, pi

import jbw


class SimpleAgent(jbw.Agent):
    def __init__(self, simulator, load_filepath=None):
        super(SimpleAgent, self).__init__(simulator, load_filepath)

    def do_next_action(self, action):
        self.move(jbw.direction.Direction(action))

    def save(self, filepath):
        pass

    def _load(self, filepath):
        pass


items = []
items.append(
    jbw.Item(
        "JellyBean",
        [1.64, 0.54, 0.40],
        [0.82, 0.27, 0.20],
        [0, 0],
        [0, 0],
        False,
        0.0,
        intensity_fn=jbw.IntensityFunction.CONSTANT,
        intensity_fn_args=[log(0.1)],
        interaction_fns=[
            [jbw.InteractionFunction.ZERO],
            [jbw.InteractionFunction.ZERO],
        ],
    )
)
items.append(
    jbw.Item(
        "JellyBean",
        [0.68, 0.01, 0.99],
        [0.68, 0.01, 0.99],
        [0, 0],
        [0, 0],
        False,
        0.0,
        intensity_fn=jbw.IntensityFunction.CONSTANT,
        intensity_fn_args=[log(0.1)],
        interaction_fns=[
            [jbw.InteractionFunction.ZERO],
            [jbw.InteractionFunction.ZERO],
        ],
    )
)




class JBWEnv:
    def __init__(self):
        self.reset()

    def reset(self, seed: int = 0):
        config = jbw.SimulatorConfig(
            max_steps_per_movement=1,
            vision_range=5,
            allowed_movement_directions=[
                jbw.ActionPolicy.ALLOWED,
                jbw.ActionPolicy.ALLOWED,
                jbw.ActionPolicy.ALLOWED,
                jbw.ActionPolicy.ALLOWED,
            ],
            allowed_turn_directions=[
                jbw.ActionPolicy.DISALLOWED,
                jbw.ActionPolicy.DISALLOWED,
                jbw.ActionPolicy.DISALLOWED,
                jbw.ActionPolicy.DISALLOWED,
            ],
            no_op_allowed=False,
            patch_size=32,
            mcmc_num_iter=4000,
            items=items,
            agent_color=[0.0, 0.0, 1.0],
            collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
            agent_field_of_view=2 * pi,
            decay_param=0,
            diffusion_param=0,
            deleted_item_lifetime=2000,
            seed=seed,
        )
        sim = jbw.Simulator(sim_config=config)
        self.agent = SimpleAgent(sim)

    def step(self, action):
        self.agent.do_next_action(action)
