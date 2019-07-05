from gym.envs.registration import register

register(
    id='box-world-v0',
    entry_point='box_world.envs:BoxWorld',
)
