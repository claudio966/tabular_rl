from gymnasium.envs.registration import register

register(
    id='gambler',
    entry_point='gambler_ruin.envs:GamblerRuin',
)
