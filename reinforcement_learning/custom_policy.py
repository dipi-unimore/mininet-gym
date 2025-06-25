from stable_baselines3.dqn.policies import DQNPolicy
import torch.nn as nn

class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(
            *args,
            **kwargs
        )