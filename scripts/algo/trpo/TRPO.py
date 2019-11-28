# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] toc-hr-collapsed=false
# # TRPO Implementation
#
# Try to create a basic policy to get the agent to try to kick the ball to the target. The paper for this algorithm can be found [here](https://arxiv.org/pdf/1502.05477.pdf).

# + [markdown] toc-hr-collapsed=false
# ## Setup
# Hyperparameters and other preliminaries.
#
# ### Imports

# +
from dm_control import suite
from dm_control import viewer
import numpy as np

import torch
# -

# ### Constants

# Get the training device and dynamically set it to the GPU if needed.

_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ## Define observation and agent inputs

def to_input(obs):
  pass


# ## Define reward function

def reward(physics):
  pass


# ## Create the environment

# +
task_kwargs = {
  'reward_func': reward
}

env = suite.load(domain_name="quadruped", 
                 task_name="soccer", 
                 visualize_reward=True, 
                 task_kwargs=task_kwargs)
