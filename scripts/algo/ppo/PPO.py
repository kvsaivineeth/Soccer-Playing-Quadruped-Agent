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
# # PPO Implementation
#
# Try to create a basic policy to get the agent to try to kick the ball to the target. The paper for this algorithm can be found [here](https://arxiv.org/pdf/1707.06347.pdf).

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

# +
# Computational device 
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Cardinalities
_walls_c = 3
_num_walls = 4
_ball_state_c = 9
_egocentric_state_c = 44
_INPUT_DIM = _walls_c * _num_walls + _ball_state_c + _egocentric_state_c


# -

# ## Define observation and agent inputs
#
# Here, an agent observation is converted into the input for TRPO. The observed features that are used are: 
# * Wall vectors for the left, right, top, and back walls of the goal
# * The ball x,y,z positions and velocicties relative to the agent
# * The state of the agent itself (joints, etc)
#
# The features are converted to be 1-dimensional and then concatenated as follows:
# $$\left[ \matrix{ left \cr
#                   right \cr
#                   top \cr
#                   back \cr
#                   ball-state \cr
#                   egocentric-state} \right]$$

def to_input(obs):
  left, right, top, back = obs['goal_walls_positions']
  ball_state = obs['ball_state']
  egocentric_state = obs['egocentric_state']
  
  return np.concatenate((
    left.ravel(),
    right.ravel(),
    top.ravel(),
    back.ravel(),
    ball_state.ravel(),
    egocentric_state.ravel()
  ))


# ## Define reward function

def reward(physics):
  return 0


# ## Create the environment

# +
task_kwargs = {
  'reward_func': reward
}

env = suite.load(domain_name="quadruped", 
                 task_name="soccer", 
                 visualize_reward=True, 
                 task_kwargs=task_kwargs)
# -

# Get the dynamic output required for TRPO

_OUTPUT_DIM = env.action_spec().shape

#

timestep = env.reset()

to_input(timestep.observation)

env.action_spec().shape
