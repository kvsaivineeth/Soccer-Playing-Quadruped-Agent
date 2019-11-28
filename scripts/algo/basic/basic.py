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
# # Basic `dm_control` playground
# Just trying to mess around with the custom `dm_control` stack.
#
# ## Imports
# -

from dm_control import suite
from dm_control import viewer
import numpy as np


# ## Custom reward function

def custom_reward(physics):
  global timestep_c
  
  if timestep_c % 500 == 0:
    print('===========')
    print('timestep: {}'.format(timestep_c))
    print('ball in goal: {}'.format(physics.ball_in_goal()))
    print('ball_to_goal_distance: {}'.format(physics.ball_to_goal_distance()))
    print('===========')
    print()
    
  return 600


# ## Observation to input

def to_input(obs):
  pass


# ## Environment creation

# +
task_kwargs = {
  'reward_func': custom_reward
}

env = suite.load(domain_name="quadruped", 
                 task_name="soccer", 
                 visualize_reward=True, 
                 task_kwargs=task_kwargs)
# -

# ## Step the simulator

# +
timestep_c = 0
timestep = env.reset()

while not timestep.last():
  timestep = env.step(np.random.random(12,))
  timestep_c += 1
