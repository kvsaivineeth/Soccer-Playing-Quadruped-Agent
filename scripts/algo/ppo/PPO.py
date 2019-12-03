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

# + [markdown] toc-hr-collapsed=false toc-hr-collapsed=false
# ### PPO Implementation
#
# Try to create a basic policy to get the agent to try to kick the ball to the target. The paper for this algorithm can be found [here](https://arxiv.org/pdf/1707.06347.pdf).

# + [markdown] toc-hr-collapsed=false toc-hr-collapsed=false
# ## Setup
# Hyperparameters and other preliminaries.
#
# ### Imports

# +
from dm_control import suite
from dm_control import viewer
import numpy as np
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# + [markdown] toc-hr-collapsed=false
# ### Constants
# -

# Get the training device and dynamically set it to the GPU if needed.

_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Constants of the MuJoCo environment. `_c` denotes the *cardinality* or the *count* of the value.

_walls_c = 3
_num_walls = 4
_ball_state_c = 9
_egocentric_state_c = 44

# Network Hyperparameters:

# +
_INPUT_DIM = _walls_c * _num_walls + _ball_state_c + _egocentric_state_c
_GAMMA = 0.99  # Discount factor
_MINIBATCH_SIZE = 32
_LEARNING_RATE = 0.0015
_ITERATIONS = 1000000
_EPOCHS = 10
_MEMORY_SIZE = 10000

_HIDDEN_LAYER_1 = 64
_HIDDEN_LAYER_2 = 32

_SEED = 2019
_EPSILON = 0.2  # Probability clip
_DROPOUT_PROB = 0.5
# -

# ### Set seeds

torch.manual_seed(_SEED)
np.random.seed(_SEED)
random.seed(_SEED)


# ## Define the environment

# ### Define observation and agent inputs
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


# ### Define reward function

def reward(physics):
  return 0


# + [markdown] toc-hr-collapsed=false
# ### Create the environment

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

_OUTPUT_DIM = env.action_spec().shape[0]


# ## Model Creation
#
# The model is a simple feed foward network with 2 hidden layers. Note that in this Actor-Critic model, the actor tries to fit to the policy and the critic tries to fit to the value function. Additionally, in this case both the actor and the critic share the same subnet to *(hopefully)* converge faster.

class PPO(nn.Module):
  def __init__(self):
    super(PPO, self).__init__()
    
    self.network_base = nn.Sequential(
      nn.Linear(_INPUT_DIM, _HIDDEN_LAYER_1), nn.Dropout(_DROPOUT_PROB), nn.Tanh(),
      nn.Linear(_HIDDEN_LAYER_1, _HIDDEN_LAYER_2), nn.Dropout(_DROPOUT_PROB), nn.Tanh(),
    )
    
    self.policy_mu = nn.Linear(_HIDDEN_LAYER_2, _OUTPUT_DIM)
    self.policy_log_std = nn.Parameter(torch.randn(_OUTPUT_DIM))
    self.value = nn.Linear(_HIDDEN_LAYER_2, 1)
    
  def forward(self, x):
    latent_state = self.network_base(x)
    
    mus = self.policy_mu(latent_state)
    sigmas = torch.exp(self.policy_log_std)
    value_s = self.value(latent_state)
    
    return mus, sigmas, value_s


# Create the network and verify the layers are good as-is.

PPO()

# ## Training

# ### Memory Managment
# Create structures and methods to help manage the memory 

# #### Exploration Transition
# Create a data type to store the transition during exploration. Can't compute advantages and such because the trajectory won't be finished by then.

Transition = collections.namedtuple('Transition',
                                    ['state',
                                     'action',
                                     'action_dist',
                                     'value',
                                     'reward',
                                     'mask'])

# #### Training Memory
# Create a data to store memories to sample for training.

Memory = collections.namedtuple('Memory',
                                ['state', 
                                 'action',
                                 'entropy',
                                 'value',
                                 'value_target',
                                 'advantage'])


# ### Define loss and training functions

def update_model(model, memory, n_steps):
  pass


# ## Exploration and actually training

# Create target and stable nets for training

policy = PPO().float().to(_DEVICE)
policy_old = PPO().float().to(_DEVICE)
policy_old.load_state_dict(policy.state_dict())

# Explore, write to memory, and train!

for i in range(_ITERATIONS):
  transitions = []
  total_reward = 0
  
  timestep = env.reset()
  
  # Explore using the previous policy
  while not timestep.last():
    input_ = to_input(timestep.observation)
    state = torch.from_numpy(input_).float().to(_DEVICE)
    
    with torch.no_grad():
      mus, sigmas, v_s = policy_old(state)
      
    actions_dist = torch.distributions.normal.Normal(mus, sigmas)
    action = actions_dist.sample().numpy()
    
    timestep = env.step(action)
    
    reward = timestep.last() or timestep.reward
    mask = 1 if timestep.last() else 0
    total_reward += timestep.reward
    
    transitions.append(Transition(state=input_, action=action, action_dist=actions_dist,
                                  value=v_s.item(), mask=mask, reward=reward))
    
  # Create the final memory to sample
  memory = []
  
  # Compute advantages using GAE
  advantages = []
  prev_v_target = prev_v = prev_adv = 0
  for trans in reversed(transitions):
    # Caculate advantages and proper V(s) values
    v_target = trans.reward + _GAMMA * prev_v_target * trans.mask
    delta = trans.reward + _GAMMA * prev_v * trans.mask - trans.value
    adv = delta + _GAMMA * prev_adv * trans.mask
    
    # Insert into memory
    advantages.insert(0, adv)
    memory.insert(0, Memory(
      value_target=v_target, advantage=None,  # Replace advantages with standardized advantages
      entropy=trans.action_dist.entropy().numpy(),
      **{k: v for k, v in trans._asdict().items()
              if k in set(Memory._fields) & set(Transition._fields)}))
    
    # Update for the next iteration
    prev_v_target = v_target
    prev_v = trans.value
    prev_adv = adv
    
    # TODO: Ensure that get_termination works and doesn't default to +1
    
  # Normalize advantages
  advs = np.array(advantages)
  advs = (advs - advs.mean()) / advs.std()
  
  for t, norm_adv in enumerate(advs):
    memory[t] = memory[t]._replace(advantage=norm_adv)
    
  # Train
  
  break


