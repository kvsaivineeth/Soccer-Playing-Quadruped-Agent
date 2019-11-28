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

# # TRPO Implementation
#
# Try to create a basic policy to get the agent to try to kick the ball to the target. The paper for this algorithm can be found [here](https://arxiv.org/pdf/1502.05477.pdf).

# ## Setup
# Hyperparameters and other preliminaries.
#
# ### Imports

from dm_control import suite
from dm_control import viewer
import numpy as np
import torch

# ### Constants

# Get the training device


