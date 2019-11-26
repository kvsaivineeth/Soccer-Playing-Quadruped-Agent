from dm_control import suite
from dm_control import viewer
import numpy as np


def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)


def custom_reward(physics):
    global previous_reward
    print('IN CUSTOM REWARD: {}'.format(previous_reward))
    previous_reward += .001
    return previous_reward


previous_reward = 0

task_kwargs = {
    'reward_func': custom_reward
}
env = suite.load(domain_name="quadruped", task_name="soccer", visualize_reward=True, task_kwargs=task_kwargs)

#viewer.launch(env)
# Iterate over a task set:
#for domain_name, task_name in suite.BENCHMARKING:
#  env = suite.load(domain_name, task_name)
#  print(domain_name," ",task_name)
#viewer.launch(env)
# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

# Launch the viewer application.
viewer.launch(env, policy=random_policy)

