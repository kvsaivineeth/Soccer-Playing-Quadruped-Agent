import gym
from gym import spaces

from dm_control import suite
from dm_env import specs
import cv2
import uuid

def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum, 
                           high=dm_control_space.maximum, 
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'), 
                           high=float('inf'), 
                           shape=dm_control_space.shape, 
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space


class OpenCVImageViewer():
    """A simple OpenCV highgui based dm_control image viewer

    This class is meant to be a drop-in replacement for
    `gym.envs.classic_control.rendering.SimpleImageViewer`
    """
    def __init__(self, *, escape_to_exit=False):
        """Construct the viewing window"""
        self._escape_to_exit = escape_to_exit
        self._window_name = str(uuid.uuid4())
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True

    def __del__(self):
        """Close the window"""
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    def imshow(self, img):
        """Show an image"""
        # Convert image to BGR format
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        # Listen for escape key, then exit if pressed
        if cv2.waitKey(1) in [27] and self._escape_to_exit:
            exit()
    
    @property
    def isopen(self):
        """Is the window open?"""
        return self._isopen

class DMSuiteEnv(gym.Env):
    def __init__(self, domain_name, task_name, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
        self.env = suite.load(domain_name, 
                              task_name, 
                              task_kwargs=task_kwargs, 
                              environment_kwargs=environment_kwargs, 
                              visualize_reward=visualize_reward)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0/self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None
    
    def seed(self, seed):
        return self.env.task.random.seed(seed)
    
    def step(self, action):
        timestep = self.env.step(action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        timestep = self.env.reset()
        return timestep.observation
    
    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs:
            kwargs['camera_id'] = 0  # Tracking camera
        use_opencv_renderer = kwargs.pop('use_opencv_renderer', True)
        
        img = self.env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                if not use_opencv_renderer:
                    from gym.envs.classic_control import rendering
                    self.viewer = rendering.SimpleImageViewer(maxwidth=1024)
                else:
                    self.viewer = OpenCVImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()
