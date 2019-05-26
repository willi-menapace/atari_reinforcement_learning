
import ptan.common.wrappers as ptanwrap
import gym
import gym.spaces as spaces
import numpy as np
import cv2

class PacmanResizeAndRecolorFrame(gym.ObservationWrapper):
    def __init__(self, env=None, recolor_eatable_ghosts=True):
        super(PacmanResizeAndRecolorFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(92, 84, 1), dtype=np.uint8)
        self.recolor_eatable_ghosts = recolor_eatable_ghosts

    def observation(self, obs):
        return PacmanResizeAndRecolorFrame.process(obs, self.recolor_eatable_ghosts)

    @staticmethod
    def get_channel_mask(image, center_values, offset=10):
        bool_mask = np.ones_like(image[:,:,0]).astype(np.bool)
        for channel, value in enumerate(center_values):
            bool_mask = bool_mask & ((image[:,:, channel] < (value + offset)) & (image[:,:, channel] > (value - offset)))

        return bool_mask

    @staticmethod
    def get_uneatable_ghosts_mask(image):
        mask = PacmanResizeAndRecolorFrame.get_channel_mask(image, (200, 72, 72))
        mask = mask | PacmanResizeAndRecolorFrame.get_channel_mask(image, (84, 184, 153))
        mask = mask | PacmanResizeAndRecolorFrame.get_channel_mask(image, (180, 122, 48))
        mask = mask | PacmanResizeAndRecolorFrame.get_channel_mask(image, (198, 89, 179))

        return mask

    @staticmethod
    def get_eatable_ghosts_mask(image):
        mask = PacmanResizeAndRecolorFrame.get_channel_mask(image, (66, 114, 194))

        return mask

    @staticmethod
    def get_pacman_mask(image):
        mask = PacmanResizeAndRecolorFrame.get_channel_mask(image, (210, 164, 74))

        return mask

    #Pacman color maps
    # 84 184 153 light blue 72l
    # 200 72 72 red 74
    # 180 122 48 yellow 71
    # 210 164 74 pacman 82
    # 198 89 179 pink 78
    # 66 114 194 eatable ghost 76
    # 0 28 136 background 53

    #Transforms the image in 84 x 84 images aligned at the top of the screen
    @staticmethod
    def process(frame, recolor_eatable_ghosts=False):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        new_img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        if recolor_eatable_ghosts:
            #Makes eatable ghosts black to distinguish them from uneatable ghosts in grayscale
            eatable_ghost_mask = PacmanResizeAndRecolorFrame.get_eatable_ghosts_mask(img)
            new_img[eatable_ghost_mask] = 0

        resized_screen = cv2.resize(new_img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[:92, :] #Aligns at top
        #Eliminates black band underneath and inserts a contour at the top
        x_t[0] = 145
        x_t[90:92] = 145
        x_t = np.reshape(x_t, [92, 84, 1])
        return x_t.astype(np.uint8)

class PacmanRewardManager(gym.Wrapper):
    def __init__(self, env=None):
        super(PacmanRewardManager, self).__init__(env)


    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        '''
        # Eating dot
        if reward == 10:
            reward = 0.5
        # Eating pill
        elif reward == 50:
            reward = 1
        # Eating ghost
        elif reward == 200:
            reward = 2
        # Eating fruit
        elif reward == 100:
            reward = 0.5
        # Being eaten
        elif done == True:
            reward = -7
        # Finishing level
        elif reward > 200:
            reward = 7
        else:
            reward = np.sign(reward)
        '''

        # Eating dot
        if reward == 10:
            reward = 0.5
        # Eating pill
        elif reward == 50:
            reward = 0
        # Eating ghost
        elif reward == 200:
            reward = 1.5
        # Eating fruit
        elif reward == 100:
            reward = 0.5
        # Being eaten
        elif done == True:
            reward = -2
        # Finishing level
        elif reward > 200:
            reward = 4
        else:
            reward = np.sign(reward)

        return obs, reward, done, info

    def reset(self):

        obs = self.env.reset()
        return obs

class AtlantisResizeAndRecolorFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtlantisResizeAndRecolorFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(92, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return AtlantisResizeAndRecolorFrame.process(obs)

    @staticmethod
    def get_channel_mask(image, center_values, offset=10):
        bool_mask = np.ones_like(image[:,:,0]).astype(np.bool)
        for channel, value in enumerate(center_values):
            bool_mask = bool_mask & ((image[:,:, channel] < (value + offset)) & (image[:,:, channel] > (value - offset)))

        return bool_mask

    @staticmethod
    def get_atlantis_mask(image):
        mask = AtlantisResizeAndRecolorFrame.get_channel_mask(image, (210, 164, 74))

        return mask

    #Transforms the image in 84 x 84 images aligned at the top of the screen
    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        new_img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        resized_screen = cv2.resize(new_img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[3:95, :]  # Aligns at top
        x_t = np.reshape(x_t, [92, 84, 1])
        return x_t.astype(np.uint8)

def wrap_pacman(env, stack_frames=4, episodic_life=True, reward_reshaping=True, recolor_eatable_ghosts=True):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    if episodic_life:
        env = ptanwrap.EpisodicLifeEnv(env)
    #Pacman skips 90 frames at end of life. This skips a random amount up to 90
    env = ptanwrap.NoopResetEnv(env, noop_max=90)
    env = ptanwrap.MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = ptanwrap.FireResetEnv(env)
    env = PacmanResizeAndRecolorFrame(env, recolor_eatable_ghosts)
    env = ptanwrap.ImageToPyTorch(env)
    env = ptanwrap.FrameStack(env, stack_frames)
    if reward_reshaping:
        env = PacmanRewardManager(env)
    return env

def wrap_pacman_testing(env):
    return wrap_pacman(env, episodic_life=False, reward_reshaping=False)

def wrap_pacman_testing_no_recolor(env):
    return wrap_pacman(env, episodic_life=False, reward_reshaping=False, recolor_eatable_ghosts=False)

def wrap_atlantis(env, stack_frames=4, episodic_life=False, reward_reshaping=True):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    if episodic_life:
        env = ptanwrap.EpisodicLifeEnv(env)
    env = ptanwrap.NoopResetEnv(env, noop_max=30)
    env = ptanwrap.MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = ptanwrap.FireResetEnv(env)
    env = AtlantisResizeAndRecolorFrame(env)
    env = ptanwrap.ImageToPyTorch(env)
    env = ptanwrap.FrameStack(env, stack_frames)
    if reward_reshaping:
        env = ptanwrap.ClippedRewardsWrapper(env)
    return env

def wrap_demon_attack(env, stack_frames=4, episodic_life=True):
  """Apply a common set of wrappers for Atari games."""
  assert 'NoFrameskip' in env.spec.id
  if episodic_life:
    env = ptanwrap.EpisodicLifeEnv(env)
  env = ptanwrap.NoopResetEnv(env, noop_max=30)
  env = ptanwrap.MaxAndSkipEnv(env, skip=4)
  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = ptanwrap.FireResetEnv(env)
  env = ptanwrap.ProcessFrame84(env)
  env = ptanwrap.ImageToPyTorch(env)
  env = ptanwrap.FrameStack(env, stack_frames)
  env = ptanwrap.ClippedRewardsWrapper(env)
  return env

def wrap_demon_attack_test(env, stack_frames=4):
  """Apply a common set of wrappers for Atari games."""
  assert 'NoFrameskip' in env.spec.id
  #if episodic_life:
    #env = ptanwrap.EpisodicLifeEnv(env)
  env = ptanwrap.NoopResetEnv(env, noop_max=30)
  env = ptanwrap.MaxAndSkipEnv(env, skip=4)
  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = ptanwrap.FireResetEnv(env)
  env = ptanwrap.ProcessFrame84(env)
  env = ptanwrap.ImageToPyTorch(env)
  env = ptanwrap.FrameStack(env, stack_frames)
  #env = ptanwrap.ClippedRewardsWrapper(env)
  return env