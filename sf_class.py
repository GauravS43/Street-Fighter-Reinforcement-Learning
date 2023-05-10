from gym import Env
from gym.spaces import MultiBinary, Box
from retro import make, Actions 
from cv2 import cvtColor, resize, INTER_CUBIC, COLOR_BGR2GRAY
from numpy import reshape, uint8


class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84,84,1), dtype = uint8)
        self.action_space = MultiBinary(12)
        self.game = env = make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions = Actions.FILTERED)
        
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        
        reward = info['score'] - self.score
        self.score = info['score']
        
        return frame_delta, reward, done, info
    
    def render(self, *args, **kwargs):
        self.game.render()
    
    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs
    
    def preprocess(self, observation):
        gray = cvtColor(observation, COLOR_BGR2GRAY)
        res = resize(gray, (84,84), interpolation = INTER_CUBIC)
        channels = reshape(res, (84,84,1))
        return channels
    
    def close(self):
        self.game.close()