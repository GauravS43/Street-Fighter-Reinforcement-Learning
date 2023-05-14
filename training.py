from sf_class import StreetFighter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from json import load
import os

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True
    

callback = TrainAndLoggingCallback(check_freq=10000, save_path='./train/')

with open("best_params.json", "r") as file:
    model_params = load(file)

model_params['n_steps'] = 4544
model_params['learning_rate'] = 5e-6

env = StreetFighter()
env = Monitor(env, './logs/')
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model = PPO('CnnPolicy', env, tensorboard_log =  './logs/', verbose = 1, **model_params)

model.load('./opt/trial_2_best_model.zip')

model.learn(total_timesteps=5000000, callback=callback)