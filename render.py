from sf_class import StreetFighter
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# random actions
# env = StreetFighter()
# obs = env.reset()
# for matches in range(1):
#     done = False
#     while not done:
#         env.render()
#         obs, rew, done, info = env.step(env.action_space.sample())


# best model
model = PPO.load('./train/best_model_300000.zip')
env = StreetFighter()
env = Monitor(env, './logs/')
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

obs = env.reset()
for matches in range(1):
    done = False
    while not done:
        env.render()
        action = model.predict(obs)[0]
        obs, rew, done, info = env.step(action)

env.close()
