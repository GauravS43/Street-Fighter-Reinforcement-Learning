from sf_class import StreetFighter
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import optuna
import os

def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    } 


def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)
        
        env = StreetFighter()
        env = Monitor(env, './logs/')
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')
        
        model = PPO('CnnPolicy', env, tensorboard_log = './logs/', verbose = 0, **model_params)
        model.learn(total_timesteps = 100000)
        
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()
        
        SAVE_PATH = os.path.join('./opt/', 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
        
        return mean_reward
    except Exception as e:
        return -1000
    

study = optuna.create_study(direction = 'maximize')
study.optimize(optimize_agent, n_trials = 100, n_jobs = 1)

#save study.best_params