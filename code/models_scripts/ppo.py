import os
from utils.utils import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor

def trainPPO():
    num_envs = 8

    log_dir = "../logs/tensorboard/"
    log_name = "ppo_carracing"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([make_env() for i in range(num_envs)])
    envs = VecMonitor(envs)

    model = PPO("CnnPolicy", 
                envs, 
                device="cuda", 
                verbose=1, 
                n_steps=1024, 
                batch_size=64, 
                n_epochs=10, 
                gamma=0.99, 
                gae_lambda=0.95, 
                clip_range=0.2,
                ent_coef=0.01, 
                max_grad_norm=0.5, 
                tensorboard_log=log_dir
                )

    model.learn(total_timesteps=1_000_000, tb_log_name=log_name)
    model.save("../models/ppo/ppo_carracing")
    envs.close()


if __name__ == "__main__":
    trainPPO()
