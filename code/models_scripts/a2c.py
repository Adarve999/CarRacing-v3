import os
from utils.utils import make_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

def trainA2C():
    num_envs = 8

    log_dir = "../logs/tensorboard/"
    log_name = "a2c_carracing"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    envs = VecMonitor(envs)

    model = A2C(
        "CnnPolicy",
        envs,
        device="cuda",
        verbose=1,
        n_steps=128,          # Steps per environment per update
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        max_grad_norm=0.5,
        learning_rate=1e-4,   # Slightly lower than default for stability
        tensorboard_log=log_dir
    )

    model.learn(total_timesteps=1_000_000, tb_log_name=log_name)
    model.save("../models/a2c/a2c_carracing")
    envs.close()

if __name__ == "__main__":
    trainA2C()
