import os
from utils.utils import make_env
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor

def trainTRPO():
    num_envs = 4

    log_dir = "../logs/tensorboard/"
    log_name = "trpo_carracing"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([make_env() for i in range(num_envs)])
    envs = VecMonitor(envs)
    
    model = TRPO(
        policy="CnnPolicy",
        env=envs,
        n_steps=1024,       
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log=log_dir,
        device="cuda",
        verbose=1,
    )


    model.learn(total_timesteps=700_000, tb_log_name=log_name)
    model.save("../models/trpo/trpo_carracing")
    envs.close()


if __name__ == "__main__":
    trainTRPO()

