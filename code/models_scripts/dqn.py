import os
from utils.utils import make_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor


def trainDQN():
    num_envs = 8

    log_dir = "../logs/tensorboard/"
    log_name = "dqn_carracing"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([make_env() for i in range(num_envs)])
    envs = VecMonitor(envs)

    model = DQN(
        "CnnPolicy",
        envs,
        device="cuda",
        verbose=1,
        buffer_size=200_000,          
        learning_starts=20_000,       
        batch_size=32,                
        train_freq=4,                
        target_update_interval=1_000,
        tau=1.0,                      
        gamma=0.99,
        learning_rate=1e-4,           
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.10,   
        max_grad_norm=10,            
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=1_000_000, tb_log_name=log_name)
    model.save("../models/dqn/dqn_carracing")
    envs.close()


if __name__ == "__main__":
    trainDQN()
