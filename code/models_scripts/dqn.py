import os
from utils.utils import make_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor


def trainDQN():
    num_envs = 6

    log_dir = "../logs/tensorboard/"
    log_name = "dqn_carracing"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([make_env() for i in range(num_envs)])
    envs = VecMonitor(envs)

    model = DQN(
        policy="CnnPolicy",
        env=envs,
        buffer_size        = 200_000,
        learning_starts    = 20_000,
        batch_size         = 64,
        gamma              = 0.99,
        target_update_interval = 10_000,
        train_freq         = 4,            
        gradient_steps     = 1,
        exploration_fraction = 0.3,        
        exploration_final_eps = 0.05,
        policy_kwargs      = dict(net_arch=[256, 256]),
        tensorboard_log    = log_dir,
        device             = "cuda",
        verbose            = 1,
    )

    model.learn(total_timesteps=2_000_000, tb_log_name=log_name)
    model.save("../models/dqn/dqn_carracing")
    envs.close()


if __name__ == "__main__":
    trainDQN()
