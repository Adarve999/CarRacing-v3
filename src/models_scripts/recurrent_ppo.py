import os
from utils.utils import make_env
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor

def trainRecurrentPPO():
    num_envs = 8

    log_dir = "../logs/tensorboard/"
    log_name = "recurrent_ppo_carracing"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([make_env() for i in range(num_envs)])
    envs = VecMonitor(envs)

    model = RecurrentPPO(
            "CnnLstmPolicy",
            envs,
            device="cuda",
            verbose=1,
            tensorboard_log=log_dir,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            # parámetros específicos del LSTM
            policy_kwargs=dict(
                lstm_hidden_size=128,   # tamaño celdas (defecto)
                n_lstm_layers=1,        # nº de capas LSTM
                share_features_extractor=True,  # usa un solo extractor CNN
            ),
        )

    model.learn(total_timesteps=1_000_000, tb_log_name=log_name)
    model.save("../models/recurrent_ppo/recurrent_ppo_carracing")
    envs.close()


if __name__ == "__main__":
    trainRecurrentPPO()
