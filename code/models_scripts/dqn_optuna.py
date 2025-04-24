import os
from utils.utils import make_env, make_eval_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
import optuna
import torch

TOTAL_FRAMES_TUNE = 200_000
SEED = 42

def objective(trial: optuna.Trial) -> float:
    
    num_envs = 8
    envs = SubprocVecEnv([make_env() for i in range(num_envs)])
    envs = VecMonitor(envs)
    
    print("cuda" if torch.cuda.is_available() else "cpu")
    lr        = trial.suggest_loguniform("learning_rate", 5e-4, 5e-3)
    buff_size = trial.suggest_categorical("buffer_size", [100_000])
    batch_sz  = trial.suggest_categorical("batch_size", [32, 64])
    tau       = trial.suggest_uniform("tau", 0.8, 1.0)
    gamma     = trial.suggest_uniform("gamma", 0.97, 0.999)
    train_fr  = trial.suggest_categorical("train_freq", [2, 4, 8])
    expl_frac = trial.suggest_uniform("exploration_fraction", 0.05, 0.2)

    model = DQN(
        "CnnPolicy", 
        envs,
        learning_rate        = lr,
        buffer_size          = buff_size,
        batch_size           = batch_sz,
        tau                  = tau,
        gamma                = gamma,
        train_freq           = (train_fr, "step"),
        target_update_interval = 10_000,
        exploration_fraction = expl_frac,
        exploration_initial_eps = 1.0,
        exploration_final_eps   = 0.05,
        learning_starts      = 50_000,
        verbose              = 0,
        seed                 = SEED,
        device               = "cuda" if torch.cuda.is_available() else "cpu"
    )

    eval_env = make_eval_env()
    model.learn(TOTAL_FRAMES_TUNE, progress_bar=False)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    envs.close()
    return mean_reward


def trainDQN_Optuna():
    num_envs = 8

    log_dir = "../logs/tensorboard/"
    log_name = "dqn_optuna_carracing"
    os.makedirs(log_dir, exist_ok=True)

    envs = SubprocVecEnv([make_env() for i in range(num_envs)])
    envs = VecMonitor(envs)


    study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=25))

    study.optimize(objective, n_trials=20, n_jobs=1)
    
    
    BEST_PARAMS = dict(
        policy="CnnPolicy",
        env=envs,
        learning_starts=50_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        target_update_interval=10_000,
        verbose=1,
        seed=SEED,
        device="cuda"
    )
    BEST_PARAMS.update(study.best_params)   
    
    # 2) Define y construye el modelo DQN con hiperparámetros
    model = DQN(**BEST_PARAMS)

    # 3) Entrena el modelo
    model.learn(total_timesteps=1_000_000, tb_log_name=log_name)

    # 4) Guarda los pesos entrenados
    model.save("../models/dqn_optuna/dqn_optuna_carracing")

    # 5) Cierra el entorno
    envs.close()


if __name__ == "__main__":
    trainDQN_Optuna()
