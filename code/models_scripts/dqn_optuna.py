import os
import torch
import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from utils.utils import make_env, make_eval_env

TOTAL_FRAMES_TUNE = 1_500_000  # 1.5M frames por trial
SEED = 42


def objective(trial: optuna.Trial) -> float:
    num_envs = 8
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    envs = VecMonitor(envs)

    print("cuda" if torch.cuda.is_available() else "cpu")

    # Búsqueda más enfocada
    lr = trial.suggest_loguniform("learning_rate", 5e-4, 5e-3)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.05, 0.4)

    model = DQN(
        "CnnPolicy",
        envs,
        learning_rate=lr,
        buffer_size=100_000,
        batch_size=32,
        tau=0.99,
        gamma=gamma,
        train_freq=(4, "step"),
        target_update_interval=10_000,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=50_000,
        verbose=0,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    eval_env = make_eval_env()
    model.learn(TOTAL_FRAMES_TUNE, progress_bar=False)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=7, deterministic=True)
    envs.close()
    return mean_reward


def trainDQN_Optuna():
    log_dir = "../logs/tensorboard/"
    log_name = "dqn_optuna_carracing_better"
    os.makedirs(log_dir, exist_ok=True)

    num_envs = 8
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    envs = VecMonitor(envs)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    study.optimize(objective, n_trials=10, n_jobs=1)

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

    model = DQN(**BEST_PARAMS)

    model.learn(total_timesteps=2_000_000, tb_log_name=log_name)

    model.save("../models/dqn_optuna/dqn_optuna_carracing_better")

    envs.close()


if __name__ == "__main__":
    trainDQN_Optuna()
