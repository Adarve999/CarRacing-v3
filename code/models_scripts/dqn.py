import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
gym.register_envs(ale_py)

# Asegúrate de tener instalado:
# pip install gymnasium[atari,accept-rom-license] stable-baselines3


def mainDQN():
    # 1) Crea el entorno MsPacman con renderizado RGB
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

    # 2) Define y construye el modelo DQN con hiperparámetros
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=200_000,
        learning_starts=80_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=4_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.12,
        verbose=1,
        device="auto",
    )

    # 3) Entrena el modelo
    model.learn(
        total_timesteps=5_000_000,
        progress_bar=True
    )

    # 4) Guarda los pesos entrenados
    model.save("../models/dqn/dqn_msPacman")

    # 5) Cierra el entorno
    env.close()


if __name__ == "__main__":
    mainDQN()
