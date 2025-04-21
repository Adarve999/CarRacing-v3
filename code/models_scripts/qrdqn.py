import gymnasium as gym
import ale_py
from sb3_contrib.qrdqn import QRDQN
gym.register_envs(ale_py)

# Asegúrate de tener instalado:
# pip install gymnasium[atari,accept-rom-license] stable-baselines3


def mainQRDQN():
    # 1) Crea el entorno MsPacman con renderizado RGB
    env = gym.make("ALE/MsPacman-v5")


    # 2) Define y construye el modelo DQN con hiperparámetros
    policy_kwargs = dict(
    n_quantiles=200,
    normalize_images=False
    )
    
    model = QRDQN(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        buffer_size=200_000,
        learning_starts=80_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.12,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto",
    )

    # 3) Entrena el modelo
    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True
    )

    # 4) Guarda los pesos entrenados
    model.save("../models/qrdqn/qrdqn_msPacman")

    # 5) Cierra el entorno
    env.close()


if __name__ == "__main__":
    mainQRDQN()
