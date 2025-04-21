import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
gym.register_envs(ale_py)

# Asegúrate de tener instalado:
# pip install gymnasium[atari,accept-rom-license] stable-baselines3


def mainPPO():
    # 1) Crea el entorno MsPacman con renderizado RGB
    env = gym.make("ALE/MsPacman-v5")

    # 2) Define y construye el modelo DQN con hiperparámetros
    model = PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=1024,             
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=2.5e-4, 
        ent_coef=0.01,
        clip_range=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda"
    )

    # 3) Entrena el modelo
    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True
    )

    # 4) Guarda los pesos entrenados
    model.save("../models/ppo/ppo_msPacman")

    # 5) Cierra el entorno
    env.close()


if __name__ == "__main__":
    mainPPO()
