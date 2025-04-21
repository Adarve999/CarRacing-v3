import gymnasium as gym
import ale_py
from stable_baselines3 import A2C
gym.register_envs(ale_py)

# Asegúrate de tener instalado:
# pip install gymnasium[atari,accept-rom-license] stable-baselines3


def mainA2C():
    # 1) Crea el entorno MsPacman con renderizado RGB
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

    # 2) Define y construye el modelo DQN con hiperparámetros
    model = A2C(
        policy="CnnPolicy",
        env=env,
        gamma=0.99,
        n_steps=1024,  # Number of steps to run for each environment per update
        vf_coef=0.25,  # Value function coefficient in the loss calculation
        ent_coef=0.01,  # Entropy coefficient for the loss calculation
        max_grad_norm=0.5,  # The maximum value for the gradient clipping
        learning_rate=7e-4,  # The learning rate
        rms_prop_eps=1e-5,  # RMSProp epsilon (default varies)
        use_rms_prop=True,  # Whether to use RMSprop (default) or Adam
        verbose=1,
        device="auto"  # Use GPU if available
    )

    # 3) Entrena el modelo
    model.learn(
        total_timesteps=5_000_000,
        progress_bar=True
    )

    # 4) Guarda los pesos entrenados
    model.save("../models/a2c/a2c_msPacman")

    # 5) Cierra el entorno
    env.close()


if __name__ == "__main__":
    mainA2C()
