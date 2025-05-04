<p align="center">
  <img src="images/dqn.gif" alt="CarRacing-v3 banner" width="50%"/>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python"></a>
  <a href="https://github.com/DLR-RM/stable-baselines3"><img alt="Stable-Baselines3" src="https://img.shields.io/badge/SB3-2.6.0-009688?logo=pytorch"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

# Conducción Autónoma en **CarRacing-v3** 🏎️ con Deep Reinforcement Learning

Este proyecto explora cómo distintos enfoques de *Deep Reinforcement Learning* resuelven la tarea de conducir un coche lo más rápido posible sin salirse de pista en **CarRacing-v3** (Gymnasium).
El entorno devuelve exclusivamente píxeles y un espacio de **5 acciones discretas**, de modo que cada algoritmo debe:

1. **Interpretar la escena visual** en tiempo real mediante una CNN.
2. **Decidir** cómo frenar, derrapar o enderezar el coche con tan solo 5 movimientos posibles.
3. **Generalizar** a nuevos circuitos generados aleatoriamente en cada episodio.

Para poner a prueba distintas filosofías de RL incluimos:

* **Double DQN** – aprendizaje basado en valores que reutiliza cada fotograma cientos de veces a través de un *replay buffer*.
* **PPO (CNN)** – gradiente de política con *clipping* que prima la estabilidad y el paralelismo masivo.
* **Recurrent PPO** – PPO + LSTM: añade memoria a corto plazo y mejora la trazada en curvas enlazadas.
* **TRPO** – actualizaciones dentro de una región de confianza (KL-constraint); ideal para estudiar convergencia sin saltos bruscos.

Todos los experimentos se entrenan durante **1 M de pasos** y guardan dos vueltas de demostración en MP4 para comparar estilos de conducción.

---

# Ejecución local desde cero

#### 1. Clona el repositorio

```bash
git clone https://github.com/Adarve999/CarRacing-v3.git
cd CarRacing-v3
```

#### 2. Crea y activa el entorno

```bash
conda env create -f requirements.yml
conda activate rl_env
```

#### 3. Entrena (ejemplo PPO)

```bash
python src/models_scripts/ppo.py
```

#### 4. Lanza TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

#### 5. Reproduce una vuelta grabada

```bash
python -m utils.show_videos videos/ppo --prefix ppo
```

---

# 📈 Conclusiones

| Algoritmo         | Recompensa media ± σ (10 episodios) | Observaciones (versión “no técnica”)                                                              |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Double DQN**    | **874 ± 33**                        | Repasa cada curva hasta bordarla y acaba firmando la mejor vuelta.       |
| **Recurrent PPO** | 830 ± 44                            | Con memoria a corto plazo, anticipa los derrapes y mantiene una trazada muy limpia.               |
| **PPO**           | 809 ± 109                           | El todoterreno: aprende rápido pero, sin memoria, a veces pierde la línea ideal en curvas largas. |
| **TRPO**          | 711 ± 191                           | Conduce con extrema prudencia: casi nunca se sale, pero tampoco bate tiempos.                     |

## ¿Por qué cada algoritmo rinde como rinde en *CarRacing-v3*?

* **Double DQN**
  Va guardando en su libreta millones de imágenes y repasándolas una y otra vez hasta pulir la maniobra exacta en cada curva.
  Como solo puede elegir entre 5 movimientos, le resulta fácil comparar y quedarse con el mejor.  Por eso acaba siendo el más rápido.

* **Recurrent PPO**
  Conduce “de oído”: además de ver la pista, recuerda lo que acaba de pasar gracias a una memoria interna.
  Ese recuerdo le permite prever derrapes y corregir la dirección con antelación, de ahí que se acerque mucho a DQN aunque necesite algo más de tiempo y de GPU.

* **PPO**
  Es la versión sin memoria.  Toma decisiones sólidas y aprende deprisa porque practica con ocho coches a la vez, pero sólo “ve” los últimos cuatro fotogramas; si la curva es larga puede reaccionar tarde y perder puntos.

* **TRPO**
  Conduce con mucha prudencia: se asegura de que cada cambio en su forma de pilotar sea pequeño para no estrellarse por sorpresa.
  Esa cautela hace la conducción muy estable, pero también le impide arriesgar cuando haría falta; por eso su puntuación se queda más baja.

---

# Authors

* **Rubén Adarve Pérez**
* **Javier Miranda**
* **Pablo de la Fuente**

Please use this bibtex if you want to cite this repository (main branch) in your publications:

```bibtex
@misc{CarRacingRL2025,
  author       = {Rubén Adarve Pérez and Javier Miranda and Pablo de la Fuente},
  title        = {Conducción autónoma en CarRacing-v3 con Deep RL},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Adarve999/CarRacing-v3}},
}
```
