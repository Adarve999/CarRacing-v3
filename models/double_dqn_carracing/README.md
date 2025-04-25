# Double DQN para CarRacing-v3

Este proyecto implementa el algoritmo **Double Deep Q-Network (Double DQN)** para entrenar un agente en el entorno "CarRacing-v3" de Gymnasium, donde un coche autónomo debe aprender a seguir una pista. El código sigue los conceptos teóricos de Reinforcement Learning (RL) presentados en el material proporcionado, combinando técnicas como Experience Replay, Double Q-Learning y redes neuronales profundas.

---

## Características Clave
- **Double Q-Learning**: Mitiga la sobreestimación de valores Q al separar la selección y evaluación de acciones.
- **Experience Replay**: Almacena y muestrea experiencias pasadas para estabilizar el entrenamiento.
- **Redes Objetivo**: Actualizadas periódicamente para reducir la inestabilidad en el aprendizaje.
- **Exploración ε-greedy**: Balancea exploración y explotación con decaimiento exponencial de ε.
- **Arquitectura CNN**: Procesa estados de imagen (96x96 píxeles) con capas convolucionales.

---

## Estructura del Código
### Componentes Principales
1. **`ReplayBuffer`**:  
   - Almacena experiencias `(state, action, reward, next_state, done)`.
   - Muestrea lotes aleatorios para romper correlaciones temporales.

2. **`DQNNet` (Red Neuronal)**:  
   - **Capas convolucionales**: Extraen características de imágenes (3 canales RGB → 32 → 64 → 64).
   - **Capas lineales**: Predicen valores Q para acciones discretas.
   - Normaliza píxeles a `[0, 1]` si es necesario.

3. **`select_action`**:  
   - Política ε-greedy: Aleatoria (exploración) o basada en Q-values (explotación).

4. **`compute_td_loss`**:  
   - Calcula la pérdida TD usando la red objetivo (`target_net`) para evaluar acciones seleccionadas por la red política (`policy_net`).

5. **`train_double_dqn`**:  
   - Entrena el modelo con Experience Replay, actualiza la red objetivo cada `target_update_freq` pasos, y guarda el modelo entrenado.

6. **`evaluate_double_dqn`**:  
   - Evalúa el modelo entrenado sin exploración (solo explotación).

---

## Relación con la Teoría (Diapositivas)
| Componente          | Concepto Teórico                          | Diapositivas |
|---------------------|-------------------------------------------|--------------|
| **Double Q-Learning** | Separa selección y evaluación de acciones | 64-74, 91-92 |
| **Experience Replay** | Muestreo de experiencias pasadas          | 85-86        |
| **Redes Objetivo**  | Actualización periódica para estabilidad  | 88-89        |
| **ε-greedy**        | Exploración vs. explotación               | 58-59        |
| **CNN**             | Procesamiento de estados complejos (imágenes) | 81-84    |

---

## Entrenamiento y Evaluación
### Requisitos
```bash
pip install gymnasium numpy torch tqdm