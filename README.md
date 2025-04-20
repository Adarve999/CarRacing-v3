## 1  Definición del contexto, objetivos y métodos de RL (1 pto)

**Contexto:**  
El entorno **MsPacmanNoFrameskip‑v4** forma parte del **Arcade Learning Environment** (ALE), un banco de pruebas estándar para agentes de Atari 2600 que ofrece observaciones visuales de 210 × 160 × 3 px y un espacio de acciones discreto (9 acciones). Tras aplicar los *wrappers* de DeepMind (frame‐skip=4, reducción a 84×84 en escala de grises y *frame stacking* de 4), la observación final es un tensor de forma (4, 84, 84) listo para la red.

**Objetivos:**  
1. Entrenar desde cero agentes (**Value‑based**: DQN, QR‑DQN; **Policy‑gradient**: PPO, A2C) capaces de superar ampliamente una política aleatoria.  
2. Comparar técnicas **value‑based** vs **policy‑gradient** en términos de velocidad de convergencia, estabilidad y rendimiento final.  
3. Analizar la influencia de:  
   - *Wrappers* Atari (frame‑skip, grises, *frame stack*)  
   - Estrategias de paralelismo (**8 entornos PPO** vs **4 entornos A2C**)  
4. Generar evidencia empírica: métricas cuantitativas (recompensa media ± σ), curvas de aprendizaje y muestras de vídeo.

**Métodos y justificación:**  
- **QR‑DQN (Quantile Regression DQN):** Extiende DQN modelando la distribución completa de retornos mediante regresión cuantílica, lo que mejora la estabilidad y la explotación-exploración.  
- **DQN clásico:** Punto de partida histórico en Atari; usa replay buffer y ε‑greedy para estimar la función Q.  
- **PPO (Proximal Policy Optimization):** Algoritmo on‑policy estable que emplea clipping para limitar actualizaciones de política, ideal para paralelizar entornos.  
- **A2C (Advantage Actor‑Critic):** Variante síncrona de A3C, ejecuta múltiples workers sin replay buffer; menor consumo de memoria pero mayor varianza.  

---

## 2  Cantidad, profundidad y calidad del trabajo (2 ptos)

- **Preparación del entorno:**  
  - Instalación automática de ROMs y registro de ALE.  
  - Diseño de *wrappers* DeepMind (frame‑skip 4, 84×84 grises, *frame stack* 4) con `make_atari_env`.  
- **Entrenamiento:**  
  - **DQN**: 5 M pasos · buffer 100 k · target‑update 10 k · ε‑decay (1→0.1).  
  - **QR‑DQN**: 20 M – 50 M pasos para beneficiarse de distribución descontada.  
  - **PPO**: 5 M pasos · 8 envs · n_steps 128 · batch 256 · clip 0.1.  
  - **A2C**: 5 M pasos · 4 envs · n_steps 5 (por defecto).  
- **Herramientas:** TensorBoard, `EvalCallback`, `CheckpointCallback`, generación de vídeo (render_mode="human"), seeds controladas.  
- **Documentación:** README comentado, justificación de hiperparámetros y protocolos de evaluación.

---

## 3  Número de técnicas empleadas y su idoneidad (2,5 ptos)

| Familia             | Algoritmo       | Razón de elección                                                                                                                        | Implementación                                   |
|---------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| **Value‑based**     | DQN             | Referencia histórica en Atari; evalúa valor Q(s,a) y explora con ε‑greedy.                                          | Stable‑Baselines3 `DQN`                           |
| **Value‑based**     | QR‑DQN          | Modela la distribución de retornos; mejora la convergencia y la estabilidad.                                         | SB3‑contrib `QRDQN`                               |
| **Policy‑gradient** | PPO             | On‑policy estable con clipping; paralelizable en múltiples entornos.                                               | Stable‑Baselines3 `PPO`                           |
| **Actor‑Crítico**   | A2C             | Síncrono, determinista, menor consumo de memoria; compara actualizaciones frecuentes vs PPO.                         | Stable‑Baselines3 `A2C`                           |

---

## 4  Interpretación de resultados y pruebas (1,5 ptos)

- **Recompensa media (10 episodios, determinista):**  
  - **QR‑DQN** ≈ 1486 ± 126  
  - **DQN** ≈ 1349 ± 906
  - **DQN + Optuna** ≈ 494 ± 125 (variabilidad de hiperparámetros)
  - **PPO** ≈ 1111 ± 128
  - **A2C** ≈ 74 ± 12 (alta varianza y on‑policy ineficiente)
- **Curvas de aprendizaje:**  
  - DQN clásico empieza a aprender tras ~1 M pasos, pero se estanca y presenta gran varianza.  
  - QR‑DQN reduce varianza y acelera convergencia gracias a la señal de distribución.  
  - PPO presenta un despegue más rápido (primeros 500 k pasos) y se estabiliza en ~1 000 pts.  
  - A2C no alcanza puntuaciones competitivas en Ms Pac‑Man, debido a su naturaleza on‑policy y pocas muestras por paso.

---

## 5  Comparativa de métodos (1,5 ptos)

| Métrica                  | QR‑DQN        | DQN            | DQN + Optuna   | PPO            | A2C            | Comentario                                                                                                 |
|--------------------------|---------------|----------------|----------------|----------------|----------------|------------------------------------------------------------------------------------------------------------|
| **Recompensa media (±σ)**| 1486 ± 126    | 1349 ± 906     | 494 ± 125      | 1111 ± 128     | 74 ± 12        | Sólo QR‑DQN modela distribuciones completas, reduciendo varianza y mejorando promedio.   |
| **Velocidad de aprendizaje** | Media‑alta    | Baja‑media     | Muy baja       | Alta           | Media‑baja     | PPO aprovecha 8 envs; DQN requirió mayor presupuesto de frames (≥ 200 M frames).       |
| **Estabilidad (σ)**      | Baja          | Alta           | Media          | Media‑baja     | Alta‑muy alta  | A2C es estable pero con techo muy bajo; DQN muestra oscilaciones fuertes.              |
| **Uso de memoria**       | Medio         | Alto (~3 GB)   | Alto           | Bajo           | Muy bajo       | A2C no usa replay buffer; PPO usa colección pequeña de rollouts; DQN requiere buffer grande.               |
| **Capacidad muestras**   | Excelente     | Buena          | Mala           | Regular        | Baja           | Off‑policy (DQN/QR‑DQN) reutiliza muestras; on‑policy (PPO/A2C) desaprovecha pasos anteriores.              |

> **Conclusión:** Para **producción** con recursos limitados, **PPO** es la opción más práctica (rápido, estable). Para **rendimiento máximo** en Ms Pac‑Man, **QR‑DQN** ofrece el mejor balance de promedio y varianza. **A2C** resulta demasiado ineficiente para este entorno.

---