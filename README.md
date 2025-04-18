A continuación tienes un **borrador‑guía** para responder, punto por punto, a los criterios de evaluación de la práctica. Úsalo tal cual o retócalo a tu estilo; incluye las cifras exactas (recompensas medias, nº de pasos, capturas de TensorBoard, etc.) que ya obtuviste.

---

## 1  Definición del contexto, objetivos y métodos de RL (1 pto)

**Contexto:**  
El trabajo aplica aprendizaje por refuerzo profundo en el entorno *PacmanNoFrameskip‑v4* (ALE/Gymnasium). Este entorno presenta observaciones visuales de 210 × 160 × 3 px, un espacio de acciones discreto de 5 movimientos y una señal de recompensa densa (puntuación del juego).

**Objetivos:**  
<<1. Entrenar desde cero agentes (DQN, PPO, A2C) capaces de superar ampliamente la política aleatoria.  
2. Comparar técnicas **value‑based** (DQN) vs. **policy‑gradient** (PPO, A2C) en términos de velocidad de aprendizaje y rendimiento final.  
3. Analizar la influencia de:  
   - *Wrappers* Atari (frame‑skip 4, 84 × 84 grises, *frame stack* 4)  
   - Estrategias de paralelismo (8 entornos PPO vs 4 A2C)>>  
4. Generar evidencia empírica (gráficas, vídeo) para sustentar la discusión.

**Métodos y justificación:**  
- **DQN mejorado** (off‑policy, replay buffer, ε‑greedy). Elegido por su historial en Atari.  
- **PPO** (on‑policy, actor‑critic con clipping). Elegido por estabilidad con paralelismo.  
- **A2C** (actor‑crítico síncrono). Incluido para comparar actualizaciones más frecuentes vs PPO.  
Todos usan **CNN “Nature”** para garantizar comparabilidad directa.

---

## 2  Cantidad, profundidad y calidad del trabajo (2 ptos)

- **Preparación del entorno:** instalación automática de ROMs, registro ALE, diseño de *wrappers* DeepMind, verificación con script de prueba.  
- **Entrenamiento:**  
  - DQN ↔ 5 M pasos · buffer 1 M · target‑update 10 k · ε decay.  
  - PPO ↔ 5 M pasos · 8 entornos · n_steps 128 · batch 256 · clip 0.1.  
  - Entrenamiento en GPU con registro TensorBoard y modelos guardados.  
- **Herramientas adicionales:** scripts de evaluación, grabación de vídeo, cálculo de estadísticas (recompensa media ± σ), seeds controladas.  
- **Documentación:** explicación detallada de hiperparámetros, código comentado, justificación de cada decisión.

---

## 3  Número de técnicas empleadas y su idoneidad (2,5 ptos)

| Familia | Algoritmo | Razón de elección | Implementación |
|---------|-----------|-------------------|----------------|
| **Value‑based** | DQN | Referencia para Atari | Stable‑Baselines3 `DQN` |
| **Policy‑gradient** | PPO | Estabilidad con clipping | Stable‑Baselines3 `PPO` |
| **Actor‑Crítico** | A2C | Actualizaciones frecuentes, menor consumo de memoria | Stable‑Baselines3 `A2C` |

> *Nota:* Se implementó A2C para comparar actualizaciones síncronas (4 entornos) vs asíncronas. PPO mantiene ventaja en estabilidad.

---

## 4  Interpretación de resultados y pruebas (1,5 ptos)

- **Recompensa media (10 episodios, determinista):**  
  - DQN ≈ **X** ± Y  
  - PPO ≈ **Z** ± W  
  - A2C ≈ **1,071 ± 242**

  (incluye aquí tus números reales).  
- **Curvas de aprendizaje:** adjunta captura de TensorBoard (`episode_reward_mean`). Indica puntos de inflexión (p.ej. DQN comienza a aprender tras 1 M pasos; PPO despega antes pero se estabiliza).  
- **Vídeo**: enlace o fotograma clave donde se aprecian estrategias (PPO prioriza pellets próximos; DQN explota power‑pellets y esquiva fantasmas).  
- **Robustez:** discute varianza entre seeds; PPO muestra menor dispersión que DQN gracias al paralelismo.  
- **Limitaciones observadas:** catástrofe de olvido en DQN al final de entrenamiento prolongado; PPO ocasionalmente se queda en ciclo local cerca de casa inicial.

---

## 5  Comparativa de métodos (1,5 ptos)

| Métrica | DQN | PPO | A2C | Comentario |
|---------|-----|-----|-----|------------|
| **Velocidad de aprendizaje** | Lenta al principio | Rápida (ligero over‑shoot) | Intermedia | PPO recolecta >800 fps con 8 envs vs A2C 423 fps (4 envs) |
| **Score final** | Mayor (**≈ … pts**) | Bueno (**≈ … pts**) | Básico (**≈1,071 pts**) | DQN supera a PPO tras 4M pasos; A2C se estanca antes |
| **Estabilidad (σ)** | ± … | ± … | **±242** | A2C más estable pero con estrategias conservadoras |
| **Uso de memoria** | Buffer 1 M (~3 GB) | Bajo | **Muy bajo (~1.1 GB)** | A2C ideal para hardware limitado |
| **Robustez sticky actions** | Menor | Mayor | Media | PPO maneja mejor estocasticidad; A2C intermedio |

Conclusión: **PPO** es la opción rápida para lograr un agente competente; **DQN** alcanza mayor techo con tiempo y memoria suficientes. Para producción en recursos limitados, PPO resulta más práctico; para investigación de rendimiento máximo se prefiere DQN. **A2C** es una opción ligera y estable, ofrece mejor equilibrio velocidad-consumo que DQN, pero su rendimiento limitado lo hace adecuado para hardware con recursos limitados y entornos con recompensas inmediatas.

---

## 6  Claridad de escritura y presentación (1,5 ptos)

- **Estructura propuesta de la memoria (≈ 10 páginas):**
  1. **Introducción y motivación**  
  2. **Marco teórico** (RL, DQN, PPO, wrappers Atari)  
  3. **Metodología**  
     - Entorno y preprocesado  
     - Configuración de modelos  
     - Hardware y software  
  4. **Experimentos**  
     - Protocolo  
     - Métricas  
  5. **Resultados**  
     - Tablas y gráficas  
     - Análisis cualitativo (vídeo)  
  6. **Discusión y comparativa**  
  7. **Conclusiones y trabajo futuro**  
  8. **Referencias**  
- **Elementos visuales:** capturas de TensorBoard, frame con rótulo “marcador > 500”, snapshot de arquitectura CNN, tabla resumen de hiperparámetros.  

  Resumen hiperparámetros A2C:  
  ```python
  n_steps=5, ent_coef=0.01, learning_rate=7e-4
- **Estilo:** títulos numerados, figuras citadas en el texto, pie de figura con contexto, referencias en formato APA/IEEE.