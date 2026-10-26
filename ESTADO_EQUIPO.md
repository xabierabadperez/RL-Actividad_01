# Estado del equipo - Actividad 1 RL
Última actualización: 2026-10-26 por Alberto

## Equipo
- Fase A: Alberto (A.1), Xabier (A.2)

## Entorno
- Gymnasium versión: 1.3.0   Entorno: LunarLander-v3
- Parámetros del entorno: continuous=False, gravity=-10.0, enable_wind=False
- Pasos/segundo medidos: ≈ 36.000 (≈ 14 s por trial de 500.000 pasos)

## Discretización
Límites de observación (los mismos en los tres niveles), en orden x, y, vx, vy, ángulo, vel. angular, pata1, pata2:
- low (mínimos): [-1.2, -0.1, -1.5, -1.5, -1.0, -1.5, 0.0, 0.0]
- high (máximos): [1.2, 1.5, 1.5, 1.5, 1.0, 1.5, 1.0, 1.0]

| Nivel | n_bins | n_estados | Acciones | Memoria tabla Q |
|---|---|---|---|---|
| low | 3 | 6.561 | 4 | ≈ 0,2 MB |
| medium | 5 | 390.625 | 4 | ≈ 12,5 MB |
| high | 7 | 5.764.801 | 4 | ≈ 184 MB |

Justificación:
- Bins 3/5/7: ≈ 60× más estados en cada nivel. Con 8 bins serían 16,7 M estados (≈ 540 MB de tabla Q), demasiado para ejecutar varios entrenamientos.
- Límites ajustados a los rangos típicos de LunarLander; los de Gymnasium por defecto (±10 en velocidades) dejaban casi todos los bins vacíos. Pendiente de comprobar con datos medidos.

## Optuna
- Pasos por trial: 500.000   Episodios de evaluación: 50   Semilla: 47   Trials por estudio: 30
- Justificación: 30 trials (mínimo 20; TPE usa 10 aleatorios iniciales, quedan 20 guiados) · 500.000 pasos (≈ 14 s por trial; los 9 estudios ≈ 1 h) · 50 episodios (igual que la evaluación final de la actividad) · semilla 47 (la de experiment_2.10.py).
- Hiperparámetros: learning rate, decaimiento del lr, episodios de decaimiento, factor de descuento y epsilon (rangos en `SEARCH_SPACE` de `config.py`).
- Reparto de estudios:
  - A.1 Alberto: sarsa (low, medium, high), qlearning (low, medium)
  - A.2 Xabier: esarsa (low, medium, high), qlearning (high)
- Estado de los 9 estudios: A.1 pendiente de lanzar · A.2 pendiente

## 9 configuraciones elegidas
(pegar tabla de results/best_configs.md cuando estén los 9 estudios)

## Decisiones pendientes o consultas al profesor
- Confirmar que un grupo de dos personas es válido (el enunciado dice 3-4).
- Decidir si se amplía el rango de `learning_rate` por encima de 0,5.
- Comprobar los límites de observación con datos medidos (percentiles 1 y 99).
