# Estado del equipo - Actividad 1 RL
Última actualización: 2026-10-26 por Alberto

## Equipo
- Fase A: Alberto (A.1), Xabier (A.2)

## Resumen de la Fase A
| | A.1 Alberto | A.2 Xabier |
|---|---|---|
| Estudios de Optuna | sarsa ×3, qlearning low y medium (5) | esarsa ×3, qlearning high (4) |
| Base de datos | `optuna/phase_a_1.db` | `optuna/phase_a_2.db` |
| Código | `config.py`, `lunar_common.py`, `optuna_study.py`, `run_phase_a_1.py` | `run_phase_a_2.py`, `merge_dbs.py`, `analyze_phase_a.py` |
| Informe | 2 Entorno, 3a Discretización, 3c/3d de sus estudios | 3b Hiperparámetros, 3c/3d de sus estudios, tabla de las 9 configuraciones |
| Análisis | Efecto de la discretización en SARSA y Q-Learning | Efecto en Expected SARSA y comparación global |

## Los 9 estudios
Cada estudio: 30 trials de 500.000 pasos. Estado: ⬜ pendiente · 🟨 en curso · ✅ hecho

| Algoritmo | low (3 bins, 6.561 estados) | medium (5 bins, 390.625) | high (7 bins, 5.764.801) |
|---|---|---|---|
| SARSA | A.1 ⬜ | A.1 ⬜ | A.1 ⬜ |
| Expected SARSA | A.2 ⬜ | A.2 ⬜ | A.2 ⬜ |
| Q-Learning | A.1 ⬜ | A.1 ⬜ | A.2 ⬜ |

## Tareas
### Entre los dos
- [ ] Revisar y aceptar `config.py` (bins, límites, rangos, pasos, semilla)
- [x] Repartir los 9 estudios
- [ ] Decidir si se amplía el rango de `learning_rate` (en pruebas llegó a 0,46 de un máximo de 0,5)
- [ ] Intercambiar las bases de datos y fusionar las ramas
- [ ] Revisión del código y del texto

### A.1 Alberto
- [x] `config.py`
- [x] `lunar_common.py` (probado)
- [x] `optuna_study.py` (probado con 2 trials)
- [x] `run_phase_a_1.py`
- [x] `requirements.txt`, `.gitignore`, `README.md`, este archivo (en curso)
- [ ] Ejecutar sus 5 estudios (≈ 35 min)
- [ ] Informe: 2 Entorno y 3a Discretización
- [ ] Informe: 3c y 3d de sus estudios
- [ ] Análisis: discretización en SARSA y Q-Learning
- [ ] Contribución individual

### A.2 Xabier
- [ ] Instalar el entorno y comprobar con `python lunar_common.py`
- [ ] `run_phase_a_2.py` (copia del de Alberto con `OWNER = "2"`)
- [ ] Ejecutar sus 4 estudios (≈ 30 min)
- [ ] `merge_dbs.py` (une las dos bases en `phase_a.db`)
- [ ] `analyze_phase_a.py` (tabla de las 9 configuraciones y gráficas)
- [ ] Informe: 3b Hiperparámetros
- [ ] Informe: 3c y 3d de sus estudios y tabla de las 9
- [ ] Análisis: Expected SARSA y comparación global
- [ ] Contribución individual

## Cómo se conectan los archivos
```mermaid
flowchart LR
    C[config.py] --> L[lunar_common.py] --> O[optuna_study.py]
    O --> R1[run_phase_a_1.py<br/>Alberto]
    O --> R2[run_phase_a_2.py<br/>Xabier]
    R1 --> D1[(phase_a_1.db)]
    R2 --> D2[(phase_a_2.db)]
    D1 --> M[merge_dbs.py<br/>Xabier]
    D2 --> M
    M --> D[(phase_a.db)]
    D --> A[analyze_phase_a.py<br/>Xabier]
    A --> Z[results: 9 configuraciones y gráficas]
```

## Entorno
- Gymnasium versión: 1.3.0   Entorno: LunarLander-v3
- Parámetros del entorno: continuous=False, gravity=-10.0, enable_wind=False
- Pasos/segundo medidos: ≈ 36.000 (≈ 14 s por trial de 500.000 pasos, MacBook de Alberto)

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
- Bins 3/5/7: ≈ 60× más estados en cada nivel. Con 8 bins serían 16,7 M estados (≈ 540 MB de tabla Q).
- Límites ajustados a los rangos típicos de LunarLander; los de Gymnasium por defecto (±10 en velocidades) dejaban casi todos los bins vacíos. Pendiente de comprobar con datos medidos.

## Optuna
- Pasos por trial: 500.000   Episodios de evaluación: 50   Semilla: 47   Trials por estudio: 30
- Justificación: 30 trials (mínimo 20; TPE usa 10 aleatorios iniciales, quedan 20 guiados) · 500.000 pasos (≈ 14 s por trial; los 9 estudios ≈ 1 h) · 50 episodios (igual que la evaluación final de la actividad) · semilla 47 (la de experiment_2.10.py).
- Hiperparámetros: learning rate, decaimiento del lr, episodios de decaimiento, factor de descuento y epsilon (rangos en `SEARCH_SPACE` de `config.py`).

## 9 configuraciones elegidas
(pegar tabla de results/best_configs.md cuando estén los 9 estudios)
