# Actividad 1 – RL: SARSA, Expected SARSA y Q-Learning en LunarLander

**Fase A:** optimización de hiperparámetros con Optuna para 3 algoritmos × 3 presupuestos de
discretización, sobre `LunarLander-v3` (Gymnasium, Box2D), usando los `DiscretizationWrappers` de `deustorl/`.

Equipo: Alberto (A.1), Xabier (A.2). Reparto de tareas y estado: ver `ESTADO_EQUIPO.md`.

## Estructura
```
deustorl/         librería de la asignatura (no se modifica)
Actividad_1/
  config.py         parámetros compartidos (entorno, bins, rangos de Optuna, pasos, trials)
  lunar_common.py   crea el entorno discretizado y entrena/evalúa una configuración
  optuna_study.py   un estudio de Optuna (1 algoritmo + 1 presupuesto)
  run_phase_a_1.py  lanza los estudios de A.1 (Alberto)
  optuna/           bases de datos de Optuna
  results/          mejores configuraciones y gráficas
ESTADO_EQUIPO.md    reparto de tareas y estado del equipo
```

## Instalación
Python 3.12 recomendado.
```bash
conda create -n rl python=3.12
conda activate rl
pip install -r requirements.txt
```
Si falla Box2D en Windows: `conda install -c conda-forge box2d-py`.

Comprobación:
```bash
cd Actividad_1
python lunar_common.py
```
Debe imprimir `(reward medio, pasos medios)` en unos segundos.

## Configuración
Todo está en `Actividad_1/config.py`. Ambos miembros deben usar los mismos valores.

| Presupuesto | bins por dimensión | Estados | Acciones |
|---|---|---|---|
| low | 3 | 6.561 | 4 |
| medium | 5 | 390.625 | 4 |
| high | 7 | 5.764.801 | 4 |

- 30 trials por estudio, 500.000 pasos por trial, 50 episodios de evaluación, semilla 47.
- Hiperparámetros explorados: learning rate, decaimiento del lr, episodios de decaimiento, factor de descuento y epsilon (rangos en `SEARCH_SPACE`).

### Justificación
- **Bins (3/5/7):** ≈ 60× más estados en cada nivel; con 8 bins la tabla Q ocuparía ≈ 540 MB.
- **30 trials:** el mínimo exigido es 20; TPE usa 10 trials aleatorios iniciales, así quedan 20 guiados.
- **500.000 pasos:** ≈ 14 s por trial; los 9 estudios caben en ≈ 1 h.
- **50 episodios de evaluación:** igual que la evaluación final pedida por la actividad.
- **Semilla 47:** la usada en `experiment_2.10.py`.
- **Hiperparámetros:** los de `experiment_2.10.py`, con rangos adaptados a LunarLander.

## Cómo ejecutar
Desde `Actividad_1/`:
```bash
python run_phase_a_1.py      # A.1: sarsa (low, medium, high), qlearning (low, medium)
```
Cada estudio tarda unos 7 minutos. Se puede interrumpir con `Ctrl+C` y relanzar: continúa donde se quedó.

Un estudio suelto:
```bash
python optuna_study.py --algo sarsa --budget low --owner 1
```

## Resultados
- `optuna/phase_a_1.db`: un estudio por algoritmo y presupuesto (30 trials).
- `results/best_<algoritmo>_<presupuesto>.json`: la mejor configuración de cada estudio.

## Fase B: entrenamientos largos (Katherine)
```bash
cd Actividad_1
python run_phase_b.py train    # 27 entrenamientos (9 configuraciones x 3 semillas)
python run_phase_b.py curves   # curvas de aprendizaje (media +/- desv. estándar) en results/figures/
```
- `qtable_io.py`: guardar/cargar la tabla Q con sus metadatos (algoritmo, presupuesto, semilla, hiperparámetros).
- `results/qtables/`: las 27 tablas Q (`.npy`) con sus metadatos (`.json`). **La carpeta no se versiona**
  (los `.npy` de presupuesto `high` pesan ~176 MB cada uno, por encima del límite de 100 MB de GitHub).
  Está comprimida entera en `results/qtables.zip` (~4.6 MB, porque la mayoría de estados nunca se visitan
  y quedan en 0).

  **Tras clonar el repo, descomprime ese archivo para regenerar la carpeta** (recrea
  `Actividad_1/results/qtables/` con los 27 `.npy` + `.json`, el zip ya trae esa estructura dentro):
  ```bash
  cd Actividad_1/results
  unzip qtables.zip
  ```
  Esto recrea `results/qtables/*.npy` junto a los `.json` ya versionados.
- `tensorboard-logs/phase_b/`: registros de TensorBoard de los 27 entrenamientos (sí se versionan; se llama así,
  en vez de `logs/`, para no caer en el `.gitignore`).
