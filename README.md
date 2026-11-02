# Actividad 1 – RL: SARSA, Expected SARSA y Q-Learning en LunarLander

**Fase A:** optimización de hiperparámetros con Optuna para 3 algoritmos × 3 presupuestos de
discretización, sobre `LunarLander-v3` (Gymnasium, Box2D), usando los `DiscretizationWrappers` de `deustorl/`.

Equipo: Alberto (A.1), Xabier (A.2), Katherine (B.1) y Mauro (B.2).
Reparto de tareas y estado: ver `ESTADO_EQUIPO.md`.

## Estructura
```
deustorl/         librería de la asignatura (no se modifica)
Actividad_1/
  config.py         parámetros compartidos (entorno, bins, rangos de Optuna, pasos, trials)
  lunar_common.py   crea el entorno discretizado y entrena/evalúa una configuración
  optuna_study.py   un estudio de Optuna (1 algoritmo + 1 presupuesto)
  run_phase_a_1.py  lanza los estudios de A.1 (Alberto)
  run_phase_b.py    entrenamientos largos y curvas de aprendizaje (Katherine)
  qtable_io.py      guarda y carga las tablas Q y sus metadatos
  run_phase_b_2.py  evalúa las tablas Q y selecciona las tres políticas finales (Mauro)
  run_phase_b_2.1.py inspección visual de las políticas finales (Mauro)
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

## Fase B: entrenamientos largos, evaluación y políticas finales

La Fase B parte de las nueve configuraciones seleccionadas en la Fase A
(`results/best_<algoritmo>_<presupuesto>.json`). Katherine entrena y genera las curvas;
Mauro evalúa las tablas guardadas, selecciona una política por discretización y observa
su comportamiento en el entorno.

Los comandos de esta sección se ejecutan desde `Actividad_1/`, con el entorno `rl` activado:

```bash
cd Actividad_1
```

### B.1. Entrenamientos y curvas de aprendizaje (Katherine)

```bash
python run_phase_b.py train
python run_phase_b.py curves
python run_phase_b.py discretization-curves
```

- **`train`:** realiza 27 entrenamientos (9 configuraciones × semillas 0, 1 y 2),
  con 2.000.000 de pasos cada uno. Guarda las tablas Q en `results/qtables/` y los
  registros de TensorBoard en `tensorboard-logs/phase_b/`.
- **`curves`:** genera nueve gráficas, una por algoritmo y discretización, en
  `results/figures/<algoritmo>_<presupuesto>_learning_curve.png`.
- **`discretization-curves`:** genera tres gráficas comparativas, una por discretización,
  con SARSA, Expected SARSA y Q-Learning juntos. Se guardan como
  `results/figures/discretizacion_low.png`, `discretizacion_medium.png` y
  `discretizacion_high.png`.

En las gráficas, cada línea representa la media de las ejecuciones de una configuración;
la sombra muestra una desviación estándar por encima y por debajo de esa media.
Los datos proceden de `train/reward`, que registra el promedio de las recompensas de
los últimos 100 episodios (o los disponibles al inicio). Las curvas individuales alinean
los registros en 200 puntos comunes; las comparativas agrupan los pasos en 150 tramos,
promedian los registros de cada ejecución dentro de cada tramo y después calculan la
media y la dispersión entre ejecuciones.

Los dos comandos de gráficas utilizan los registros existentes: no vuelven a entrenar.
Para entrenar únicamente una configuración, se puede usar:

```bash
python run_phase_b.py train --only qlearning_high
```

Los registros también se pueden abrir directamente en TensorBoard:

```bash
tensorboard --logdir tensorboard-logs/phase_b
```

### Recuperar las tablas Q después de clonar

`qtable_io.py` guarda cada tabla en un archivo `.npy` y sus metadatos en un `.json`
(algoritmo, discretización, semilla, hiperparámetros y número de estados y acciones).
La carpeta `results/qtables/` no se versiona: cada tabla de discretización alta ocupa
unos 176 MiB, por encima del límite de 100 MB de GitHub.

Para recuperar los 27 pares de archivos sin repetir los entrenamientos, descomprime
el archivo incluido en el repositorio. Desde `Actividad_1/`:

```bash
unzip results/qtables.zip -d results
```

El ZIP ya contiene la carpeta `qtables/`, por lo que este comando recrea
`Actividad_1/results/qtables/`. Los registros de `tensorboard-logs/phase_b/` sí están
versionados y permiten regenerar las gráficas.

### B.2. Evaluación y selección de las políticas finales (Mauro)

Una vez disponibles las tablas Q y sus metadatos en `results/qtables/`, ejecuta:

```bash
python run_phase_b_2.py
```

El script carga las tablas mediante `qtable_io.load_qtable` y evalúa cada una durante
**50 episodios**, usando la política greedy (`max_policy`). Antes de evaluar cada tabla,
fija la semilla 42 en `random`, NumPy y el entorno. Obtiene la recompensa media y el
número medio de pasos por episodio.

La consola muestra los resultados de cada tabla y un resumen por algoritmo y
discretización, incluyendo el promedio entre semillas y la mejor semilla. Para elegir
las políticas finales, selecciona **la tabla con mayor recompensa media de evaluación
de cada discretización**, entre los algoritmos y semillas evaluados. La selección se
basa en la mejor tabla individual, no en el promedio de las tres semillas de un algoritmo.

Las tres ganadoras se guardan, con sus metadatos, en estas subcarpetas:

```text
results/final_policies/
  best_policy_low/
  best_policy_medium/
  best_policy_high/
```

Cada subcarpeta contiene un `.npy` y un `.json` con el nombre
`<algoritmo>_<presupuesto>_seed<semilla>`. El resumen numérico de la evaluación se
imprime en consola; el script no genera una tabla CSV de resultados.

También se incluyen las políticas ya seleccionadas en `results/final_policies.zip`.
Para recuperarlas directamente sin repetir la evaluación, desde `Actividad_1/`:

```bash
unzip results/final_policies.zip -d results
```

### B.2.1. Inspección visual de las políticas finales (Mauro)

Con las tres políticas disponibles en `results/final_policies/`, ejecuta:

```bash
python run_phase_b_2.1.py
```

El script carga la ganadora de cada discretización y muestra su comportamiento en
LunarLander mediante Pygame. En el bloque final del archivo, `RENDER = True` activa
la ventana y `EPISODES = 10` establece diez episodios por política. Para ejecutar la
demostración solo en consola, cambia `RENDER` a `False`. Estas opciones se modifican
en el archivo; no son argumentos de línea de comandos.

Durante la inspección se pueden describir el control de la inclinación, las correcciones
de trayectoria y la forma de aterrizar o fallar. La consola también muestra la recompensa
y los pasos medios de esta demostración. Estos diez episodios sirven para observar
el comportamiento; la selección de ganadoras corresponde a la evaluación de 50 episodios
del paso anterior. Las curvas de aprendizaje, por su parte, describen el entrenamiento.
