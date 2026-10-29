"""
Parámetros compartidos.

"""
import sys
from pathlib import Path

# Raiz del repo /deustorl para hacer el import necesario
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- Entorno ---
ENV_ID = "LunarLander-v3"
SEED = 47

# Parametros explicitos de Lunar Lander 
ENV_KWARGS = dict(continuous=False, gravity=-10.0, enable_wind=False)

# --- límites de observacion ---
# x, y, vx, vy, ángulo, vel. ángular, pata1, pata2
OBS_LOW = [-1.2, -0.1, -1.5, -1.5, -1.0, -1.5, 0.0, 0.0]
OBS_HIGH = [1.2, 1.5, 1.5, 1.5, 1.0, 1.5, 1.0, 1.0]

# Presupuestos de discretización: n_bins por dimensión 
BUDGETS = {
    "low" : 3,     # 3^8 = 6.561 estados 
    "medium" : 5,  # 5^8 = 390.625 estados 
    "high" : 7,    # 7^8 = 5.764.801 estados 
}

ALGOS = ["sarsa", "esarsa", "qlearning"]

# Mismo presupuesto de entrenamiento en todos los trials
N_TRIALS = 30
TRAIN_STEPS = 500_000
EVAL_EPISODES = 50

# Espacio de busqueda Optuna 
SEARCH_SPACE = {
    "learning_rate": dict(low=1e-3, high=0.5, log=True),
    "learning_rate_decay": dict(low=0.8, high=1.0, step=0.02),
    "lr_episodes_decay": [50, 100, 500, 1000],
    "discount_rate": dict(low=0.9, high=1.0, step=0.01),
    "epsilon": dict(low=0.0, high=0.5, step=0.05),
}

# Carpetas de salida
OPTUNA_DIR = Path(__file__).resolve().parent / "optuna"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# --- Fase B: entrenamientos largos ---
# Mismo presupuesto de pasos para las 9 configuraciones.
# (más alto que TRAIN_STEPS de la Fase A porque aquí se busca ver la convergencia).
PHASE_B_TRAIN_STEPS = 2_000_000
PHASE_B_SEEDS = [0, 1, 2]
PHASE_B_EVAL_EPISODES = 50  # usado en la evaluación de Mauro
QTABLE_DIR = RESULTS_DIR / "qtables"
# "tensorboard-logs" versionado en git
PHASE_B_LOG_DIR = Path(__file__).resolve().parent / "tensorboard-logs" / "phase_b"

def study_name(algo, budget):
    return f"{algo}_{budget}"

# Cada persona guarda su porpia BD
def db_url(owner):
    return f"sqlite:///{(OPTUNA_DIR / f'phase_a_{owner}.db').as_posix()}"     

