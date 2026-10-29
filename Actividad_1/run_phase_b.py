"""
Katherine - Fase B: entrenamientos largos y curvas de aprendizaje.

Entrena las 9 configuraciones seleccionadas en la Fase A (results/best_<algo>_<budget>.json)
con config.PHASE_B_SEEDS semillas cada una (27 entrenamientos por defecto), guarda cada tabla Q
con sus metadatos (qtable_io.py) y registra la evolución en TensorBoard. Después, a partir de
esos registros, genera las curvas de aprendizaje (media +/- dispersión entre semillas).

Uso:
    python run_phase_b.py train                        # lanza los 27 entrenamientos
    python run_phase_b.py train --only qlearning_high   # solo una configuración (9 posibles)
    python run_phase_b.py curves                        # genera las gráficas en results/figures
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config as cfg
import deustorl.expected_sarsa as _esarsa_mod
import deustorl.qlearning as _qlearning_mod
import deustorl.sarsa as _sarsa_mod
from deustorl.common import EpsilonGreedyPolicy
from deustorl.helpers import TensorboardLogger as _TensorboardLogger
from lunar_common import ALGO_CLASSES, make_env
from qtable_io import save_qtable


def _patch_tensorboard(run_name):
    """Fuerza el nombre de carpeta de TensorboardLogger a run_name, para poder
    agrupar después los logs por configuración y semilla (ver generate_curves)."""

    class _NamedLogger(_TensorboardLogger):
        def __init__(self, algo_name="algo", episode_period=1, log_dir="logs"):
            super().__init__(algo_name=run_name, episode_period=episode_period, log_dir=str(cfg.PHASE_B_LOG_DIR))

    for module in (_sarsa_mod, _esarsa_mod, _qlearning_mod):
        module.TensorboardLogger = _NamedLogger


def load_best_configs():
    """Lee las 9 configuraciones seleccionadas en la Fase A."""
    configs = {}
    for algo in cfg.ALGOS:
        for budget in cfg.BUDGETS:
            path = cfg.RESULTS_DIR / f"best_{cfg.study_name(algo, budget)}.json"
            with open(path) as f:
                configs[(algo, budget)] = json.load(f)
    return configs


def train_one(algo, budget, params, seed):
    """Entrena una configuración con una semilla y devuelve la tabla Q entrenada."""
    _patch_tensorboard(cfg.study_name(algo, budget) + f"_seed{seed}")

    env = make_env(budget)
    env.reset(seed=seed)
    algo_obj = ALGO_CLASSES[algo](env)

    algo_obj.learn(
        EpsilonGreedyPolicy(epsilon=params["epsilon"]),
        n_steps=cfg.PHASE_B_TRAIN_STEPS,
        discount_rate=params["discount_rate"],
        lr=params["learning_rate"],
        lrdecay=params["learning_rate_decay"],
        n_episodes_decay=params["lr_episodes_decay"],
    )
    env.close()
    return algo_obj.q_table


def run_all(only=None):
    """Lanza los entrenamientos (filtrables con --only) y guarda cada tabla Q resultante."""
    configs = load_best_configs()
    for (algo, budget), best in configs.items():
        name = cfg.study_name(algo, budget)
        if only and name != only:
            continue
        for seed in cfg.PHASE_B_SEEDS:
            print(f"=== {name} / seed {seed} ===", flush=True)
            q_table = train_one(algo, budget, best["params"], seed)
            path = save_qtable(q_table, cfg.QTABLE_DIR, algo=algo, budget=budget, seed=seed, params=best["params"])
            print(f"Guardada {path}.npy / {path}.json")


def _load_scalar(log_dir, tag="train/reward"):
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def generate_curves(figures_dir=None):
    """Para cada algoritmo y presupuesto, lee los logs de TensorBoard de sus semillas
    y dibuja la curva de aprendizaje: media +/- desviación estándar entre semillas."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir or (cfg.RESULTS_DIR / "figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    for algo in cfg.ALGOS:
        for budget in cfg.BUDGETS:
            prefix = cfg.study_name(algo, budget) + "_seed"
            run_dirs = sorted(d for d in cfg.PHASE_B_LOG_DIR.glob(f"{prefix}*") if d.is_dir())
            if not run_dirs:
                print(f"Sin logs todavía para {algo}/{budget}, se omite.")
                continue

            runs = [_load_scalar(d) for d in run_dirs]
            max_common_step = min(steps[-1] for steps, _ in runs)
            grid = np.linspace(0, max_common_step, 200)
            aligned = np.array([np.interp(grid, steps, values) for steps, values in runs])

            mean = aligned.mean(axis=0)
            std = aligned.std(axis=0)

            plt.figure()
            plt.plot(grid, mean, label="media")
            plt.fill_between(grid, mean - std, mean + std, alpha=0.3, label="+/- 1 desv. estándar")
            plt.xlabel("Pasos de entrenamiento")
            plt.ylabel("Recompensa media por episodio")
            plt.title(f"{algo} - {budget} ({len(run_dirs)} semillas)")
            plt.legend()
            plt.tight_layout()

            out_path = figures_dir / f"{algo}_{budget}_learning_curve.png"
            plt.savefig(out_path)
            plt.close()
            print(f"Guardada {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="lanza los entrenamientos largos")
    p_train.add_argument("--only", help="limita a una configuración, p.ej. qlearning_high")

    sub.add_parser("curves", help="genera las curvas de aprendizaje a partir de TensorBoard")

    args = parser.parse_args()
    if args.cmd == "train":
        run_all(only=args.only)
    elif args.cmd == "curves":
        generate_curves()


if __name__ == "__main__":
    main()
