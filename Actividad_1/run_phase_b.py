"""
Katherine - Fase B: entrenamientos largos y curvas de aprendizaje.

Entrena las 9 configuraciones seleccionadas en la Fase A (results/best_<algo>_<budget>.json)
con config.PHASE_B_SEEDS semillas cada una (27 entrenamientos por defecto), guarda cada tabla Q
con sus metadatos (qtable_io.py) y registra la evolución en TensorBoard. Después, a partir de
esos registros, genera las curvas de aprendizaje (media +/- dispersión entre semillas).

Uso:
    python run_phase_b.py train                        # lanza los 27 entrenamientos
    python run_phase_b.py train --only qlearning_high   # solo una configuración (9 posibles)
    python run_phase_b.py curves                        # una gráfica por algoritmo+presupuesto
    python run_phase_b.py discretization-curves          # una gráfica por presupuesto, con los 3 algoritmos juntos
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


BUDGET_LABELS = {"low": "baja", "medium": "media", "high": "alta"}
ALGO_LABELS = {"sarsa": "SARSA", "esarsa": "Expected SARSA", "qlearning": "Q-Learning"}
N_SEGMENTS = 150  # en cuántos tramos se divide el eje de pasos, para suavizar la curva


def average_per_segment(runs, segment_edges):
    """Divide el eje de pasos en tramos (definidos por 'segment_edges'). Para cada semilla,
    calcula el promedio de sus puntos que caen en cada tramo (una semilla puede tener
    varios episodios en el mismo tramo). Al final, con los promedios de las 3 semillas
    en cada tramo, calcula la media y la desviación estándar entre esas 3 semillas."""
    seed_averages = []
    for steps, rewards in runs:
        segment_of_each_point = np.digitize(steps, segment_edges) - 1
        seed_average = np.full(len(segment_edges) - 1, np.nan)
        for segment in range(len(seed_average)):
            points_in_segment = segment_of_each_point == segment
            if points_in_segment.any():
                seed_average[segment] = rewards[points_in_segment].mean()
        seed_averages.append(seed_average)

    seed_averages = np.array(seed_averages)  # una fila por semilla
    mean = np.nanmean(seed_averages, axis=0)
    std = np.nanstd(seed_averages, axis=0)
    return mean, std


def plot_discretization_curves(figures_dir=None):
    """Una figura por presupuesto de discretización (low/medium/high), con los 3
    algoritmos superpuestos: línea = media de las 3 semillas, sombra = +/- 1 desviación
    estándar. Mismo estilo y escala que las gráficas del informe."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir or (cfg.RESULTS_DIR / "figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Divide los 2.000.000 de pasos en 150 tramos iguales; cada punto de la gráfica
    # representa uno de esos tramos.
    segment_edges = np.linspace(0, cfg.PHASE_B_TRAIN_STEPS, N_SEGMENTS + 1)
    segment_centers = (segment_edges[:-1] + segment_edges[1:]) / 2

    for budget in cfg.BUDGETS:
        plt.figure(figsize=(8, 5))
        has_data = False

        for algo in cfg.ALGOS:
            prefix = cfg.study_name(algo, budget) + "_seed"
            run_dirs = sorted(d for d in cfg.PHASE_B_LOG_DIR.glob(f"{prefix}*") if d.is_dir())
            if not run_dirs:
                print(f"Sin logs todavía para {algo}/{budget}, se omite.")
                continue
            has_data = True

            runs = [_load_scalar(run_dir) for run_dir in run_dirs]
            mean, std = average_per_segment(runs, segment_edges)
            has_value = ~np.isnan(mean)  # algunos tramos pueden quedar sin ningún punto

            plt.plot(segment_centers[has_value], mean[has_value], label=ALGO_LABELS[algo])
            plt.fill_between(
                segment_centers[has_value],
                (mean - std)[has_value],
                (mean + std)[has_value],
                alpha=0.25,
            )

        if not has_data:
            plt.close()
            continue

        plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        plt.xlim(0, cfg.PHASE_B_TRAIN_STEPS)
        plt.ylim(-400, 200)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Pasos de entrenamiento")
        plt.ylabel("Recompensa de entrenamiento\n(media móvil de hasta 100 episodios)")
        plt.title(
            f"Discretización {BUDGET_LABELS[budget]} ({cfg.BUDGETS[budget]} intervalos): "
            "media de 3 ejecuciones +/- 1 desviación estándar"
        )
        plt.legend()
        plt.tight_layout()

        out_path = figures_dir / f"discretizacion_{budget}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Guardada {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="lanza los entrenamientos largos")
    p_train.add_argument("--only", help="limita a una configuración, p.ej. qlearning_high")

    sub.add_parser("curves", help="genera las curvas de aprendizaje a partir de TensorBoard")
    sub.add_parser("discretization-curves", help="compara los 3 algoritmos por presupuesto de discretización")

    args = parser.parse_args()
    if args.cmd == "train":
        run_all(only=args.only)
    elif args.cmd == "curves":
        generate_curves()
    elif args.cmd == "discretization-curves":
        plot_discretization_curves()


if __name__ == "__main__":
    main()
