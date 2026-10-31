"""
Análisis de la Fase A a partir de la base unida (phase_a_all.db): tabla de las
9 configuraciones, efecto del presupuesto de discretización por algoritmo y gráficas.
Ejecución: python analyze_phase_a.py

No modifica la base de datos ni los best_*.json; sobrescribe sus propias salidas.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import TrialState

import config as cfg

DB = cfg.OPTUNA_DIR / "phase_a_all.db"
FIG_DIR = cfg.RESULTS_DIR / "optuna_figures"
BEST_CONFIGS_MD = cfg.RESULTS_DIR / "best_configs.md"
BEST_CONFIGS_CSV = cfg.RESULTS_DIR / "best_configs.csv"
CSV_DECIMALS = 6  # evita ruido de coma flotante (p. ej. 0.8200000000000001)
EFFECT_MD = cfg.RESULTS_DIR / "discretization_effect.md"

SOLVED_REWARD = 200  # umbral de episodio resuelto (enunciado)
FIG_DPI = 150
FIG_SIZE = (8, 4.5)  # figura de un solo panel, como las curvas de la Fase B

ALGO_NAMES = {"sarsa": "SARSA", "esarsa": "Expected SARSA", "qlearning": "Q-Learning"}
# Paleta de las curvas de la Fase B (colores tomados a ojo de las figuras; coincide con
# la paleta "colorblind" de seaborn)
ALGO_COLORS = {"sarsa": "#0173B2", "esarsa": "#DE8F05", "qlearning": "#029E73"}
BUDGET_NAMES = {"low": "baja", "medium": "media", "high": "alta"}
REWARD_LABEL = "Recompensa de evaluación\n(media de 50 episodios)"


def apply_style():
    """Mismo aspecto que las curvas de la Fase B: sin bordes arriba/derecha, rejilla
    tenue y línea de referencia discontinua."""
    plt.switch_backend("Agg")  # solo guardar PNG, sin abrir ventanas
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def sqlite_url(path):
    return f"sqlite:///{Path(path).resolve().as_posix()}"


def n_states(budget):
    # mismo n_bins en las 8 dimensiones (DiscretizedObservationWrapper)
    return cfg.BUDGETS[budget] ** len(cfg.OBS_LOW)


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


def budget_title(budget):
    return f"Discretización {BUDGET_NAMES[budget]} ({cfg.BUDGETS[budget]} intervalos)"


def budget_panel_title(budget):
    return f"Discretización {BUDGET_NAMES[budget]}\n({cfg.BUDGETS[budget]} intervalos, {fmt_int(n_states(budget))} estados)"


def rewards(study):
    """Reward de cada trial completado, en orden de ejecución."""
    return [t.value for t in study.trials if t.state == TrialState.COMPLETE]


def load_studies():
    """Carga los 9 estudios {(algo, budget): study} y comprueba que estén completos."""
    if not DB.exists():
        raise SystemExit(f"{DB} no existe. Ejecuta antes merge_dbs.py.")
    studies = {}
    for algo in cfg.ALGOS:
        for budget in cfg.BUDGETS:
            name = cfg.study_name(algo, budget)
            try:
                study = optuna.load_study(study_name=name, storage=sqlite_url(DB))
            except KeyError:
                raise SystemExit(f"Falta el estudio '{name}' en {DB.name}.")
            done = len(rewards(study))
            if done < cfg.N_TRIALS:
                raise SystemExit(f"{name}: {done}/{cfg.N_TRIALS} trials completos.")
            studies[(algo, budget)] = study
    return studies


def check_against_json(studies):
    """Avisa si el mejor reward de la base no coincide con results/best_*.json."""
    for (algo, budget), study in studies.items():
        name = cfg.study_name(algo, budget)
        path = cfg.RESULTS_DIR / f"best_{name}.json"
        if not path.exists():
            print(f"AVISO: falta {path.name}")
            continue
        with open(path) as f:
            saved = json.load(f)
        if not np.isclose(saved["reward"], study.best_value):
            print(f"AVISO: {path.name} ({saved['reward']:.2f}) "
                  f"difiere de la base ({study.best_value:.2f})")


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def best_configs_table(studies):
    """Tabla A: la mejor configuración de cada estudio (las 9 del enunciado)."""
    header = ["Algoritmo", "Presupuesto", "n_bins", "Estados", "Mejor reward", "Pasos eval.",
              "learning_rate", "lr_decay", "lr_episodes_decay", "discount_rate", "epsilon"]
    rows = []
    for (algo, budget), study in studies.items():
        best = study.best_trial
        p = best.params
        rows.append([
            ALGO_NAMES[algo], budget, cfg.BUDGETS[budget], fmt_int(n_states(budget)),
            f"{best.value:.2f}", f"{best.user_attrs.get('avg_eval_steps', float('nan')):.0f}",
            f"{p['learning_rate']:.4f}", f"{p['learning_rate_decay']:.2f}",
            p["lr_episodes_decay"], f"{p['discount_rate']:.2f}", f"{p['epsilon']:.2f}",
        ])
    return md_table(header, rows)


def write_best_configs_csv(studies):
    """Las 9 configuraciones en CSV, con los mismos nombres de campo que los best_*.json
    y los valores sin formatear."""
    fields = ["algo", "budget", "n_bins", "n_states", "reward", "avg_eval_steps",
              "learning_rate", "learning_rate_decay", "lr_episodes_decay",
              "discount_rate", "epsilon"]
    with open(BEST_CONFIGS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (algo, budget), study in studies.items():
            best = study.best_trial
            row = {"algo": algo, "budget": budget, "n_bins": cfg.BUDGETS[budget],
                   "n_states": n_states(budget), "reward": best.value,
                   "avg_eval_steps": best.user_attrs.get("avg_eval_steps"), **best.params}
            writer.writerow({k: round(v, CSV_DECIMALS) if isinstance(v, float) else v
                             for k, v in row.items()})


def effect_table(studies):
    """Tabla B: cómo cambia el rendimiento con el presupuesto de discretización.
    El mejor mide el potencial; la mediana, si el presupuesto funciona sin afinar."""
    header = ["Algoritmo", "Presupuesto", "Estados", "Mejor", "Mediana", "Q1", "Q3",
              f"Trials > 0 (de {cfg.N_TRIALS})"]
    rows = []
    for (algo, budget), study in studies.items():
        r = np.array(rewards(study))
        q1, q3 = np.percentile(r, [25, 75])
        rows.append([
            ALGO_NAMES[algo], budget, fmt_int(n_states(budget)), f"{r.max():.1f}",
            f"{np.median(r):.1f}", f"{q1:.1f}", f"{q3:.1f}", int((r > 0).sum()),
        ])
    return md_table(header, rows)


def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def zero_line(ax):
    ax.axhline(0, ls="--", color="gray", lw=0.8, zorder=1)


def fig_best_vs_states(studies):
    """Fig 1: mejor reward frente al tamaño del espacio de estados."""
    states = [n_states(b) for b in cfg.BUDGETS]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for algo in cfg.ALGOS:
        best = [studies[(algo, b)].best_value for b in cfg.BUDGETS]
        ax.plot(states, best, marker="o", lw=2, color=ALGO_COLORS[algo], label=ALGO_NAMES[algo])
    ax.axhline(SOLVED_REWARD, ls="--", color="gray", lw=0.8, label=f"resuelto ({SOLVED_REWARD})")
    zero_line(ax)
    ax.set_xscale("log")  # 6.561 → 5.764.801 no cabe en un eje lineal
    ax.set_xticks(states)
    ax.set_xticklabels([f"{BUDGET_NAMES[b]} ({cfg.BUDGETS[b]} intervalos)\n{fmt_int(n)} estados"
                        for b, n in zip(cfg.BUDGETS, states)])
    ax.minorticks_off()
    ax.set_xlabel("Presupuesto de discretización")
    ax.set_ylabel(REWARD_LABEL)
    ax.set_title("Mejor configuración de Optuna según el presupuesto de discretización")
    ax.legend(loc="upper left")
    save_fig(fig, "fig1_best_vs_states.png")


def fig_distribution(studies):
    """Fig 2: distribución de los 30 rewards de cada estudio, por presupuesto."""
    rng = np.random.default_rng(cfg.SEED)
    fig, axes = plt.subplots(1, len(cfg.BUDGETS), figsize=(12, 4.5), sharey=True)
    for ax, budget in zip(axes, cfg.BUDGETS):
        data = [rewards(studies[(a, budget)]) for a in cfg.ALGOS]
        box = ax.boxplot(data, showfliers=False, patch_artist=True,
                         medianprops=dict(color="black", lw=1.2))
        for patch, algo in zip(box["boxes"], cfg.ALGOS):
            patch.set(facecolor=ALGO_COLORS[algo], alpha=0.25, edgecolor=ALGO_COLORS[algo])
        ax.set_xticks(range(1, len(cfg.ALGOS) + 1))
        ax.set_xticklabels([ALGO_NAMES[a].replace(" ", "\n") for a in cfg.ALGOS])
        for i, (algo, r) in enumerate(zip(cfg.ALGOS, data), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(r)), r, s=12, alpha=0.7,
                       color=ALGO_COLORS[algo], zorder=3)
        zero_line(ax)
        ax.set_title(budget_panel_title(budget), fontsize=10)
    axes[0].set_ylabel("Recompensa de evaluación de cada trial\n(media de 50 episodios)")
    fig.suptitle(f"Distribución de la recompensa de los {cfg.N_TRIALS} trials de cada estudio",
                 y=1.06)
    save_fig(fig, "fig2_reward_distribution.png")


def fig_optimization_history(studies):
    """Fig 3: reward de cada trial y mejor valor hasta ese trial, un panel por estudio."""
    fig, axes = plt.subplots(len(cfg.ALGOS), len(cfg.BUDGETS), figsize=(12, 9),
                             sharex=True, sharey=True)
    for i, algo in enumerate(cfg.ALGOS):
        for j, budget in enumerate(cfg.BUDGETS):
            ax = axes[i, j]
            r = np.array(rewards(studies[(algo, budget)]))
            ax.scatter(range(len(r)), r, s=12, alpha=0.4, color=ALGO_COLORS[algo])
            ax.plot(np.maximum.accumulate(r), lw=2, color=ALGO_COLORS[algo])
            zero_line(ax)
            if i == 0:
                ax.set_title(budget_panel_title(budget), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ALGO_NAMES[algo]}\nrecompensa de evaluación")
            if i == len(cfg.ALGOS) - 1:
                ax.set_xlabel("Trial")
    fig.suptitle("Historia de optimización (línea: mejor recompensa hasta ese trial)")
    save_fig(fig, "fig3_optimization_history.png")


def fig_param_scatter(studies):
    """Fig 4: reward frente a learning_rate (log) y epsilon, un panel por presupuesto."""
    params = [("learning_rate", True), ("epsilon", False)]
    fig, axes = plt.subplots(len(params), len(cfg.BUDGETS), figsize=(12, 7), sharey=True)
    for i, (param, log) in enumerate(params):
        for j, budget in enumerate(cfg.BUDGETS):
            ax = axes[i, j]
            for algo in cfg.ALGOS:
                trials = [t for t in studies[(algo, budget)].trials
                          if t.state == TrialState.COMPLETE]
                ax.scatter([t.params[param] for t in trials], [t.value for t in trials],
                           s=14, alpha=0.6, color=ALGO_COLORS[algo], label=ALGO_NAMES[algo])
            if log:
                ax.set_xscale("log")
            zero_line(ax)
            if i == 0:
                ax.set_title(budget_panel_title(budget), fontsize=10)
            if j == 0:
                ax.set_ylabel("Recompensa de evaluación\n(media de 50 episodios)")
            ax.set_xlabel(param + (" (escala log)" if log else ""))
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.subplots_adjust(hspace=0.4)  # separa la etiqueta de la fila 1 de la fila 2
    fig.suptitle("Recompensa de cada trial según learning_rate y epsilon")
    save_fig(fig, "fig4_param_scatter.png")


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    apply_style()
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    studies = load_studies()
    check_against_json(studies)

    table_a = best_configs_table(studies)
    table_b = effect_table(studies)
    BEST_CONFIGS_MD.write_text(
        "# Mejores configuraciones por algoritmo y presupuesto\n\n" + table_a + "\n",
        encoding="utf-8")
    write_best_configs_csv(studies)
    EFFECT_MD.write_text(
        "# Efecto del presupuesto de discretización\n\n" + table_b + "\n", encoding="utf-8")

    fig_best_vs_states(studies)
    fig_distribution(studies)
    fig_optimization_history(studies)
    fig_param_scatter(studies)

    print(table_a, end="\n\n")
    print(table_b, end="\n\n")
    print(f"Tablas en {cfg.RESULTS_DIR}; figuras en {FIG_DIR}")


if __name__ == "__main__":
    main()