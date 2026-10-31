"""
Mauro - Fase B 2: evaluación masiva y selección de las políticas ganadoras.

Carga y evalúa 27 tablas Q generadas en la Fase B
(3 presupuestos x 3 algoritmos x 3 semillas) mediante 'evaluate_policy'. Para relaizar una
comparativa justa, se fija la semilla global de evaluación (EVAL_SEED = 42) a lo
largo de N episodios por agente. Finalmente, selecciona el modelo con mayor recompensa media
para cada presupuesto (low, medium y high) y exporta las 3 políticas ganadoras en'results/final_policies/'.

Uso:
    python run_phase_b_2.py                # Ejecuta la evaluación y guarda los 3 ganadores (low medium high)
"""

import os
import glob
import random
import numpy as np

import config as cfg
import lunar_common
import qtable_io
from deustorl.common import evaluate_policy, max_policy

EVAL_SEED = 42

def set_seed(seed=EVAL_SEED):
    """Fija las semillas globales para garantizar reproducibilidad exacta."""
    random.seed(seed)
    np.random.seed(seed)

def evaluate_all_qtables(qtables_dir, n_episodes=50):
    """
    Carga y evalúa las 27 tablas Q de la Fase B bajo la semilla global EVAL_SEED=42.
    """
    pattern = os.path.join(qtables_dir, "*.npy")
    qtable_files = [f for f in glob.glob(pattern) if "best_policy" not in os.path.basename(f)]

    print("\n==================================================================")
    print(f" DETECTADOS {len(qtable_files)} ARCHIVOS EN '{qtables_dir}'")
    print("==================================================================")

    if not qtable_files:
        print(f"Error: No se encontraron tablas .npy en '{qtables_dir}'.")
        return []

    results = []

    for idx, filepath in enumerate(qtable_files, 1):
        set_seed(EVAL_SEED)

        q_table, metadata = qtable_io.load_qtable(filepath)

        filename = os.path.basename(filepath).lower()

        algo = metadata.get("algo")
        if not algo:
            if "esarsa" in filename:
                algo = "esarsa"
            elif "sarsa" in filename:
                algo = "sarsa"
            elif "qlearning" in filename or "q_learning" in filename:
                algo = "qlearning"

        budget = metadata.get("budget")
        if not budget:
            if "low" in filename:
                budget = "low"
            elif "medium" in filename:
                budget = "medium"
            elif "high" in filename:
                budget = "high"

        seed = metadata.get("seed")
        if seed is None:
            for s in cfg.PHASE_B_SEEDS:  # verificamos la semilla
                if f"seed{s}" in filename or f"s{s}" in filename:
                    seed = s
                    break
            if seed is None:
                seed = 0


        env = lunar_common.make_env(budget)
        env.reset(seed=EVAL_SEED)

        try:
            mean_reward, mean_steps = evaluate_policy(
                env=env,
                q_table=q_table,
                policy=max_policy,
                n_episodes=n_episodes,
                verbose=False
            )
        except Exception as err:
            print(f" Error evaluando {os.path.basename(filepath)}: {err}")
            env.close()
            continue

        env.close()

        print(f"[{idx:02d}/27] {os.path.basename(filepath):<30} | Algo: {algo:<10} | Budget: {budget:<6} | Seed Train: {seed} | Reward: {mean_reward:6.2f}")

        results.append({
            "filepath": filepath,
            "algo": algo,
            "budget": budget,
            "seed": seed,
            "mean_reward": mean_reward,
            "mean_steps": mean_steps,
            "q_table": q_table,
            "params": metadata.get("params", {})
        })

    return results


def select_and_save_best_policies(results, output_dir):
    """
    Agrupa los resultados, calcula las medias por algoritmo y exporta
    la mejor tabla Q absoluta para cada presupuesto (low, medium, high).
    """
    os.makedirs(output_dir, exist_ok=True)
    budgets = ["low", "medium", "high"]
    algos = ["sarsa", "esarsa", "qlearning"]
    best_policies = {}

    print("\n==================================================================")
    print(" RESUMEN Y COMPARATIVA FASE B")
    print("==================================================================")

    for budget in budgets:
        print(f"\n--- PRESUPUESTO: {budget.upper()} ---")
        budget_candidates = []

        for algo in algos:
            runs = [r for r in results if r["budget"] == budget and r["algo"] == algo]
            if not runs:
                continue

            avg_reward = np.mean([r["mean_reward"] for r in runs])
            best_run = max(runs, key=lambda x: x["mean_reward"])
            budget_candidates.append(best_run)

            print(f"  * {algo.upper():<12} | Reward Promedio (3 semillas): {avg_reward:6.2f} | Mejor Semilla ({best_run['seed']}): {best_run['mean_reward']:.2f}")

        if budget_candidates:
            winner = max(budget_candidates, key=lambda x: x["mean_reward"])
            best_policies[budget] = winner

            final_path = os.path.join(output_dir, f"best_policy_{budget}")

            qtable_io.save_qtable(
                winner["q_table"],
                final_path,
                algo=winner["algo"],
                budget=winner["budget"],
                seed=winner["seed"],
                params=winner.get("params", {"mean_reward": winner["mean_reward"]})
            )

    print("\n==================================================================")
    print(" POLÍTICAS FINALES SELECCIONADAS (final_policies)")
    print("==================================================================")
    for budget, winner in best_policies.items():
        print(f"  * [{budget.upper():<6}] Algoritmo: {winner['algo'].upper():<10} | Semilla Train: {winner['seed']} | Reward: {winner['mean_reward']:.2f}")


if __name__ == "__main__":
    set_seed(EVAL_SEED)
    path_qtables = cfg.QTABLE_DIR
    path_best_policies = cfg.RESULTS_DIR / "final_policies"

    raw_results = evaluate_all_qtables(qtables_dir=str(path_qtables), n_episodes=50)

    if raw_results:
        select_and_save_best_policies(raw_results, output_dir=str(path_best_policies))