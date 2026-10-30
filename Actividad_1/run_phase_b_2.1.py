"""
Mauro - Fase B 2.1: inspección visual y demostración de las políticas finales.

Carga  las 3 políticas ganadoras seleccionadas (guardadas en
'results/final_policies/') para los presupuestos low, medium y high. Ejecuta una demostración
cualitativa de vuelo y aterrizaje en 'LunarLander-v3'.

Uso:
    python visualize_best_policies.py      # Ejecuta la demostración gráfica (según la variable RENDER)
"""
import glob
import os
import random
import time
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

def load_qtable_safe(filepath):
    """Carga la tabla Q e información de metadatos."""
    try:
        q_table, metadata = qtable_io.load_qtable(filepath)
        return q_table, metadata
    except Exception:
        q_table = np.load(filepath, allow_pickle=True)
        metadata = {"algo": "Ganador", "seed": "N/A"}
        return q_table, metadata

def visualize_final_policies(policies_dir=None, n_episodes=2, render=False):
    """
    Carga e inspecciona las políticas de final_policies.
    Si render=True, habilita la interfaz gráfica de Pygame.
    """
    if policies_dir is None:
        policies_dir = cfg.RESULTS_DIR / "final_policies"

    budgets = ["low", "medium", "high"]

    mode_str = "VISUAL (Pygame)" if render else "SIN GRÁFICOS (Modo Consola)"
    print("==================================================================")
    print(f" DEMOSTRACIÓN DE LAS 3 POLÍTICAS FINALES [{mode_str}]")
    print("==================================================================")

    for budget in budgets:
        pattern_subfolder = os.path.join(str(policies_dir), f"*best_policy_{budget}*", "*.npy")
        pattern_direct = os.path.join(str(policies_dir), f"*{budget}*.npy")

        matches = glob.glob(pattern_subfolder) or glob.glob(pattern_direct)

        if not matches:
            print(f"\n[Aviso] No se encontró la tabla Q final para el presupuesto '{budget.upper()}' en '{policies_dir}'.")
            continue

        filepath = matches[0]
        q_table, metadata = load_qtable_safe(filepath)

        algo = metadata.get("algo", "Ganador")
        seed = metadata.get("seed", "N/A")

        print(f"\n[{'Visualizando' if render else 'Evaluando'}] Presupuesto {budget.upper()} -> Algoritmo: {str(algo).upper()} (Semilla Train {seed})")

        set_seed(EVAL_SEED)

        # 1. Crear entorno según lunar_common.py
        env = lunar_common.make_env(budget)

        # 2. Inyectar render_mode solo si render=True
        if render and hasattr(env, "unwrapped"):
            env.unwrapped.render_mode = "human"

        env.reset(seed=EVAL_SEED)

        try:
            mean_reward, mean_steps = evaluate_policy(
                env=env,
                q_table=q_table,
                policy=max_policy,
                n_episodes=n_episodes,
                verbose=False
            )
            print(f"  -> Resultado ({n_episodes} ep): Reward Medio = {mean_reward:.2f} | Pasos Medios = {mean_steps:.1f}")
        except Exception as e:
            print(f"  -> Error durante la ejecución: {e}")
        finally:
            env.close()

        if render:
            time.sleep(1.5)


if __name__ == "__main__":
    RENDER = True 
    EPISODES = 2 
    visualize_final_policies(n_episodes=EPISODES, render=RENDER)