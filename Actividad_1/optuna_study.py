"""
Un estudio de optuna para UN algoritmo y UN presupuesto de discretización.
Ejecución: python optuna_study.py --algo sarsa --budget low --owner 1 [--trials 30]
"""

import argparse
import json

import optuna
from optuna.samplers import TPESampler

import config as cfg
from lunar_common import silence_tensorboard, train_and_evaluate

def suggest_params(trial):
    """Hiperparámetros que propone Optuna en cada trial 
    (rangos en config.SEARCH_SPACE)."""

    s = cfg.SEARCH_SPACE

    return {
        "learning_rate": trial.suggest_float("learning_rate", **s["learning_rate"]),
        "learning_rate_decay": trial.suggest_float("learning_rate_decay", **s["learning_rate_decay"]),
        "lr_episodes_decay": trial.suggest_categorical("lr_episodes_decay", s["lr_episodes_decay"]),
        "discount_rate": trial.suggest_float("discount_rate", **s["discount_rate"]),
        "epsilon": trial.suggest_float("epsilon", **s["epsilon"]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=cfg.ALGOS)
    parser.add_argument("--budget", required=True, choices=list(cfg.BUDGETS))
    parser.add_argument("--owner", required=True, help="1 (Alberto), o 2 (Xabier)")
    parser.add_argument("--trials", type=int, default=cfg.N_TRIALS)
    args = parser.parse_args()

    silence_tensorboard()
    cfg.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = cfg.study_name(args.algo, args.budget)

    def objective(trial):
        params = suggest_params(trial)
        reward, steps = train_and_evaluate(args.algo, args.budget, params)
        trial.set_user_attr("avg_eval_steps", steps)
        return reward
    
    study = optuna.create_study(
        study_name=name,
        storage=cfg.db_url(args.owner),
        sampler=TPESampler(seed=cfg.SEED),
        direction = "maximize",
        load_if_exists=True, 
        )

     # Reanudable: solo ejecuta los trials que faltan hasta args.trials
    done = len([t for t in study.trials if t.state.is_finished()])
    pending = args.trials - done
    if pending > 0:
        study.optimize(objective, n_trials = pending)

    print(f"[{name}] mejor reward = {study.best_value:.2f}  params = {study.best_params}")
    best = {
        "algo": args.algo,
        "budget": args.budget,
        "n_bins": cfg.BUDGETS[args.budget],
        "reward": study.best_value,
        "params": study.best_params,
    }

    with open(cfg.RESULTS_DIR / f"best_{name}.json", "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
