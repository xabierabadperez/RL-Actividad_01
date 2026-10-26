"""
Entorno y entrenamiento/evaluación de una configuración (Fase A)

"""

import contextlib
import io
import random

import gymnasium as gym

import config as cfg
import deustorl.expected_sarsa as _esarsa_mod
import deustorl.qlearning as _qlearning_mod
import deustorl.sarsa as _sarsa_mod
from deustorl.common import EpsilonGreedyPolicy, evaluate_policy, max_policy
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.helpers import DiscretizedObservationWrapper
from deustorl.qlearning import QLearning
from deustorl.sarsa import Sarsa

ALGO_CLASSES = {
    "sarsa": Sarsa,
    "esarsa": ExpectedSarsa,
    "qlearning": QLearning,
}

class _NullLogger:
    """Sustituye al TensorboardLogger en la Fase A 
    (evita una carpeta de logs por trial)."""

    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

def silence_tensorboard():
    """ Desactiva tensorboard sin modificar deustorl"""

    for module in (_sarsa_mod, _esarsa_mod, _qlearning_mod):
        module.TensorboardLogger = _NullLogger

def make_env(budget):
    """ LunarLander con la observación discretizada según el presupuesto 
        (low/medium/high)
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return DiscretizedObservationWrapper(
            gym.make(cfg.ENV_ID, **cfg.ENV_KWARGS),
            n_bins=cfg.BUDGETS[budget],
            low=cfg.OBS_LOW,
            high=cfg.OBS_HIGH,
        )

def train_and_evaluate(algo_name, budget, params, seed=cfg.SEED, n_steps=None):
    """
    Entrena con 'params' y devuelve (reward medio, pasos medios) de la evaluación greedy.
    params: learning_rate, learning_rate_decay, lr_episodes_decay, discount_rate, epsilon.
    n_steps: solo para pruebas rápidas; por defecto cfg.TRAIN_STEPS.

    """
    random.seed(seed)
    env = make_env(budget)
    env.reset(seed=seed)
    algo = ALGO_CLASSES[algo_name](env)

    algo.learn(
        EpsilonGreedyPolicy(epsilon=params["epsilon"]),
        n_steps=n_steps or cfg.TRAIN_STEPS,
        discount_rate=params["discount_rate"],
        lr=params["learning_rate"],
        lrdecay=params["learning_rate_decay"],
        n_episodes_decay=params["lr_episodes_decay"],
    )

    with contextlib.redirect_stdout(io.StringIO()):
        avg_reward, avg_steps = evaluate_policy(
            algo.env, algo.q_table, max_policy, n_episodes=cfg.EVAL_EPISODES
        )
    env.close()
    return avg_reward, avg_steps

if __name__ == "__main__":
    # Prueba rápida : pyhton lunar common
    silence_tensorboard()
    p = dict(learning_rate=0.1, learning_rate_decay=0.95, lr_episodes_decay=100, discount_rate=0.99, epsilon=0.1)
    print(train_and_evaluate("qlearning", "low", p, n_steps=20_000))
