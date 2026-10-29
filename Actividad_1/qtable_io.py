"""
Guardar y cargar tablas Q de la Fase B, con los metadatos necesarios para
identificarlas (algoritmo, presupuesto de discretización, semilla y configuración).
"""
import json
import sys
from pathlib import Path

import numpy as np

# Permite "python qtable_io.py ..." suelto, sin pasar por config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deustorl.common import QTable


def run_id(algo, budget, seed):
    return f"{algo}_{budget}_seed{seed}"


def save_qtable(q_table, out_dir, *, algo, budget, seed, params, **extra_meta):
    """Guarda la tabla (.npy) y sus metadatos (.json) como out_dir/<algo>_<budget>_seed<seed>.*"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / run_id(algo, budget, seed)

    np.save(base.with_suffix(".npy"), q_table.table)
    meta = {
        "algo": algo,
        "budget": budget,
        "seed": seed,
        "params": params,
        "n_states": int(q_table.n_states),
        "n_actions": int(q_table.n_actions),
        **extra_meta,
    }
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return base


def load_qtable(path):
    """Carga una tabla Q guardada con save_qtable (path con o sin extensión).
    Devuelve (q_table, metadatos)."""
    base = Path(path).with_suffix("")
    with open(base.with_suffix(".json")) as f:
        meta = json.load(f)
    q_table = QTable(meta["n_states"], meta["n_actions"])
    q_table.table = np.load(base.with_suffix(".npy"))
    return q_table, meta


def list_saved_qtables(out_dir):
    """Metadatos (.json) de todas las tablas Q guardadas en out_dir, ordenados por nombre."""
    return sorted(Path(out_dir).glob("*.json"))


DEFAULT_CSV_DIR = Path(__file__).resolve().parent / "qtables_csv"
DEFAULT_CSV_SAMPLE = 1000


def qtable_to_csv(npy_path, out_dir=None, sample=DEFAULT_CSV_SAMPLE):
    """Convierte un .npy de tabla Q a .csv, solo para inspección manual. Incluye una columna
    "state" para identificar la fila. Por defecto exporta una muestra aleatoria de "sample"
    estados; usa sample=None para exportar la tabla completa (puede ser muy grande en "high").
    Devuelve la ruta del .csv generado.
    python qtable_io.py results/qtables/qlearning_high_seed0.npy  """
    npy_path = Path(npy_path).with_suffix(".npy")
    table = np.load(npy_path)
    n_states, n_actions = table.shape

    if sample is not None and sample < n_states:
        states = np.sort(np.random.default_rng(0).choice(n_states, size=sample, replace=False))
    else:
        states = np.arange(n_states)

    out_dir = Path(out_dir or DEFAULT_CSV_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / npy_path.with_suffix(".csv").name

    header = "state," + ",".join(f"action_{a}" for a in range(n_actions))
    rows = np.column_stack([states, table[states]])
    np.savetxt(out_path, rows, delimiter=",", header=header, comments="", fmt=["%d"] + ["%.18e"] * n_actions)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convierte una tabla Q (.npy) a .csv para inspección manual.")
    parser.add_argument("npy_path", help="ruta al .npy (o al nombre base sin extensión)")
    parser.add_argument("--out-dir", default=None, help=f"carpeta de salida (por defecto {DEFAULT_CSV_DIR})")
    parser.add_argument("--sample", type=int, default=DEFAULT_CSV_SAMPLE,
                         help=f"número de estados a exportar al azar (por defecto {DEFAULT_CSV_SAMPLE}); ignorado si se usa --all")
    parser.add_argument("--all", action="store_true", help="exporta la tabla completa en vez de una muestra")
    args = parser.parse_args()

    out_path = qtable_to_csv(args.npy_path, args.out_dir, sample=None if args.all else args.sample)
    print(f"Guardado {out_path}")
