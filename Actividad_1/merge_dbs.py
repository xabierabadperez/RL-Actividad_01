"""
Une las bases de Optuna de cada persona (phase_a_1.db, phase_a_2.db) en una sola
para el análisis conjunto de los 9 estudios. Se ejecuta UNA vez, cuando los dos
entrenamientos han terminado.
Ejecución: python merge_dbs.py [--inputs a.db b.db] [--output all.db] [--force]

No modifica las bases de origen: copia los estudios (trials, parámetros y
atributos) a una base nueva.
"""

import argparse
from pathlib import Path

import optuna
from optuna.trial import TrialState

import config as cfg

DEFAULT_INPUTS = [cfg.OPTUNA_DIR / "phase_a_1.db", cfg.OPTUNA_DIR / "phase_a_2.db"]
DEFAULT_OUTPUT = cfg.OPTUNA_DIR / "phase_a_all.db"


def sqlite_url(path):
    return f"sqlite:///{Path(path).resolve().as_posix()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true",
                        help="unir aunque haya estudios sin terminar (resultados parciales)")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Todas las comprobaciones van antes de escribir nada en la base de salida
    if args.output.exists():
        raise SystemExit(f"{args.output} ya existe. Bórrala o usa --output con otro nombre.")
    for db in args.inputs:
        if not db.exists() or db.stat().st_size == 0:
            raise SystemExit(f"{db} no existe o está vacía (¿se ha ejecutado su script?).")

    origen = {}
    for db in args.inputs:
        for summary in optuna.get_all_study_summaries(storage=sqlite_url(db)):
            if summary.study_name in origen:
                raise SystemExit(
                    f"El estudio '{summary.study_name}' está en {origen[summary.study_name].name} "
                    f"y en {db.name}: no se puede unir sin perder trials."
                )
            origen[summary.study_name] = db

    # Un estudio terminado no tiene trials en curso y llega a N_TRIALS completos
    sin_terminar = []
    for name, db in sorted(origen.items()):
        trials = optuna.load_study(study_name=name, storage=sqlite_url(db)).trials
        complete = sum(t.state == TrialState.COMPLETE for t in trials)
        en_curso = sum(t.state in (TrialState.RUNNING, TrialState.WAITING) for t in trials)
        if en_curso or complete < cfg.N_TRIALS:
            sin_terminar.append(f"  {name} ({db.name}): {complete}/{cfg.N_TRIALS} completos, {en_curso} en curso")
    if sin_terminar and not args.force:
        raise SystemExit("Hay estudios sin terminar (usa --force para unirlos igualmente):\n"
                         + "\n".join(sin_terminar))

    to_storage = sqlite_url(args.output)
    for name, db in sorted(origen.items()):
        optuna.copy_study(from_study_name=name, from_storage=sqlite_url(db),
                          to_storage=to_storage, to_study_name=name)

    print(f"Base unida: {args.output}\n")
    print(f"{'estudio':<20}{'origen':<16}{'trials':>7}{'COMPLETE':>10}{'mejor reward':>14}")
    for name in sorted(origen):
        study = optuna.load_study(study_name=name, storage=to_storage)
        done = [t for t in study.trials if t.state == TrialState.COMPLETE]
        best = f"{study.best_value:.2f}" if done else "-"
        print(f"{name:<20}{origen[name].name:<16}{len(study.trials):>7}{len(done):>10}{best:>14}")

    esperados = {cfg.study_name(a, b) for a in cfg.ALGOS for b in cfg.BUDGETS}
    if faltan := sorted(esperados - set(origen)):
        print(f"\nAVISO: faltan estudios: {', '.join(faltan)}")
    else:
        print(f"\nLos {len(esperados)} estudios están presentes.")


if __name__ == "__main__":
    main()
