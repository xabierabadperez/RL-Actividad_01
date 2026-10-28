"""
A.2 (Xabier) : Se lanzan los 4 estudios en secuencia. Uso: python run_phase_a_2.py

"""
import subprocess
import sys
from pathlib import Path

# Persona que ejecuta: decide en qué base de datos se guardarán los resultados
# (2 -> optuna/phase_a_2.db)
OWNER = "2"
STUDIES = [
    ("esarsa", "low"),
    ("esarsa", "medium"),
    ("esarsa", "high"),
    ("qlearning", "high"),
]

# Ruta completa de optuna_study.py
SCRIPT = Path(__file__).resolve().parent / "optuna_study.py"

# Recorre mis estudios uno detrás de otro (cada uno son 30 trials)
for algo, budget in STUDIES:
    print(f"=== {algo} / {budget} ===", flush=True)
    # Equivale a ejecutar en la terminal:
    #  python optuna_study.py --algo <algo> --budget <budget> --owner 2
    subprocess.run(
        [sys.executable, str(SCRIPT), "--algo", algo, "--budget", budget, "--owner", OWNER],
        check=True,
    )