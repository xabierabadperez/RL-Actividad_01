# Mejores configuraciones por algoritmo y presupuesto

| Algoritmo | Presupuesto | n_bins | Estados | Mejor reward | Pasos eval. | learning_rate | lr_decay | lr_episodes_decay | discount_rate | epsilon |
|---|---|---|---|---|---|---|---|---|---|---|
| SARSA | low | 3 | 6.561 | 9.99 | 165 | 0.0328 | 1.00 | 500 | 0.93 | 0.05 |
| SARSA | medium | 5 | 390.625 | 130.74 | 304 | 0.1023 | 0.88 | 1000 | 0.95 | 0.00 |
| SARSA | high | 7 | 5.764.801 | 112.29 | 379 | 0.4640 | 1.00 | 1000 | 0.97 | 0.00 |
| Expected SARSA | low | 3 | 6.561 | 18.13 | 166 | 0.1164 | 0.88 | 100 | 0.96 | 0.05 |
| Expected SARSA | medium | 5 | 390.625 | 132.41 | 307 | 0.0725 | 0.82 | 500 | 0.93 | 0.00 |
| Expected SARSA | high | 7 | 5.764.801 | 126.78 | 365 | 0.2179 | 0.92 | 100 | 0.93 | 0.00 |
| Q-Learning | low | 3 | 6.561 | 10.87 | 161 | 0.2053 | 0.94 | 100 | 0.95 | 0.15 |
| Q-Learning | medium | 5 | 390.625 | 148.55 | 317 | 0.0805 | 0.98 | 100 | 0.96 | 0.00 |
| Q-Learning | high | 7 | 5.764.801 | 96.48 | 375 | 0.4555 | 0.92 | 100 | 0.95 | 0.00 |
