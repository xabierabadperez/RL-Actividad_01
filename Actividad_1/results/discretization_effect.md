# Efecto del presupuesto de discretización

| Algoritmo | Presupuesto | Estados | Mejor | Mediana | Q1 | Q3 | Trials > 0 (de 30) |
|---|---|---|---|---|---|---|---|
| SARSA | low | 6.561 | 10.0 | -78.4 | -135.9 | -33.6 | 2 |
| SARSA | medium | 390.625 | 130.7 | -46.2 | -139.0 | 64.7 | 11 |
| SARSA | high | 5.764.801 | 112.3 | -56.1 | -178.4 | 35.5 | 10 |
| Expected SARSA | low | 6.561 | 18.1 | -58.4 | -145.5 | -27.5 | 2 |
| Expected SARSA | medium | 390.625 | 132.4 | -2.1 | -121.1 | 68.0 | 14 |
| Expected SARSA | high | 5.764.801 | 126.8 | -112.8 | -207.0 | -19.6 | 6 |
| Q-Learning | low | 6.561 | 10.9 | -58.7 | -122.1 | -36.4 | 2 |
| Q-Learning | medium | 390.625 | 148.5 | -19.1 | -75.6 | 49.7 | 13 |
| Q-Learning | high | 5.764.801 | 96.5 | -43.2 | -152.6 | 10.6 | 9 |
