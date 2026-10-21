# Real-time phase switching on ENWL1 with Fluvius 2024 households

Exploratory. 24 interval-directions on 2024-06-28, 2024-12-15 (hours 3, 8, 12, 13, 14, 18, 20, 3, 8, 12, 18, 20). 104 heat-pump / EV homes have a phase-switching device; 200 design and 200 held-out scenarios per interval; design rule: >= 98% of design scenarios within all four-wire limits (OpenDSS). Uniform envelope per home, caps 7.4 kW import / 5.0 kW export.

## Headline

| | Fixed phases | Balance heuristic | Sweep-guided search |
|---|---|---|---|
| Mean import envelope (kW per home) | 0.48 | 0.60 | 0.62 |
| Mean export envelope (kW per home) | 0.20 | 0.37 | 0.40 |
| Intervals with envelope > 0 | 10 / 24 | 13 / 24 | 12 / 24 |
| Mean design share within limits (OpenDSS) | 76.3% | 87.9% | 88.5% |
| Mean held-out share within limits | 77.0% | 86.9% | 87.0% |
| Mean switches per interval | 0 | 68 | 63 |
| Mean decision time per interval-direction (s, laptop CPU) | 1.5 | 1.8 | 26.7 |

## Does the search beat the heuristic?

Clearly better = envelope larger by > 0.1 kW, or, where both envelopes are 0, design share higher by > 5% points.

| Comparison | Intervals clearly better | clearly worse |
|---|---|---|
| balance vs fixed | 20 / 24 | 0 / 24 |
| search vs fixed | 18 / 24 | 0 / 24 |
| **search vs balance** | **6 / 24** | 2 / 24 |

## Per interval

| Day | Hour | Direction | Fixed kW (design share) | Balance kW (share, switches) | Search kW (share, switches) |
|---|---|---|---|---|---|
| 2024-06-28 | 03:00 | export | 1.15 (98.0%) | 1.16 (98.5%, 77) | 1.23 (98.0%, 80) |
| 2024-06-28 | 03:00 | import | 0.39 (98.0%) | 0.82 (98.0%, 65) | 0.83 (98.0%, 67) |
| 2024-06-28 | 08:00 | export | 0.00 (97.0%) | 0.31 (98.0%, 70) | 0.49 (98.0%, 79) |
| 2024-06-28 | 08:00 | import | 1.00 (98.0%) | 1.05 (98.0%, 75) | 1.13 (98.0%, 74) |
| 2024-06-28 | 12:00 | export | 0.00 (7.0%) | 0.00 (56.0%, 73) | 0.00 (68.0%, 65) |
| 2024-06-28 | 12:00 | import | 1.09 (98.0%) | 1.27 (98.0%, 71) | 1.41 (98.0%, 70) |
| 2024-06-28 | 13:00 | export | 0.00 (3.0%) | 0.00 (72.0%, 51) | 0.00 (78.5%, 52) |
| 2024-06-28 | 13:00 | import | 1.14 (98.0%) | 1.28 (98.0%, 74) | 1.48 (98.0%, 75) |
| 2024-06-28 | 14:00 | export | 0.00 (10.0%) | 0.00 (52.5%, 54) | 0.00 (56.5%, 62) |
| 2024-06-28 | 14:00 | import | 1.11 (98.0%) | 1.28 (98.0%, 73) | 1.08 (98.0%, 72) |
| 2024-06-28 | 18:00 | export | 0.17 (98.0%) | 0.62 (98.0%, 67) | 0.67 (98.0%, 67) |
| 2024-06-28 | 18:00 | import | 0.74 (98.0%) | 0.84 (98.0%, 68) | 0.76 (98.0%, 75) |
| 2024-06-28 | 20:00 | export | 1.06 (98.5%) | 1.12 (98.0%, 65) | 1.12 (98.0%, 72) |
| 2024-06-28 | 20:00 | import | 0.27 (98.5%) | 0.65 (98.5%, 71) | 0.70 (98.0%, 70) |
| 2024-12-15 | 03:00 | export | 0.00 (65.5%) | 0.00 (80.5%, 73) | 0.00 (82.0%, 12) |
| 2024-12-15 | 03:00 | import | 0.00 (53.0%) | 0.00 (66.5%, 60) | 0.00 (63.5%, 64) |
| 2024-12-15 | 08:00 | export | 0.00 (96.0%) | 1.22 (98.0%, 70) | 1.28 (98.0%, 22) |
| 2024-12-15 | 08:00 | import | 0.00 (92.5%) | 0.03 (98.0%, 73) | 0.00 (97.5%, 70) |
| 2024-12-15 | 12:00 | export | 0.00 (82.0%) | 0.00 (89.5%, 62) | 0.00 (89.5%, 74) |
| 2024-12-15 | 12:00 | import | 0.00 (77.5%) | 0.00 (83.5%, 70) | 0.00 (93.0%, 65) |
| 2024-12-15 | 18:00 | export | 0.00 (65.0%) | 0.00 (80.5%, 67) | 0.00 (80.0%, 75) |
| 2024-12-15 | 18:00 | import | 0.00 (60.5%) | 0.00 (77.0%, 66) | 0.00 (63.0%, 24) |
| 2024-12-15 | 20:00 | export | 0.00 (73.0%) | 0.00 (91.5%, 69) | 0.00 (90.0%, 68) |
| 2024-12-15 | 20:00 | import | 0.00 (68.5%) | 0.00 (85.0%, 63) | 0.00 (87.5%, 59) |
