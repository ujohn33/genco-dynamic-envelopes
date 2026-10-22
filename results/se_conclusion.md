# State estimation for real-time envelopes: conclusion (exploratory)

Setting: case33bw, each load bus a distribution transformer serving real Fluvius 2024 homes (3,715 homes;
30% heat pump, 40% PV) plus 1,083 EVs from a session generator (time-varying Poisson arrivals, Weibull
survival plug-in durations) calibrated to Fluvius EV households. One state per 15-min step of 2024,
days split train / validation / test. Envelopes at <= 5% unsafe (scale factor calibrated on validation
days), 2,000 test states, 1% meter noise.

## Envelope kept relative to perfect information

| Estimator | 30% transformers monitored: export / import | Feeder-end voltages only: export / import |
|---|---|---|
| Day-ahead forecast only | 81% / 38% | 81% / 38% |
| WLS, independent prior | 94% / 75% | 94% / 70% |
| **WLS, correlated prior (Bayesian)** | **96% / 87%** | **93% / 85%** |
| GENCO SE, meters only (60 epochs) | 74% / 69% | - |
| GENCO SE, meters + forecast (200 epochs) | 72% / 34% | 2% / 26% |
| GENCO + WLS (GENCO voltages as pseudo-measurements) | 95% / 82% | 95% / 69% |

## Findings

1. Real-time estimation is worth it: with 30% of transformers monitored, correlated-prior WLS more
   than doubles the safe import envelope compared with the day-ahead forecast (87% vs 38%).
2. Classical Bayesian WLS (power-flow physics + the covariance of historical forecast errors) is the
   best estimator here, at about 2-3 ms per state.
3. GENCO in StateEstimation mode (gridfm-graphkit 0.9, trained from scratch on 21k training-day states,
   A100) loses to it, also when given the same information (the forecast at unmetered buses):
   - with forecasts its load error (8.3 kW per bus) is close to independent-prior WLS (8.0), but its
     estimates lean towards too much load, so its envelopes are rarely unsafe but much too small;
   - it does not transfer to a metering layout it was not trained on (feeder-end voltages only);
   - its voltages are ~10x less accurate than WLS's (0.004 vs 0.0004 pu), so adding them to WLS
     as pseudo-measurements makes WLS worse, not better.
4. On a balanced MV feeder with a good network model, state estimation is a well-posed, nearly
   linear-Gaussian problem where classical Bayesian estimation is hard to beat. A learned estimator
   would be needed where the model or the prior is poor (unknown topology or impedances, strongly
   non-Gaussian behaviour, very sparse data), which this study did not test.

Runs: GENCO SE meters-only Hydra job 13361018; with forecasts 13361663 (200 epochs, 2 h 18 min, A100).
Tables: se_fluvius_*_classical.md, se_fluvius_telemetry_genco_ep60.md, se_fluvius_*_genco_forecast.md.
