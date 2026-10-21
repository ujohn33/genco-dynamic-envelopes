# Real-time envelopes from sparse meters, case33bw (fluvius states, placement telemetry (30% of load buses, seed 0): 2000 test / 600 validation states)

Meters (1% noise): voltage at buses [0, 1, 2, 3, 6, 8, 9, 13, 16, 20, 27]; P/Q at buses [0, 1, 2, 3, 6, 8, 9, 13, 16, 20, 27]; feeder-head flow. Envelope: largest proportional export / import level keeping the estimated state within limits, applied to the true state. 'At <= 5% unsafe': scaled by kappa calibrated on validation; 'kept' = issued envelope / perfect-information envelope, summed over test states.

| Method | Vm MAE (pu) | Load P MAE (kW/bus) | Export raw unsafe | Import raw unsafe | Export kept at <= 5% unsafe | Import kept at <= 5% unsafe (unsafe) | ms/state |
|---|---|---|---|---|---|---|---|
| forecast only | 0.0044 | 13.6 | 38.3% | 35.9% | 81.4% | 37.7% (5.3%) | 0.00 |
| WLS, independent prior (std 0.5x) | 0.0009 | 8.0 | 25.4% | 34.2% | 93.9% | 75.0% (3.8%) | 2.12 |
| WLS, correlated prior | 0.0004 | 4.5 | 36.0% | 33.4% | 95.5% | 87.4% (3.2%) | 2.33 |
| GENCO SE | 0.0094 | 34.8 | 0.5% | 16.4% | 74.3% | 68.7% (5.4%) | 15.16 |
| GENCO + WLS (correlated prior) | 0.0004 | 4.5 | 35.0% | 36.8% | 95.4% | 80.2% (3.6%) | 2.20 |
| perfect information | 0.0000 | 0.0 | 0.0% | 0.0% | 100.0% | 100.0% (0.0%) | 0.00 |

GENCO SE: gridfm-graphkit 0.9.0.3 GNS_heterogeneous (hidden 48, 12 layers), StateEstimation task, trained from scratch on the 21,024 training-day states for 60 epochs on one A100 (41 min; Hydra job 13361018). Validation Vm MAE plateaued near 0.9 pu for 44 epochs, then fell to 0.0135 pu by epoch 60 and was still falling: the model is not converged. GENCO gets no forecast input (graphkit SE only takes measured/unmeasured quantities); WLS uses the day-ahead forecast and, for the correlated prior, training-day error statistics.
