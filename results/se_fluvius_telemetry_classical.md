# Real-time envelopes from sparse meters, case33bw (fluvius states, placement telemetry (30% of load buses, seed 0): 2000 test / 600 validation states)

Meters (1% noise): voltage at buses [0, 1, 2, 3, 6, 8, 9, 13, 16, 20, 27]; P/Q at buses [0, 1, 2, 3, 6, 8, 9, 13, 16, 20, 27]; feeder-head flow. Envelope: largest proportional export / import level keeping the estimated state within limits, applied to the true state. 'At <= 5% unsafe': scaled by kappa calibrated on validation; 'kept' = issued envelope / perfect-information envelope, summed over test states.

| Method | Vm MAE (pu) | Load P MAE (kW/bus) | Export raw unsafe | Import raw unsafe | Export kept at <= 5% unsafe | Import kept at <= 5% unsafe (unsafe) | ms/state |
|---|---|---|---|---|---|---|---|
| forecast only | 0.0044 | 13.6 | 38.3% | 35.9% | 81.4% | 37.7% (5.3%) | 0.00 |
| WLS, independent prior (std 0.5x) | 0.0009 | 8.0 | 25.4% | 34.2% | 93.9% | 75.0% (3.8%) | 1.83 |
| WLS, correlated prior | 0.0004 | 4.5 | 36.0% | 33.4% | 95.5% | 87.4% (3.2%) | 1.94 |
| perfect information | 0.0000 | 0.0 | 0.0% | 0.0% | 100.0% | 100.0% (0.0%) | 0.00 |
