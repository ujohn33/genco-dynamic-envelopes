# Real-time envelopes from sparse meters, case33bw (fluvius states, placement feeder_ends: 2000 test / 600 validation states)

Meters (1% noise): voltage at buses [0, 17, 21, 24, 32]; P/Q at buses [0]; feeder-head flow. Envelope: largest proportional export / import level keeping the estimated state within limits, applied to the true state. 'At <= 5% unsafe': scaled by kappa calibrated on validation; 'kept' = issued envelope / perfect-information envelope, summed over test states.

| Method | Vm MAE (pu) | Load P MAE (kW/bus) | Export raw unsafe | Import raw unsafe | Export kept at <= 5% unsafe | Import kept at <= 5% unsafe (unsafe) | ms/state |
|---|---|---|---|---|---|---|---|
| forecast only | 0.0044 | 13.6 | 38.3% | 35.9% | 81.4% | 37.7% (5.3%) | 0.00 |
| WLS, independent prior (std 0.25x) | 0.0012 | 10.6 | 24.4% | 33.6% | 93.7% | 70.2% (3.9%) | 1.90 |
| WLS, correlated prior | 0.0005 | 6.3 | 37.9% | 33.5% | 93.1% | 85.0% (3.8%) | 1.90 |
| perfect information | 0.0000 | 0.0 | 0.0% | 0.0% | 100.0% | 100.0% (0.0%) | 0.00 |
