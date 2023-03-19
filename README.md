# MJRegrasping

Experimenting with using mujoco from C++.

TODO:

- [x] benchmark
- [x] fix joint control. The current joints are controlled with very weak position servos. Instead, we should control them with intvelocity actuators, then wrap that in a controller in C++ that takes position commands and sets the target for the invelocity controller.
- [ ] visualize a set of rollouts in rviz
- [ ] implement MPPI in C++?
- [ ] re-implement the test_muj

## Benchmarking

- simulating 200 steps in a loop once takes ~12ms
- simulating 200 steps in a loop 4 times takes ~50ms
- simulating 200 steps in a loop 100 times takes ~1.35 seconds
- simulating 200 steps in a loop 1000 times takes ~17 seconds

On my 8-core laptop (I think? could be wrong.)

| n samples | serial (s) | parallel (s) |
| --------- | ---------- | ------------ |
| 1 | 0.017 | 0.013 |
| 4 | 0.060 | 0.021 |
| 8 | 0.124 | 0.026 |
| 80 | 1.303 | 0.259 |
| 800 | 12.808 | 3.533 |

I tested on Odin, and we achieve ~50 parallel rollouts. I'm not sure 50 is enough for MPPI, we might still need "batches" where if we want 200 rollouts we do them with a threadpool with 50 workers.