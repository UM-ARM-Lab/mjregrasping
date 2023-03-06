# MJRegrasping

Experimenting with using mujoco from C++.

TODO:

- [x] benchmark
- [x] fix joint control. The current joints are controlled with very weak position servos. Instead, we should control them with intvelocity actuators, then wrap that in a controller in C++ that takes position commands and sets the target for the invelocity controller.
- [ ] Write a C++ library that registers the custom callback and takes in the `mjmodel` and `mjdata`
- [ ] Demonstrate using this library from python, using the mujoco python bindings, but not `dm_control`

## Benchmarking

- simulating 200 steps in a loop once takes ~12ms
- simulating 200 steps in a loop 4 times takes ~50ms
- simulating 200 steps in a loop 100 times takes ~1.35 seconds
- simulating 200 steps in a loop 1000 times takes ~17 seconds
- simulating 200 steps 4 times in a threadpool takes