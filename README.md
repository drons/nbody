# [N-body simulation program](https://en.wikipedia.org/wiki/N-body_simulation)
[![Build Status](https://travis-ci.org/drons/nbody.svg?branch=master)](https://travis-ci.org/drons/nbody)
[![Build Status](https://ci.appveyor.com/api/projects/status/vvttxq12cd39e81g/branch/master?svg=true)](https://ci.appveyor.com/project/drons/nbody/branch/master)
[![codecov](https://codecov.io/gh/drons/nbody/branch/master/graph/badge.svg)](https://codecov.io/gh/drons/nbody)
[![Coverity](https://scan.coverity.com/projects/9436/badge.svg)](https://scan.coverity.com/projects/drons-nbody)


## Features
### Integration methods
Method alias | Order | Description | Implicit | Embedded
-------------|-------|-------------|----------|----------
adams | up to 5 | [Adams–Bashforth method](https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods) |  :heavy_minus_sign: |  :heavy_minus_sign:
euler | 1 | [Classic Euler method](https://en.wikipedia.org/wiki/Euler_method) |  :heavy_minus_sign: |  :heavy_minus_sign:
rk4 | 4 | [Classic Runge-Kutta 4-order method](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Classic_fourth-order_method) |  :heavy_minus_sign: |  :heavy_minus_sign:
rk_butcher | - | [Runge-Kutta method with arbitrary Butcher tableu](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) |  :heavy_minus_sign: |  :heavy_minus_sign:
rkck | 5 | [Runge-Kutta-Cash–Karp 5-order method](https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method) |  :heavy_minus_sign: |  :star:
rkdp | 5 | [Runge-Kutta-Dormand–Prince 5-order method](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) |  :heavy_minus_sign: |  :star:
rkf | 7 | Runge-Kutta-Fehlberg 7-order method. See [1)](README.md#refs) p. 180 |  :heavy_minus_sign: |  :star:
rkgl | 6 | [Gauss–Legendre 6-order method](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Gauss%E2%80%93Legendre_methods) |  :star: |  :heavy_minus_sign:
rklc | 4 | [Runge-Kutta-Lobatto IIIC 4-order method](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Lobatto_IIIC_methods) |  :star: |  :heavy_minus_sign:
trapeze | 2 | [Trapeze method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) |  :star: |  :heavy_minus_sign:


### Compute engines
Engine alias | Approximate | Description
-------------|-------------|-------------
ah | :star:  | Single threaded engine with [Ahmad-Cohen](https://www.astronomyclub.xyz/time-steps/ahmadcohen-method.html) universe force simulation. See [2)](README.md#refs)
block |  :heavy_minus_sign: | Multi-threaded (OpenMP) engine with block-by-block force computation
opencl |  :heavy_minus_sign:  | Parallel OpenCL engine
openmp |  :heavy_minus_sign:  | Multi-threaded (OpenMP) engine
simple |  :heavy_minus_sign:  | Simple single threaded engine
simple_bh |  :star:  | Multi-threaded (OpenMP) engine with [Burnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) force simulation

### How to run
#### Simulation
To run n-body problem simulation use 'nbody-simulation' programm.

##### Simulation control
Argument | Description
---------|-------------
`--stars_count` | Stars count
`--box_size` | 'Universe' box size.
`--output` | Output stream name.
`--max_part_size` | Max stream file size (splits a stream into multiple files).
`--max_time` | Max simulation time.
`--dump_step` | Time step to dump simulation state to stream.
`--check_step` | Time step to verify the fundamental laws of physics. Conservation of impulse, angular momentum, energy.
`--verbose` | Print detailed simulation information.

##### Engine control arguments are:

Argument | Description
---------|-------------
`--engine` | Compute engine [type](#compute-engines).
`--distance_to_node_radius_ratio` | Simulation accuracy control for Burnes-Hut engine.
`--traverse_type` | Space tree traverse type for Burnes-Hut engine. Possible values are `cycle` or `nested_tree`.
`--tree_layout` | Space tree layout type for Burnes-Hut engine. Possible values are `tree` or `heap`.
`--full_recompute_rate` | Full force recompute rate in cucles (Ahmad-Cohen engine).
`--max_dist` | The maximum distance at which the force is calculated completely at each step  (Ahmad-Cohen engine).
`--min_force` | The minimum force of attraction at which it is calculated completely at each step (Ahmad-Cohen engine).
`--device` | Platforms/devices list for OpenCL based engines. Format: Platform1_ID:Device1,Device2;Platform2_ID:Device1,Device2... For example:  `--device=0:0,1` - first and second devices from first platform (with same context), `--device=0:0;0:1` - first and second devices from first platform (with separate contexts)

##### Solver control arguments are:

Argument | Description
---------|-------------
`--solver` | Solver [type](#integration-methods).
`--max_step`| Solvers max time step
`--min_step`| Embeded solvers min time step
`--rank`   | Adams–Bashforth solver rank (1...5).
`--refine_steps_count` | Refine step count for __implicit__ solvers.
`--error_threshold` | Step error threshold for __embeded__ solvers. If the error at the current step is greater than the threshold, then we decrease the time step and repeat the step.
`--max_recursion`   | Max recursion level for __embeded__ solvers.
`--substep_subdivisions` | Number of __embeded__ solver substeps into which the current step is divided at the next level of recursion when the error greater than `error_threshold`.

#### Player
To view simulation results run 'nbody-player' programm.
For now only one argument supported: `--input=<input stream name>`.
Other parameters controlled via UI.

### Gallery

[![Galaxy crash simulation](http://img.youtube.com/vi/AYzgTC0qqV0/0.jpg)](https://www.youtube.com/watch?v=AYzgTC0qqV0 "Galaxy crash simulation")


## Refs
1) [Hairer, Ernst; Nørsett, Syvert Paul; Wanner, Gerhard (1993), Solving ordinary differential equations I: Nonstiff problems, Berlin, New York](http://www.hds.bme.hu/~fhegedus/00%20-%20Numerics/B1993%20Solving%20Ordinary%20Differential%20Equations%20I%20-%20Nonstiff%20Problems.pdf)
2) [A Numerical Integration Scheme  for the N-Body Gravitational Problem	A. AHMAD AND L. COHEN 1973](https://courses.physics.ucsd.edu/2016/Winter/physics141/Lectures/Lecture8/AhmadCohen.pdf)
