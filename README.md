# Confined-Space Gas Leak Localization

Simulates a gas leak in a sealed house using a finite-difference solver for
the 2D diffusion PDE, then localizes the source from noisy fixed sensors
via a discrete Bayesian filter, and plans a route to the estimated source
with A* pathfinding.

## Motivation

Diffusion is a poor model for open-air gas dispersion, since wind and
turbulence dominate. In a confined space, a sealed room or house, it's a
reasonable model: airflow is constrained and walls genuinely matter. The
scenario here is a stove-style leak in a kitchen, with sensors elsewhere
in the house trying to localize it before it becomes obvious.

## Day 1: Diffusion simulation (C++)

A 50x50 grid models concentration spreading via `∂C/∂t = D∇²C`, using a
5-point Laplacian stencil in space and explicit Euler in time.

The house has a central hallway, four rooms, and a front door. Walls use a
reflecting (zero-flux) boundary condition: gas bounces back instead of
disappearing, confirmed by checking total mass grows linearly with time
rather than leaking away.

Stability requires `D*dt/dx² <= 0.25` (a CFL condition). Violating this
makes the simulation diverge instead of just losing accuracy.

## Day 2: Sensors (C++)

Five fixed sensors (one per room, one in the hallway) sample true
concentration each step and add Gaussian noise: `reading = true + N(0, σ²)`.

Sensors near the kitchen show a clear rising signal. Far rooms stay flat
near the noise floor for most of the run, showing real detection lag
caused by the house's doorway layout.

## Day 3: Bayesian localization (Python prototype)

A discrete Bayes filter tracks a probability distribution over every open
cell being the true source.

Forward model: predicted reading is `K0 * step * exp(-distance / L)`. The
amplitude scales with time because the sealed room never reaches a steady
state (confirmed Day 1), so a fixed amplitude would underpredict late
readings.

Update: `belief *= Gaussian_likelihood(reading | expected)`, renormalized.

`K0` and `L` are fit via nonlinear least squares against all logged
readings rather than hand-tuned, which also yields an independent MLE
point-estimate of the source as a cross-check against the filter's peak.

Entropy, `H = -Σ p log(p)`, tracks how uncertain the filter still is.

Known limitations: the likelihood model uses straight-line distance and
ignores walls. With one dominant sensor, distance alone constrains
candidates to an arc, not a point. Consecutive readings are correlated, so
updates are subsampled every 15 steps to avoid overcounting evidence.

## Day 4: A* path planning (Python prototype)

Standard A* (Manhattan heuristic, 4-connected grid) finds the shortest
open-cell route from the front door to the current belief peak, correctly
navigating doorways and never crossing a wall.

## How to run

C++ simulation:
```
g++ -std=c++17 -O2 -Wall -o diffusion diffusion.cpp
./diffusion
```
Writes a timestamped run folder under `data/` with grid snapshots,
`sensors.csv`, `sensor_positions.csv`, and `walls.csv`.

Python analysis:
```
pip install numpy pandas matplotlib scipy pillow
python animate_run.py data
python plot_sensors.py data
python belief_update.py data
python path_planning.py data
```
Each script auto-detects the most recent run folder if pointed at `data/`.

Days 1-4 are validated. Day 5 (live C++ integration of the belief filter
and A*) has an initial implementation still being tested. Day 6 (baseline
comparison and final combined animation) has not started.
