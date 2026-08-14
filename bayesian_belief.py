import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares

SENSOR_NOISE_STD = 0.15
L_INIT = 10.0    # initial guess for falloff length scale, refined by calibration
K0_INIT = 0.02   # initial guess for amplitude growth rate, refined by calibration


def resolve_run_dir(path):
    if os.path.exists(os.path.join(path, "sensors.csv")):
        return path
    run_folders = glob.glob(os.path.join(path, "run_*"))
    if not run_folders:
        print(f"No run folders found inside '{path}'.")
        sys.exit(1)
    return max(run_folders, key=os.path.getmtime)


def load_data(run_dir):
    walls = np.loadtxt(f"{run_dir}/walls.csv", delimiter=",")
    sensor_pos = pd.read_csv(f"{run_dir}/sensor_positions.csv").set_index("sensor_id")
    sensors = pd.read_csv(f"{run_dir}/sensors.csv")
    return walls, sensor_pos, sensors


def calibrate_params(sensors, sensor_pos, walls):
    """Fits (x0, y0, K0, L) by nonlinear least squares against every logged
    sensor reading. x0, y0 are a byproduct MLE point-estimate of the source
    location."""

    merged = sensors.merge(sensor_pos, on="sensor_id")
    step = merged["step"].to_numpy()
    sx = merged["x"].to_numpy()
    sy = merged["y"].to_numpy()
    reading = merged["reading"].to_numpy()

    n = walls.shape[0]

    def residuals(params):
        x0, y0, k0, l = params
        dist = np.sqrt((sx - x0) ** 2 + (sy - y0) ** 2)
        predicted = k0 * step * np.exp(-dist / l)
        return predicted - reading

    init = [n / 2, n / 2, K0_INIT, L_INIT]
    bounds = ([0, 0, 1e-6, 1.0], [n, n, 10.0, n])
    result = least_squares(residuals, init, bounds=bounds)

    x0, y0, k0, l = result.x
    return {"x0": x0, "y0": y0, "K0": k0, "L": l}


def expected_reading(cand_x, cand_y, sx, sy, step, K0, L):
    dist = np.sqrt((cand_x - sx) ** 2 + (cand_y - sy) ** 2)
    K = K0 * step
    return K * np.exp(-dist / L)


def bayes_update(belief, cand_x, cand_y, sx, sy, reading, step, K0, L):
    expected = expected_reading(cand_x, cand_y, sx, sy, step, K0, L)
    likelihood = np.exp(-0.5 * ((reading - expected) / SENSOR_NOISE_STD) ** 2)
    belief = belief * likelihood
    belief /= belief.sum()
    return belief


def entropy(belief):
    p = belief[belief > 0]
    return -np.sum(p * np.log(p))


def run_belief_tracking(run_dir):
    walls, sensor_pos, sensors = load_data(run_dir)
    n = walls.shape[0]

    calib = calibrate_params(sensors, sensor_pos, walls)
    K0, L = calib["K0"], calib["L"]
    print(f"Calibrated: K0={K0:.5f}, L={L:.2f}  "
          f"(MLE source estimate: ({calib['x0']:.1f}, {calib['y0']:.1f}))")

    cand_x, cand_y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    belief = np.where(walls == 1, 0.0, 1.0)
    belief /= belief.sum()
    entropy_history = []
    for step, group in sensors.groupby("step"):
        if step % 15 != 0:   # subsample: consecutive readings are highly
            continue          # correlated, using all of them overcounts evidence
        for _, row in group.iterrows():
            sx = sensor_pos.loc[row.sensor_id, "x"]
            sy = sensor_pos.loc[row.sensor_id, "y"]
            belief = bayes_update(belief, cand_x, cand_y, sx, sy, row.reading, step, K0, L)
        entropy_history.append((step, entropy(belief)))
    return belief, entropy_history, walls, sensor_pos, calib


def plot_results(belief, entropy_history, walls, sensor_pos, calib, true_source=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: belief heatmap ---
    ax = axes[0]

    # Belief is usually extremely peaked after enough updates -- most cells
    # underflow to exactly 0.0 from repeated multiplication, and the
    # surviving nonzero cells can span 100+ orders of magnitude. A linear
    # color scale makes everything except the single peak pixel look
    # identical (near-black/purple). A log scale with a floor at max*1e-6
    # (instead of the true, absurdly tiny minimum) shows real structure in
    # the cells that still meaningfully compete with the peak.
    peak_val = belief.max()
    floor = peak_val * 1e-6
    display = np.where(belief > floor, belief, np.nan)  # NaN -> transparent/blank, not "0 on log scale" (undefined)

    im = ax.imshow(display, cmap="viridis", norm=LogNorm(vmin=floor, vmax=peak_val))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("probability source is at this cell (log scale)")

    wall_overlay = ma.masked_where(walls == 0, walls)
    ax.imshow(wall_overlay, cmap="Greys", vmin=0, vmax=1, alpha=0.8)
    ax.scatter(sensor_pos["y"], sensor_pos["x"], c="cyan", marker="^",
               s=80, edgecolors="black", label="sensors", zorder=3)

    peak = np.unravel_index(np.argmax(belief), belief.shape)
    ax.scatter([peak[1]], [peak[0]], c="yellow", marker="*", s=250,
               edgecolors="black", label="Bayes filter peak", zorder=4)
    ax.scatter([calib["y0"]], [calib["x0"]], c="orange", marker="D", s=80,
               edgecolors="black", label="MLE estimate (batch fit)", zorder=4)

    if true_source:
        ax.scatter([true_source[1]], [true_source[0]], c="red", marker="x",
                   s=120, linewidths=2.5, label="true source", zorder=4)
        error = np.sqrt((peak[0] - true_source[0]) ** 2 + (peak[1] - true_source[1]) ** 2)
        mle_error = np.sqrt((calib["x0"] - true_source[0]) ** 2 + (calib["y0"] - true_source[1]) ** 2)
        ax.set_xlabel(f"Bayes filter error: {error:.1f} cells   |   MLE error: {mle_error:.1f} cells")

    ax.set_title("Where the agent thinks the leak is\n(brighter = more probable)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- Right: entropy over time ---
    steps, ent_nats = zip(*entropy_history)
    ent_bits = np.array(ent_nats) / np.log(2)  # convert nats -> bits, more intuitive

    ax2 = axes[1]
    ax2.plot(steps, ent_bits, color="tab:blue", linewidth=2)
    ax2.axhline(ent_bits[0], color="gray", linestyle="--", linewidth=1,
                label=f"starting uncertainty ({ent_bits[0]:.1f} bits, uniform guess)")
    ax2.scatter([steps[-1]], [ent_bits[-1]], color="tab:blue", zorder=5)
    ax2.annotate(f"final: {ent_bits[-1]:.2f} bits",
                 xy=(steps[-1], ent_bits[-1]), xytext=(-90, 20),
                 textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"))

    ax2.set_xlabel("simulation step")
    ax2.set_ylabel("entropy of belief (bits)")
    ax2.set_title("How uncertain the agent is about the source location\n"
                  "(entropy falling = agent is getting more confident)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("belief_result.png", dpi=150)
    print("Saved belief_result.png")

    reduction_pct = 100 * (1 - ent_bits[-1] / ent_bits[0])
    print(f"Belief peak at: {peak}  (true source: {true_source})")
    print(f"Entropy: {ent_bits[0]:.2f} -> {ent_bits[-1]:.2f} bits "
          f"({reduction_pct:.0f}% reduction in uncertainty)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python belief_update.py <run_folder_or_data_dir>")
        sys.exit(1)
    run_dir = resolve_run_dir(sys.argv[1])
    belief, entropy_history, walls, sensor_pos, calib = run_belief_tracking(run_dir)
    plot_results(belief, entropy_history, walls, sensor_pos, calib, true_source=(35, 38))