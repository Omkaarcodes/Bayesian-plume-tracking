import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma

SENSOR_NOISE_STD = 0.15
L = 10.0     # falloff length scale, in grid cells
K0 = 0.02    # amplitude growth rate per step -- calibrated so predicted
# readings land near actual observed magnitudes late in the run
# (room is sealed, so total gas mass grows ~linearly with time,
# confirmed on Day 1 -- so we scale K linearly with step too,
# rather than using one fixed K that assumes a steady state
# this sealed room never actually reaches)


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


def expected_reading(cand_x, cand_y, sx, sy, step):
    dist = np.sqrt((cand_x - sx) ** 2 + (cand_y - sy) ** 2)
    K = K0 * step
    return K * np.exp(-dist / L)


def bayes_update(belief, cand_x, cand_y, sx, sy, reading, step):
    expected = expected_reading(cand_x, cand_y, sx, sy, step)
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
            belief = bayes_update(belief, cand_x, cand_y, sx, sy, row.reading, step)
        entropy_history.append((step, entropy(belief)))
    return belief, entropy_history, walls, sensor_pos


def plot_results(belief, entropy_history, walls, sensor_pos, true_source=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: belief heatmap ---
    ax = axes[0]
    im = ax.imshow(belief, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("probability source is at this cell")

    wall_overlay = ma.masked_where(walls == 0, walls)
    ax.imshow(wall_overlay, cmap="Greys", vmin=0, vmax=1, alpha=0.8)
    ax.scatter(sensor_pos["y"], sensor_pos["x"], c="cyan", marker="^",
               s=80, edgecolors="black", label="sensors", zorder=3)

    peak = np.unravel_index(np.argmax(belief), belief.shape)
    ax.scatter([peak[1]], [peak[0]], c="yellow", marker="*", s=250,
               edgecolors="black", label="belief's best guess", zorder=4)

    if true_source:
        ax.scatter([true_source[1]], [true_source[0]], c="red", marker="x",
                   s=120, linewidths=2.5, label="true source", zorder=4)
        error = np.sqrt((peak[0] - true_source[0]) ** 2 + (peak[1] - true_source[1]) ** 2)
        ax.set_xlabel(f"Localization error: {error:.1f} grid cells")

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
    belief, entropy_history, walls, sensor_pos = run_belief_tracking(run_dir)
    plot_results(belief, entropy_history, walls, sensor_pos, true_source=(35, 38))