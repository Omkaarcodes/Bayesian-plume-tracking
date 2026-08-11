"""
Animates one simulation run's CSV snapshots into a GIF: the gas plume
spreading through the house, with walls drawn on top.

Usage: python animate_run.py data/run_2026-08-11_19-05-46
"""

import sys
import glob
import re
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_run(run_dir):
    """Loads every snapshot CSV in a run folder, sorted by step number
    (not alphabetically -- 'snapshot_100' would otherwise sort before
    'snapshot_20' as plain text), plus the walls.csv for that run."""

    paths = glob.glob(f"{run_dir}/snapshot_*.csv")

    # Pull the integer step number out of each filename so we can sort
    # numerically. re.search finds the digits between "snapshot_" and ".csv".
    def step_number(path):
        match = re.search(r"snapshot_(\d+)\.csv", path)
        return int(match.group(1))

    paths.sort(key=step_number)

    grids = [np.loadtxt(p, delimiter=",") for p in paths]
    steps = [step_number(p) for p in paths]
    walls = np.loadtxt(f"{run_dir}/walls.csv", delimiter=",")

    return grids, steps, walls


def make_animation(grids, steps, walls, output_path="plume.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))

    # vmax fixes the color scale to the LAST frame's peak concentration,
    # not each frame's own peak. Without this, matplotlib would rescale
    # colors every frame, making the plume look like it's constantly at
    # "full intensity" even at step 0 -- fixing vmax is what makes the
    # growth actually visible over time.
    vmax = grids[-1].max()

    im = ax.imshow(grids[0], cmap="hot", vmin=0, vmax=vmax)

    # Walls drawn once, as a static overlay -- they don't change frame to
    # frame, so there's no need to redraw them inside update().
    wall_overlay = ma.masked_where(walls == 0, walls)
    ax.imshow(wall_overlay, cmap="Blues", vmin=0, vmax=1, alpha=0.9)

    title = ax.set_title(f"step {steps[0]}")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_index):
        # This function is called once per frame by FuncAnimation. It only
        # needs to update what actually changes
        im.set_data(grids[frame_index])
        title.set_text(f"step {steps[frame_index]}")
        return im, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(grids), interval=100, blit=False
    )

    anim.save(output_path, writer="pillow", fps=10)
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python animate_run.py <path_to_run_folder>")
        sys.exit(1)

    run_dir = sys.argv[1]
    grids, steps, walls = load_run(run_dir)
    print(f"Loaded {len(grids)} snapshots from {run_dir}")

    make_animation(grids, steps, walls)