

import sys
import glob
import re
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def prep_run(run_dir):


    paths = glob.glob(f"{run_dir}/snapshot_*.csv")

    def step_num(path):
        match = re.search(r"snapshot_(\d+)\.csv", path)
        return int(match.group(1))

    paths.sort(key=step_num)

    grids = [np.loadtxt(p, delimiter=",") for p in paths]
    steps = [step_num(p) for p in paths]
    walls = np.loadtxt(f"{run_dir}/walls.csv", delimiter=",")

    return grids, steps, walls


def animate(grids, steps, walls, output_path="plume.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))

    vmax = grids[-1].max()

    im = ax.imshow(grids[0], cmap="hot", vmin=0, vmax=vmax)

    # Walls drawn once, as a static overlay -- they don't change frame to
    # frame.
    wall_overlay = ma.masked_where(walls == 0, walls)
    ax.imshow(wall_overlay, cmap="Blues", vmin=0, vmax=1, alpha=0.9)

    title = ax.set_title(f"step {steps[0]}")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_index):
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