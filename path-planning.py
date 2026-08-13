import sys
import os
import glob
import heapq
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

from bayesian_belief import load_data, run_belief_tracking

START = (48, 24)  # just inside the front door


def resolve_run_dir(path):
    if os.path.exists(os.path.join(path, "sensors.csv")):
        return path
    run_folders = glob.glob(os.path.join(path, "run_*"))
    if not run_folders:
        print(f"No run folders found inside '{path}'.")
        sys.exit(1)
    return max(run_folders, key=os.path.getmtime)


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance


def neighbors(pos, walls):
    x, y = pos
    n = walls.shape[0]
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and walls[nx, ny] == 0:
            yield (nx, ny)


def astar(walls, start, goal):
    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nxt in neighbors(current, walls):
            tentative_g = g + 1
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_set, (f, tentative_g, nxt))

    return None  # no path found


def plot_path(walls, path, start, goal, true_source=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.zeros_like(walls), cmap="Greys", vmin=0, vmax=1)
    wall_overlay = ma.masked_where(walls == 0, walls)
    ax.imshow(wall_overlay, cmap="Blues", vmin=0, vmax=1, alpha=0.9)

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(ys, xs, color="orange", linewidth=2, label=f"A* path ({len(path)} steps)")

    ax.scatter([start[1]], [start[0]], c="lime", marker="o", s=100,
               edgecolors="black", label="agent start", zorder=5)
    ax.scatter([goal[1]], [goal[0]], c="yellow", marker="*", s=200,
               edgecolors="black", label="target (belief peak)", zorder=5)
    if true_source:
        ax.scatter([true_source[1]], [true_source[0]], c="red", marker="x",
                   s=100, linewidths=2, label="true source", zorder=5)

    ax.set_title("A* path from agent start to current belief peak")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("path_result.png", dpi=150)
    print("Saved path_result.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python path_planning.py <run_folder_or_data_dir>")
        sys.exit(1)

    run_dir = resolve_run_dir(sys.argv[1])
    walls, sensor_pos, sensors = load_data(run_dir)
    belief, entropy_history, walls, sensor_pos = run_belief_tracking(run_dir)

    goal = tuple(int(v) for v in np.unravel_index(np.argmax(belief), belief.shape))
    print(f"Target (belief peak): {goal}")

    path = astar(walls, START, goal)
    if path is None:
        print("No path found!")
        sys.exit(1)
    print(f"Path found: {len(path)} steps")

    plot_path(walls, path, START, goal, true_source=(35, 38))