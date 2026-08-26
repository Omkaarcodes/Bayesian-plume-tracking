import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


ROOMS = {
    0: "bedroom",
    1: "living room",
    2: "bathroom",
    3: "kitchen (source)",
    4: "hallway",
}


def resolve_run_dir(path):
    if os.path.exists(os.path.join(path, "sensors.csv")):
        return path
    run_folders = glob.glob(os.path.join(path, "run_*"))
    if not run_folders:
        print(f"No run folders found inside '{path}'.")
        sys.exit(1)
    return max(run_folders, key=os.path.getmtime)


def plot_sensors(run_dir):
    df = pd.read_csv(f"{run_dir}/sensors.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for sid, group in df.groupby("sensor_id"):
        label = ROOM_NAMES.get(sid, f"sensor {sid}")
        ax.plot(group["step"], group["reading"], label=label)

    ax.set_xlabel("step")
    ax.set_ylabel("sensor reading (noisy)")
    ax.set_title("Sensor readings over time")
    ax.legend()
    plt.tight_layout()

    out_path = "sensor_readings.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_sensors.py <run_folder_or_data_dir>")
        sys.exit(1)

    run_dir = resolve_run_dir(sys.argv[1])
    plot_sensors(run_dir)