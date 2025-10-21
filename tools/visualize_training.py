import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd
import numpy as np

# ✅ Updated path
LOG_DIR = os.path.join("data", "training_logs")

# -------------------------------------------------------------
# Load the latest training log file
# -------------------------------------------------------------
def load_latest_records():
    if not os.path.exists(LOG_DIR):
        print("⚠️ Log folder not found:", LOG_DIR)
        return []

    files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]
    if not files:
        print("⚠️ No training log files found in", LOG_DIR)
        return []

    # Get the latest log file by modification time
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)))
    latest_path = os.path.join(LOG_DIR, latest_file)

    print(f"📖 Loading latest log file: {latest_file}")

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to read log file {latest_file}: {e}")
        return []


# -------------------------------------------------------------
# Export training data to CSV
# -------------------------------------------------------------
def export_to_csv(records):
    if not records:
        print("⚠️ No records to export.")
        return
    df = pd.DataFrame(records)
    os.makedirs("data/exports", exist_ok=True)
    output_path = os.path.join("data", "exports", "training_log_export.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Exported training log to CSV: {output_path}")


# -------------------------------------------------------------
# Visualize average reward per stat
# -------------------------------------------------------------
def visualize_stat_rewards(records):
    if not records:
        print("⚠️ No data for reward visualization.")
        return
    df = pd.DataFrame(records)
    if "decision" not in df or "reward" not in df:
        return
    avg = df.groupby("decision")["reward"].mean()
    avg.plot(kind="bar", color="skyblue", title="Average Reward per Training Type")
    plt.ylabel("Average Reward")
    plt.show()


# -------------------------------------------------------------
# Main visualization (energy, stats, reward trends)
# -------------------------------------------------------------
def visualize(records):
    if not records:
        print("⚠️ No records found.")
        return

    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
    energies = [r.get("energy", 0) for r in records]
    rewards = [r.get("reward", 0) for r in records]
    avg_stats = [
        sum(r.get("current_stats", {}).values()) / len(r.get("current_stats", {}) or [1])
        for r in records
    ]

    if len(rewards) >= 5:
        rewards_smoothed = np.convolve(rewards, np.ones(5) / 5, mode="valid")
    else:
        rewards_smoothed = rewards

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, energies, label="Energy", color="skyblue")
    plt.legend(); plt.ylabel("Energy")

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, avg_stats, label="Average Stats", color="orange")
    plt.legend(); plt.ylabel("Stats")

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, rewards, label="Reward", color="green", alpha=0.4)
    plt.plot(timestamps[-len(rewards_smoothed):], rewards_smoothed, color="darkgreen", label="Smoothed Reward")
    plt.legend(); plt.ylabel("Reward"); plt.xlabel("Time")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Live visualization mode (auto-refresh)
# -------------------------------------------------------------
def live_visualization(refresh_interval=30):
    """Auto-refresh visualization whenever a log file updates."""
    print("📊 Live visualization started. Press Ctrl + C to stop.")
    plt.ion()

    last_file = None
    last_size = 0

    while True:
        try:
            # Detect latest file
            files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]
            if not files:
                print("⚠️ No log file found yet.")
                time.sleep(refresh_interval)
                continue

            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)))
            latest_path = os.path.join(LOG_DIR, latest_file)
            new_size = os.path.getsize(latest_path)

            # Only reload if file changed or switched
            if latest_file != last_file or new_size != last_size:
                last_file, last_size = latest_file, new_size

                plt.clf()
                records = load_latest_records()
                if not records:
                    time.sleep(refresh_interval)
                    continue

                timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
                energies = [r.get("energy", 0) for r in records]
                rewards = [r.get("reward", 0) for r in records]
                avg_stats = [
                    sum(r.get("current_stats", {}).values()) / len(r.get("current_stats", {}) or [1])
                    for r in records
                ]

                if len(rewards) >= 5:
                    rewards_smoothed = np.convolve(rewards, np.ones(5) / 5, mode="valid")
                else:
                    rewards_smoothed = rewards

                plt.subplot(3, 1, 1)
                plt.plot(timestamps, energies, color="skyblue")
                plt.ylabel("Energy")

                plt.subplot(3, 1, 2)
                plt.plot(timestamps, avg_stats, color="orange")
                plt.ylabel("Avg Stats")

                plt.subplot(3, 1, 3)
                plt.plot(timestamps, rewards, color="green", alpha=0.4)
                plt.plot(timestamps[-len(rewards_smoothed):], rewards_smoothed, color="darkgreen")
                plt.ylabel("Reward"); plt.xlabel("Time")

                plt.tight_layout()
                plt.pause(0.5)

            time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n🛑 Visualization stopped by user.")
            break
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
            time.sleep(refresh_interval)


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    records = load_latest_records()
    export_to_csv(records)
    visualize_stat_rewards(records)
    visualize(records)
    live_visualization(refresh_interval=30)
