import json, os
from utils.log import info, warning
from collections import defaultdict
from statistics import mean

MEMORY_PATH = os.path.join("data", "decision_memory.json")

# -----------------------------------------------------------------
# Load and Save
# -----------------------------------------------------------------
def load_memory():
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            info(f"🧠 Decision memory loaded ({len(data)} contexts).")
            return data
        except Exception as e:
            warning(f"Failed to load decision memory: {e}")
    return {}

def save_memory(memory):
    try:
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        info("💾 Decision memory saved.")
    except Exception as e:
        warning(f"Failed to save decision memory: {e}")

# -----------------------------------------------------------------
# Update memory based on training results
# -----------------------------------------------------------------
def remember_decision(phase, energy, decision, reward):
    """
    Store the average reward per (phase, energy bucket, decision)
    Example key: "mid_50_speed"
    """
    memory = load_memory()

    # Simplify energy into buckets
    energy_bucket = int(energy // 20) * 20  # e.g. 0, 20, 40, 60, 80
    key = f"{phase}_{energy_bucket}_{decision}"

    if key not in memory:
        memory[key] = {"count": 0, "avg_reward": 0}

    mem = memory[key]
    mem["count"] += 1
    mem["avg_reward"] = round((mem["avg_reward"] * (mem["count"] - 1) + reward) / mem["count"], 3)
    memory[key] = mem

    save_memory(memory)

# -----------------------------------------------------------------
# Retrieve memory-based bias
# -----------------------------------------------------------------
def get_memory_bias(phase, energy, decision):
    """
    Return a small bias value based on past rewards in similar contexts.
    """
    memory = load_memory()
    energy_bucket = int(energy // 20) * 20
    key = f"{phase}_{energy_bucket}_{decision}"

    if key in memory:
        bias = memory[key]["avg_reward"]
        return bias * 0.1  # scale down so it nudges decisions slightly
    return 0
