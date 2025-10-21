import json, os
from statistics import mean
from utils.log import info, warning
from collections import defaultdict

# File paths
DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "training_logs")
BRAIN_PATH = os.path.join(DATA_DIR, "brain.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")

# -------------------------------------------------------------
# Load brain.json (persistent learning)
# -------------------------------------------------------------
def load_brain():
    if os.path.exists(BRAIN_PATH):
        try:
            with open(BRAIN_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            info(f"🧠 Loaded brain data with {len(data)} entries.")
            return data
        except Exception as e:
            warning(f"Failed to load brain: {e}")
    return {}

# -------------------------------------------------------------
# Save brain.json
# -------------------------------------------------------------
def save_brain(data):
    try:
        os.makedirs(os.path.dirname(BRAIN_PATH), exist_ok=True)
        with open(BRAIN_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        info("💾 Brain updated and saved.")
    except Exception as e:
        warning(f"Failed to save brain: {e}")

# -------------------------------------------------------------
# Save summary.json
# -------------------------------------------------------------
def save_summary(averages, character="Unknown"):
    try:
        summary = {
            "character": character,
            "averages": averages,
        }
        os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        info("📊 Summary report saved.")
    except Exception as e:
        warning(f"Failed to save summary report: {e}")

# -------------------------------------------------------------
# Main learning routine
# -------------------------------------------------------------
def calculate_average_outcomes():
    """
    Loads the latest daily log (e.g., 2025-10-20.json) and calculates
    average rewards per training type for the CURRENT character.
    Saves learning results persistently to brain.json and summary.json.
    """
    from core import state  # Avoid circular import

    if not os.path.exists(LOG_DIR):
        warning("⚠️ No training log directory found.")
        return {}

    # --- Auto-detect latest log ---
    json_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]
    if not json_files:
        warning("⚠️ No training log found yet — nothing to learn.")
        return {}

    latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)))
    path = os.path.join(LOG_DIR, latest_file)
    info(f"📖 Loading latest log file: {latest_file}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except json.JSONDecodeError:
        warning(f"⚠️ Could not read log file: {latest_file}")
        return {}

    # --- Filter by current character ---
    current_char = getattr(state, "CURRENT_CHARACTER", "Unknown")
    records = [r for r in records if r.get("character") == current_char]
    if not records:
        warning(f"⚠️ No past data found for {current_char}. Starting fresh.")
        return {}

    # --- Compute averages ---
    stat_rewards = defaultdict(list)
    for r in records:
        decision = r.get("decision")
        reward = r.get("reward", 0)
        if decision:
            stat_rewards[decision].append(reward)

    if not stat_rewards:
        warning("⚠️ No valid decision data found.")
        return {}

    averaged = {k: round(mean(v), 3) for k, v in stat_rewards.items()}
    info(f"📈 Calculated average outcomes for {current_char}: {averaged}")

    # --- Update persistent brain ---
    brain = load_brain()
    for stat, value in averaged.items():
        old = brain.get(stat, 0)
        brain[stat] = round((old * 0.7 + value * 0.3), 3)
    save_brain(brain)
    save_summary(averaged, current_char)

    return averaged
