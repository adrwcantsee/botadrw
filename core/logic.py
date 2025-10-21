import core.state as state
from core.state import check_current_year, stat_state, check_energy_level, check_aptitudes
from utils.log import info, warning, debug
import utils.constants as constants
from core.recorder import save_turn_data
from core.learner import calculate_average_outcomes
from core.decision_memory import remember_decision, get_memory_bias
import datetime, os


# -------------------------------------------------------------
# Stat priority config
# -------------------------------------------------------------
def get_stat_priority(stat_key: str) -> int:
    return state.PRIORITY_STAT.index(stat_key) if stat_key in state.PRIORITY_STAT else 999


# -------------------------------------------------------------
# Recommended stat weights by race distance
# -------------------------------------------------------------
RECOMMENDED_STATS_BY_DISTANCE = {
    "sprint": {"speed": 0.45, "power": 0.30, "stamina": 0.10, "wit": 0.10, "guts": 0.05},
    "mile":   {"speed": 0.40, "power": 0.25, "stamina": 0.20, "wit": 0.10, "guts": 0.05},
    "medium": {"speed": 0.35, "stamina": 0.30, "power": 0.20, "wit": 0.10, "guts": 0.05},
    "long":   {"speed": 0.30, "stamina": 0.40, "power": 0.15, "wit": 0.10, "guts": 0.05},
}


# -------------------------------------------------------------
# Priority weight multiplier
# -------------------------------------------------------------
PRIORITY_WEIGHTS_LIST = {
    "HEAVY": 0.75,
    "MEDIUM": 0.5,
    "LIGHT": 0.25,
    "NONE": 0,
}


# -------------------------------------------------------------
# Choose the training with most supports (fallback logic)
# -------------------------------------------------------------
def most_support_card(results):
    wit_data = results.get("wit")
    non_wit_results = {k: v for k, v in results.items() if k != "wit" and int(v["failure"]) <= state.MAX_FAILURE}
    energy_level, _ = check_energy_level()

    if energy_level < state.SKIP_TRAINING_ENERGY:
        info("⚡ Energy too low for safe training. Resting instead.")
        return None

    if len(non_wit_results) == 0 and wit_data and int(wit_data["failure"]) <= state.MAX_FAILURE and wit_data["total_supports"] >= 2:
        info("All other trainings unsafe — WIT has enough support cards.")
        return "wit"

    filtered_results = {k: v for k, v in results.items() if int(v["failure"]) <= state.MAX_FAILURE}
    if not filtered_results:
        info("No safe training found — resting.")
        return None

    best_key, _ = max(filtered_results.items(), key=training_score)
    return best_key


# -------------------------------------------------------------
# Training score: guide logic + memory + learned data + distance bias
# -------------------------------------------------------------
def training_score(x):
    global learned
    stat_name, data = x
    priority_weight = PRIORITY_WEIGHTS_LIST.get(state.PRIORITY_WEIGHT, 0.5)

    # --- Base calculation ---
    base = data["total_supports"] + 0.5 * data.get("total_hints", 0)

    # Friendship impact
    friendship_value = (
        data["total_friendship_levels"]["green"]
        + data["total_friendship_levels"]["blue"] * 1.1
        + data["total_friendship_levels"]["gray"] * 1.2
    )
    base += friendship_value * 0.1

    # Rainbow bonuses
    total_rainbows = (
        data.get("friendship_levels", {}).get("max", 0)
        + data.get("friendship_levels", {}).get("yellow", 0)
    )
    base += total_rainbows * 2

    # Priority and learning effects
    multiplier = 1 + state.PRIORITY_EFFECTS_LIST[get_stat_priority(stat_name)] * priority_weight
    learn_bonus = learned.get(stat_name, 0) / 100.0 if "learned" in globals() and learned else 0
    total = base * multiplier * (1 + learn_bonus)

    # --- NEW: Race distance–based stat bias ---
    aptitude = getattr(state, "APTITUDES", {})
    distance_type = "medium"  # fallback
    for key in aptitude.keys():
        if "distance_" in key and aptitude[key].lower() in ["a", "b"]:
            distance_type = key.replace("distance_", "")
            break

    distance_weights = RECOMMENDED_STATS_BY_DISTANCE.get(distance_type, {})
    distance_bias = distance_weights.get(stat_name, 0)
    total *= (1 + distance_bias * 0.25)  # 25% bias weight

    # --- Memory-based bias ---
    phase = getattr(state, "current_phase", "unknown")
    energy_level, _ = check_energy_level()
    memory_bias = get_memory_bias(phase, energy_level, stat_name)
    total *= (1 + memory_bias)

    debug(f"{stat_name} -> total={total:.2f}, bias={memory_bias:.2f}, learn_bonus={learn_bonus:.2f}, dist={distance_type}")
    return (total, -get_stat_priority(stat_name))


# -------------------------------------------------------------
# Filter capped stats
# -------------------------------------------------------------
def filter_by_stat_caps(results, current_stats):
    return {s: d for s, d in results.items() if current_stats.get(s, 0) < state.STAT_CAPS.get(s, 1200)}


# -------------------------------------------------------------
# SMART ENERGY + SUMMER LOGIC + MAIN DECISION
# -------------------------------------------------------------
def do_something(results):
    global learned
    year = check_current_year()
    current_stats = stat_state()
    energy_level, _ = check_energy_level()

    # Detect phase early
    if "Junior Year" in year:
        phase = "early"
    elif "Classic Year" in year:
        phase = "mid"
    elif "Senior Year" in year:
        phase = "late"
    else:
        phase = "unknown"
    state.current_phase = phase

    # --- SMART ENERGY CONSERVATION ---
    month = getattr(state, "CURRENT_MONTH", "") or ""
    month = month.lower()

    # Summer prep for Classic/Senior June
    if ("june" in month or "late june" in month) and phase in ["mid", "late"] and energy_level < 80:
        info(f"☀️ Preparing for summer training — resting to reach full energy ({energy_level:.1f}%).")
        return auto_rest(year, phase, energy_level, current_stats)

    # Save energy tiers
    if energy_level < 30:
        info(f"💤 Energy critically low ({energy_level:.1f}%) — resting.")
        return auto_rest(year, phase, energy_level, current_stats)
    elif energy_level < 50:
        if not has_high_support(results):
            info(f"⚡ Energy moderate ({energy_level:.1f}%) — not enough supports, resting.")
            return auto_rest(year, phase, energy_level, current_stats)
    elif energy_level < 70:
        if not has_good_support(results):
            info(f"⚡ Energy medium ({energy_level:.1f}%) — saving energy for stronger turns.")
            return auto_rest(year, phase, energy_level, current_stats)

    # Load learned data
    learned = calculate_average_outcomes()
    info(f"📚 Loaded learned weights: {learned if learned else 'none'}")

    # Filter by caps
    filtered = filter_by_stat_caps(results, current_stats)
    if not filtered:
        info("All stats capped or no valid training.")
        return None

    # Pick best training
    if phase == "early":
        result, _ = max([(k, v["total_supports"]) for k, v in filtered.items()], key=lambda x: x[1], default=(None, 0))
    else:
        result = rainbow_training(filtered) or most_support_card(filtered)

    if not result:
        return auto_rest(year, phase, energy_level, current_stats)

    # Record + reinforce
    try:
        chosen_data = filtered.get(result, {})
        reward = compute_reward(current_stats, energy_level, chosen_data)
        record_training(year, phase, energy_level, current_stats, result, reward)
        remember_decision(phase, energy_level, result, reward)
    except Exception as e:
        warning(f"❌ Could not save training data: {e}")

    return result


# -------------------------------------------------------------
# Reward calculation & record helpers
# -------------------------------------------------------------
def compute_reward(current_stats, energy, chosen_data):
    avg_stat = sum(current_stats.values()) / len(current_stats)
    reward = avg_stat / 1000
    reward += 0.3 if energy > 60 else -0.2 if energy < 30 else 0
    reward += 0.4 * chosen_data.get("total_supports", 0)
    reward -= 0.3 * (chosen_data.get("failure", 0) / 100)
    return round(reward, 3)


def record_training(year, phase, energy, current_stats, result, reward):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "character": getattr(state, "CURRENT_CHARACTER", "Unknown"),
        "year": year,
        "phase": phase,
        "energy": energy,
        "current_stats": current_stats,
        "decision": result,
        "reward": reward,
        "metadata": {
            "script_version": "v1.7",
            "ai_mode": "smart-energy+distance-bias",
            "trainer_behavior": "auto",
            "priority_weight": state.PRIORITY_WEIGHT,
            "session_id": os.getenv("SESSION_ID", "default"),
        },
    }
    save_turn_data(record)
    info(f"📘 Training record saved for {result.upper()} (reward={reward}).")


def auto_rest(year, phase, energy, current_stats):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "character": getattr(state, "CURRENT_CHARACTER", "Unknown"),
        "year": year,
        "phase": phase,
        "energy": energy,
        "current_stats": current_stats,
        "decision": "rest",
        "reward": 0,
        "metadata": {"script_version": "v1.7", "ai_mode": "auto-rest"},
    }
    save_turn_data(record)
    info("💤 Auto-rest recorded.")
    try:
        remember_decision(phase, energy, "rest", 0)
    except Exception as e:
        warning(f"⚠️ Could not update decision memory for rest: {e}")
    return None


# -------------------------------------------------------------
# Support check helpers
# -------------------------------------------------------------
def has_high_support(results):
    for v in results.values():
        if v.get("total_supports", 0) >= 3:
            return True
        if v.get("friendship_levels", {}).get("max", 0) > 0:
            return True
    return False


def has_good_support(results):
    return any(v.get("total_supports", 0) >= 2 for v in results.values())


# -------------------------------------------------------------
# Rainbow training logic
# -------------------------------------------------------------
def rainbow_training(results):
    candidates = {}
    for stat_name, data in results.items():
        fail = int(data["failure"])
        if fail > state.MAX_FAILURE:
            continue
        total_rainbow = data.get("friendship_levels", {}).get("max", 0) + data.get("friendship_levels", {}).get("yellow", 0)
        score = total_rainbow * 3.0 + data["total_supports"] * 1.2 + data["total_hints"] * 0.5
        candidates[stat_name] = score
    if not candidates:
        return None
    best_key = max(candidates, key=candidates.get)
    info(f"🌈 Rainbow training: {best_key.upper()} ({candidates[best_key]:.2f})")
    return best_key


# -------------------------------------------------------------
# Race logic & aptitude matching
# -------------------------------------------------------------
def decide_race_for_goal(year, turn, criteria, keywords):
    no_race = (False, None)
    if year == "Junior Year Pre-Debut" or turn >= 10:
        return no_race

    criteria_text = criteria or ""
    if any(word in criteria_text for word in keywords):
        info("🎯 Criteria word found — evaluating race options.")
        if "Progress" in criteria_text:
            if "G1" in criteria_text or "GI" in criteria_text:
                race_list = constants.RACE_LOOKUP.get(year, [])
                if not race_list:
                    return False, None
                best_race = filter_races_by_aptitude(race_list, state.APTITUDES)
                return True, best_race["name"] if best_race else None
            return False, "any"
        return False, "any"
    return no_race


def filter_races_by_aptitude(race_list, aptitudes):
    GRADE_SCORE = {"a": 2, "b": 1}
    results = []
    for race in race_list:
        surface_key = f"surface_{race['terrain'].lower()}"
        distance_key = f"distance_{race['distance']['type'].lower()}"
        s = GRADE_SCORE.get(aptitudes.get(surface_key, ""), 0)
        d = GRADE_SCORE.get(aptitudes.get(distance_key, ""), 0)
        if s and d:
            score = s + d
            results.append((score, race["fans"]["gained"], race))
    if not results:
        return None
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return results[0][2]


def should_rest(energy_level):
    if energy_level < state.SKIP_TRAINING_ENERGY:
        info(f"💤 Energy {energy_level} too low — resting.")
        return True
    return False
