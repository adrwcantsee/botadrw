import threading
import time
from utils.log import info, warning, debug
from core import recognizer

SCAN_INTERVAL = 1.0  # seconds between scans


def analyze_screen():
    """Analyzes current screen to detect training/race/menu states."""
    try:
        templates = {
            "train_btn": "assets/buttons/train.png",
            "race_btn": "assets/buttons/race.png",
            "rest_btn": "assets/buttons/rest.png",
            "next_btn": "assets/buttons/next.png",
        }

        results = recognizer.multi_match_templates(templates, threshold=0.85)
        found = [k for k, v in results.items() if v]

        if found:
            debug(f"🔎 Found UI elements: {found}")
        else:
            debug("No known buttons detected — possibly loading or transition screen.")

        # Optional: detect low-energy region (adjust coords if needed)
        energy_region = (1500, 920, 1650, 940)
        energy_gray_pixels = recognizer.count_pixels_of_color([117,117,117], energy_region)
        if energy_gray_pixels > 500:
            debug(f"⚡ Low energy detected ({energy_gray_pixels} gray pixels).")

    except Exception as e:
        warning(f"Scanner error: {e}")


def scanner_loop():
    """Main scanning loop — runs every few seconds."""
    info("🧠 Scanner thread started.")
    while True:
        analyze_screen()
        time.sleep(SCAN_INTERVAL)


def start_scanner():
    """Starts the scanner in a background thread."""
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    return t
