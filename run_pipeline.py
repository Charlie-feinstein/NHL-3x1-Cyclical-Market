# -*- coding: utf-8 -*-
"""
Unified Pipeline Runner — NHL Overtime Model

Runs all pipeline steps in the correct order with dependency validation.
Each step checks that its required input files exist before running.

Usage:
    Spyder:  Change START_STEP below, then run the file.
    CLI:     python run_pipeline.py --step 3   (start from step 3)

Steps:
    0. Scrape Raw Data               → data/raw/*.csv (incremental)
    1. xG Feature Engineering        → xg_training_data.csv
    2. xG Model Training             → shot_xg_{year}.csv
    3. Goalie Feature Engineering     → goalie_game_features_{year}.csv
    4. Goalie Model Training          → goalie_deployment_predictions_{year}.csv
    5. Regulation Feature Engineering → regulation_features_{year}.csv
    6. Regulation Model Training      → regulation_predictions_{year}.csv
    7. OT Feature Engineering         → ot_features_{year}.csv
    8. OT Model Training             → ot_predictions_{year}.csv

@author: chazf
"""

import sys
import os
import time
import argparse

# ── Configuration ────────────────────────────────────────────────────────────
# Change this to skip completed steps when running in Spyder
# 0 = start with data scraping, 1 = skip scraping (default)
START_STEP = 0

CUTOFF_YEARS = ["2024", "2025"]
TRAIN_CUTOFFS = ["2024-10-01", "2025-10-01"]

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import PROCESSED_DIR, RAW_DIR, MODEL_DIR

# ── Static imports ───────────────────────────────────────────────────────────
# Scrapers (Step 0)
from scrapers.game_scraper import scrape_game_ids
from scrapers.pbp_boxscore_scraper import scrape_all_games as scrape_pbp
from scrapers.goalie_stats_scraper import scrape_all as scrape_goalie_stats
from scrapers.team_stats_scraper import scrape_all as scrape_team_stats
from scrapers.standings_scraper import scrape_standings

# Feature engineering (Steps 1, 3, 5, 7)
from features.xg_features import build_xg_features
from features.goalie_features import build_goalie_features
from features.regulation_features import build_regulation_features
from features.ot_features import build_ot_features

# Model training (Steps 2, 4, 6, 8)
from models.xg_model import train_xg_model
from models.goalie_model import train_goalie_model
from models.regulation_model import train_regulation_model
from models.ot_model import train_ot_model


# ── Dependency definitions ───────────────────────────────────────────────────
# Each step lists the files that must exist before it can run.
# "{year}" is expanded for each cutoff year.

STEP_DEPS = {
    0: {
        "name": "Scrape Raw Data",
        "requires": [],
    },
    1: {
        "name": "xG Feature Engineering",
        "requires": [
            os.path.join(RAW_DIR, "game_ids.csv"),
        ],
    },
    2: {
        "name": "xG Model Training",
        "requires": [
            os.path.join(PROCESSED_DIR, "xg_training_data.csv"),
        ],
    },
    3: {
        "name": "Goalie Feature Engineering",
        "requires": [
            os.path.join(PROCESSED_DIR, "shot_xg_{year}.csv"),
        ],
    },
    4: {
        "name": "Goalie Model Training",
        "requires": [
            os.path.join(PROCESSED_DIR, "goalie_game_features_{year}.csv"),
        ],
    },
    5: {
        "name": "Regulation Feature Engineering",
        "requires": [
            os.path.join(PROCESSED_DIR, "goalie_deployment_predictions_{year}.csv"),
        ],
    },
    6: {
        "name": "Regulation Model Training",
        "requires": [
            os.path.join(PROCESSED_DIR, "regulation_features_{year}.csv"),
        ],
    },
    7: {
        "name": "OT Feature Engineering",
        "requires": [
            os.path.join(PROCESSED_DIR, "regulation_predictions_{year}.csv"),
        ],
    },
    8: {
        "name": "OT Model Training",
        "requires": [
            os.path.join(PROCESSED_DIR, "ot_features_{year}.csv"),
        ],
    },
}


def check_dependencies(step_num):
    """Verify required input files exist before running a step."""
    deps = STEP_DEPS[step_num]
    missing = []
    for template in deps["requires"]:
        if "{year}" in template:
            for year in CUTOFF_YEARS:
                path = template.format(year=year)
                if not os.path.exists(path):
                    missing.append(path)
        else:
            if not os.path.exists(template):
                missing.append(template)
    return missing


def run_step(step_num):
    """Run a single pipeline step."""
    name = STEP_DEPS[step_num]["name"]
    print(f"\n{'='*70}")
    print(f"  Step {step_num}/8: {name}")
    print(f"{'='*70}")

    # Check dependencies
    missing = check_dependencies(step_num)
    if missing:
        print(f"\n  MISSING dependencies for step {step_num}:")
        for f in missing:
            print(f"    - {f}")
        print(f"\n  Run earlier steps first (or set START_STEP=0).")
        sys.exit(1)

    t0 = time.time()

    if step_num == 0:
        # All scrapers are incremental (skip already-fetched data) except
        # game_scraper which needs force_refresh=True to pick up new games.
        print("\n  [0a] Game schedule (force_refresh — re-fetches all seasons)...")
        scrape_game_ids(force_refresh=True)
        print("\n  [0b] Play-by-play + boxscores (incremental)...")
        scrape_pbp()
        print("\n  [0c] Goalie stats (incremental)...")
        scrape_goalie_stats()
        print("\n  [0d] Team stats (incremental)...")
        scrape_team_stats()
        print("\n  [0e] Standings (incremental)...")
        scrape_standings()

    elif step_num == 1:
        build_xg_features()

    elif step_num == 2:
        for cutoff in TRAIN_CUTOFFS:
            train_xg_model(train_cutoff=cutoff)

    elif step_num == 3:
        for year in CUTOFF_YEARS:
            build_goalie_features(cutoff_year=year)

    elif step_num == 4:
        for cutoff in TRAIN_CUTOFFS:
            train_goalie_model(train_cutoff=cutoff)

    elif step_num == 5:
        for year in CUTOFF_YEARS:
            build_regulation_features(cutoff_year=year)

    elif step_num == 6:
        for cutoff in TRAIN_CUTOFFS:
            train_regulation_model(train_cutoff=cutoff)

    elif step_num == 7:
        for year in CUTOFF_YEARS:
            build_ot_features(cutoff_year=year)

    elif step_num == 8:
        for cutoff in TRAIN_CUTOFFS:
            train_ot_model(test_season_start=cutoff)

    elapsed = time.time() - t0
    print(f"\n  Step {step_num} completed in {elapsed:.1f}s")
    return elapsed


def main():
    # CLI arg overrides START_STEP
    parser = argparse.ArgumentParser(description="NHL OT Model Pipeline Runner")
    parser.add_argument("--step", type=int, default=None,
                        help="Start from this step (0=scrape, 1-8=pipeline)")
    args, _ = parser.parse_known_args()

    start = args.step if args.step is not None else START_STEP

    if start < 0 or start > 8:
        print(f"Invalid step: {start}. Must be 0-8.")
        sys.exit(1)

    print(f"\nNHL Overtime Model — Pipeline Runner")
    print(f"Starting from step {start}/8")
    print(f"Cutoff years: {CUTOFF_YEARS}")
    print(f"Train cutoffs: {TRAIN_CUTOFFS}")

    timings = {}
    total_t0 = time.time()

    for step_num in range(start, 9):
        elapsed = run_step(step_num)
        timings[step_num] = elapsed

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  Pipeline Complete!")
    print(f"{'='*70}")
    for step_num, elapsed in timings.items():
        name = STEP_DEPS[step_num]["name"]
        print(f"  Step {step_num}: {name:35s} {elapsed:7.1f}s")
    print(f"  {'─'*50}")
    print(f"  Total: {total_elapsed:44.1f}s")
    print()


if __name__ == "__main__":
    main()
