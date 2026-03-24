# -*- coding: utf-8 -*-
"""
Shared Configuration — NHL Overtime Model

Centralizes paths and season boundaries used across all pipeline files.

@author: chazf
"""

import os

PROJECT_DIR = r"D:\Python\NHL Overtime Model"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Paths
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_DIR, "model_artifacts")

GAME_IDS_FILE = os.path.join(RAW_DIR, "game_ids.csv")
ODDS_2026_FILE = os.path.join(PROJECT_DIR, "nhl_odds_2026.csv")
ODDS_HIST_FILE = os.path.join(PROJECT_DIR, "odds_collected.csv")
THREE_WAY_ODDS_FILE = os.path.join(RAW_DIR, "three_way_odds.csv")

# The Odds API (https://the-odds-api.com)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# Season boundaries (October 1st each year)
SEASON_BOUNDARIES = {
    20242025: "2024-10-01",
    20252026: "2025-10-01",
}
