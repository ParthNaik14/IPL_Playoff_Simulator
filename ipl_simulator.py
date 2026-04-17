import os
import re
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import copy

team_colors = {
    "Gujarat Titans": {"bg": "#0A0A0A", "text": "#F5C518"},
    "Royal Challengers Bengaluru": {"bg": "#A71930", "text": "#0A0A0A"},
    "Delhi Capitals": {"bg": "#17449B", "text": "#FF0000"},
    "Punjab Kings": {"bg": "#D21F3C", "text": "#FFD700"},
    "Kolkata Knight Riders": {"bg": "#2E0854", "text": "#FFD700"},
    "Lucknow Super Giants": {"bg": "#13294B", "text": "#FF5722"},
    "Rajasthan Royals": {"bg": "#EA1574", "text": "#FFFFFF"},
    "Mumbai Indians": {"bg": "#045093", "text": "#FFD700"},
    "Chennai Super Kings": {"bg": "#F7E600", "text": "#17449B"},
    "Sunrisers Hyderabad": {"bg": "#F44336", "text": "#000000"}
}

teams = [
    "Delhi Capitals", "Gujarat Titans", "Royal Challengers Bengaluru", "Punjab Kings",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Rajasthan Royals",
    "Mumbai Indians", "Chennai Super Kings", "Sunrisers Hyderabad"
]

# ---------------------------------------------------------------------------
# IPL 2026 — points table (auto-updated by commit_result)
# ---------------------------------------------------------------------------
updated_points_data = {
    "Royal Challengers Bengaluru": {"points": 8, "matches": 5, "runs_for": 1043, "overs_faced": "90.5", "runs_against": 978, "overs_bowled": "98.0"},
    "Punjab Kings": {"points": 9, "matches": 5, "runs_for": 796, "overs_faced": "73.1", "runs_against": 785, "overs_bowled": "80.0"},
    "Mumbai Indians": {"points": 2, "matches": 5, "runs_for": 926, "overs_faced": "90.1", "runs_against": 972, "overs_bowled": "85.4"},
    "Gujarat Titans": {"points": 6, "matches": 5, "runs_for": 922, "overs_faced": "98.2", "runs_against": 928, "overs_bowled": "99.1"},
    "Delhi Capitals": {"points": 4, "matches": 4, "runs_for": 707, "overs_faced": "75.2", "runs_against": 725, "overs_bowled": "80.0"},
    "Kolkata Knight Riders": {"points": 1, "matches": 6, "runs_for": 902, "overs_faced": "100.0", "runs_against": 1005, "overs_bowled": "98.5"},
    "Lucknow Super Giants": {"points": 4, "matches": 5, "runs_for": 793, "overs_faced": "99.5", "runs_against": 796, "overs_bowled": "91.0"},
    "Sunrisers Hyderabad": {"points": 4, "matches": 5, "runs_for": 1018, "overs_faced": "100.0", "runs_against": 906, "overs_bowled": "94.2"},
    "Rajasthan Royals": {"points": 8, "matches": 5, "runs_for": 849, "overs_faced": "81.1", "runs_against": 871, "overs_bowled": "91.0"},
    "Chennai Super Kings": {"points": 4, "matches": 5, "runs_for": 947, "overs_faced": "100.0", "runs_against": 937, "overs_bowled": "90.5"},
}

# ---------------------------------------------------------------------------
# Elo ratings — auto-updated by commit_result (margin-aware)
# ---------------------------------------------------------------------------
elo_ratings = {
    "Mumbai Indians":                        1469.26,
    "Gujarat Titans":                        1512.23,
    "Royal Challengers Bengaluru":           1549.86,
    "Punjab Kings":                          1559.94,
    "Lucknow Super Giants":                  1467.64,
    "Kolkata Knight Riders":                 1446.22,
    "Rajasthan Royals":                      1532.34,
    "Sunrisers Hyderabad":                   1489.92,
    "Delhi Capitals":                        1499.19,
    "Chennai Super Kings":                   1479.0,
}

ELO_K = 32

recent_form = {
    "Royal Challengers Bengaluru":           [1, 1, 0, 1, 1],
    "Gujarat Titans":                        [0, 0, 1, 1, 1],
    "Mumbai Indians":                        [1, 0, 0, 0, 0],
    "Punjab Kings":                          [1, 1, 9, 1, 1],
    "Kolkata Knight Riders":                 [0, 9, 0, 0, 0],
    "Delhi Capitals":                        [1, 1, 0, 0],
    "Lucknow Super Giants":                  [0, 1, 1, 0, 0],
    "Sunrisers Hyderabad":                   [0, 1, 0, 0, 1],
    "Rajasthan Royals":                      [1, 1, 1, 1, 0],
    "Chennai Super Kings":                   [0, 0, 0, 1, 1],
}

FORM_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30]


def get_form_score(team):
    results = recent_form[team]
    if not results:
        return 0.5

    # 1. FILTER: Create a list that physically removes any 9s
    math_results = [r for r in results if r != 9]

    # 2. SAFETY: If a team has ONLY played NRs, give them neutral 0.5
    if not math_results:
        return 0.5

    # 3. WEIGHTS: Use the length of the FILTERED list
    w = FORM_WEIGHTS[-len(math_results):]
    total_w = sum(w)

    # 4. CALC: Use math_results, NOT the original results list
    return sum(wi * r for wi, r in zip(w, math_results)) / total_w


# ---------------------------------------------------------------------------
# Venue-specific home advantage multipliers
# ---------------------------------------------------------------------------
home_advantage = {
    "CHENNAI": 1.10,
    "JAIPUR": 1.09,
    "AHMEDABAD": 1.08,
    "MUMBAI": 1.07,
    "KOLKATA": 1.06,
    "HYDERABAD": 1.05,
    "BENGALURU": 1.04,
    "DELHI": 1.03,
    "LUCKNOW": 1.02,
    "NEW CHANDIGARH": 1.01,
    "DHARAMSHALA": 1.00,
    "RAIPUR": 1.00,
    "GUWAHATI": 1.00,
}


def get_home_boost(venue):
    return home_advantage.get(venue.upper().strip(), 1.03)


# ---------------------------------------------------------------------------
# IPL 2026 Schedule — Full 70 matches
# ---------------------------------------------------------------------------
remaining_matches = [
    {"home": "Royal Challengers Bengaluru", "away": "Delhi Capitals",              "venue": "Bengaluru",       "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad",         "away": "Chennai Super Kings",         "venue": "Hyderabad",       "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders",       "away": "Rajasthan Royals",            "venue": "Kolkata",         "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings",                "away": "Lucknow Super Giants",        "venue": "New Chandigarh",  "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans",              "away": "Mumbai Indians",              "venue": "Ahmedabad",       "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad",         "away": "Delhi Capitals",              "venue": "Hyderabad",       "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants",        "away": "Rajasthan Royals",            "venue": "Lucknow",         "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians",              "away": "Chennai Super Kings",         "venue": "Mumbai",          "result": None, "margin": None, "applied": False},
    {"home": "Royal Challengers Bengaluru", "away": "Gujarat Titans",              "venue": "Bengaluru",       "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals",              "away": "Punjab Kings",                "venue": "Delhi",           "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals",            "away": "Sunrisers Hyderabad",         "venue": "Jaipur",          "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings",         "away": "Gujarat Titans",              "venue": "Chennai",         "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants",        "away": "Kolkata Knight Riders",       "venue": "Lucknow",         "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals",              "away": "Royal Challengers Bengaluru", "venue": "Delhi",           "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings",                "away": "Rajasthan Royals",            "venue": "New Chandigarh",  "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians",              "away": "Sunrisers Hyderabad",         "venue": "Mumbai",          "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans",              "away": "Royal Challengers Bengaluru", "venue": "Ahmedabad",       "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals",            "away": "Delhi Capitals",              "venue": "Jaipur",          "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings",         "away": "Mumbai Indians",              "venue": "Chennai",         "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad",         "away": "Kolkata Knight Riders",       "venue": "Hyderabad",       "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans",              "away": "Punjab Kings",                "venue": "Ahmedabad",       "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians",              "away": "Lucknow Super Giants",        "venue": "Mumbai",          "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals",              "away": "Chennai Super Kings",         "venue": "Delhi",           "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad",         "away": "Punjab Kings",                "venue": "Hyderabad",       "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants",        "away": "Royal Challengers Bengaluru", "venue": "Lucknow",         "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals",              "away": "Kolkata Knight Riders",       "venue": "Delhi",           "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals",            "away": "Gujarat Titans",              "venue": "Jaipur",          "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings",         "away": "Lucknow Super Giants",        "venue": "Chennai",         "result": None, "margin": None, "applied": False},
    {"home": "Royal Challengers Bengaluru", "away": "Mumbai Indians",              "venue": "Raipur",          "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings",                "away": "Delhi Capitals",              "venue": "Dharamshala",     "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans",              "away": "Sunrisers Hyderabad",         "venue": "Ahmedabad",       "result": None, "margin": None, "applied": False},
    {"home": "Royal Challengers Bengaluru", "away": "Kolkata Knight Riders",       "venue": "Raipur",          "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings",                "away": "Mumbai Indians",              "venue": "Dharamshala",     "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants",        "away": "Chennai Super Kings",         "venue": "Lucknow",         "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders",       "away": "Gujarat Titans",              "venue": "Kolkata",         "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings",                "away": "Royal Challengers Bengaluru", "venue": "Dharamshala",     "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals",              "away": "Rajasthan Royals",            "venue": "Delhi",           "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings",         "away": "Sunrisers Hyderabad",         "venue": "Chennai",         "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals",            "away": "Lucknow Super Giants",        "venue": "Jaipur",          "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders",       "away": "Mumbai Indians",              "venue": "Kolkata",         "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans",              "away": "Chennai Super Kings",         "venue": "Ahmedabad",       "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad",         "away": "Royal Challengers Bengaluru", "venue": "Hyderabad",       "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants",        "away": "Punjab Kings",                "venue": "Lucknow",         "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians",              "away": "Rajasthan Royals",            "venue": "Mumbai",          "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders",       "away": "Delhi Capitals",              "venue": "Kolkata",         "result": None, "margin": None, "applied": False},
]

TOTAL_MATCHES = 70
MATCHES_COMMITTED = 25

# ---------------------------------------------------------------------------
# Committed results log
# ---------------------------------------------------------------------------
committed_results = [{"home": "Royal Challengers Bengaluru", "away": "Sunrisers Hyderabad", "venue": "Bengaluru", "winner": "Royal Challengers Bengaluru", "abandoned": False, "winner_runs": 203, "winner_overs_str": "15.4", "loser_runs": 201, "loser_overs_str": "20.0", "elo_before": {"Royal Challengers Bengaluru": 1509.8, "Sunrisers Hyderabad": 1492.1}, "form_before": {"Royal Challengers Bengaluru": [], "Sunrisers Hyderabad": []}}, {"home": "Mumbai Indians", "away": "Kolkata Knight Riders", "venue": "Mumbai", "winner": "Mumbai Indians", "abandoned": False, "winner_runs": 224, "winner_overs_str": "19.1", "loser_runs": 220, "loser_overs_str": "20.0", "elo_before": {"Mumbai Indians": 1515.0, "Kolkata Knight Riders": 1498.7}, "form_before": {"Mumbai Indians": [], "Kolkata Knight Riders": []}}, {"home": "Rajasthan Royals", "away": "Chennai Super Kings", "venue": "Guwahati", "winner": "Rajasthan Royals", "abandoned": False, "winner_runs": 128, "winner_overs_str": "12.1", "loser_runs": 127, "loser_overs_str": "20.0", "elo_before": {"Rajasthan Royals": 1495.3, "Chennai Super Kings": 1485.0}, "form_before": {"Rajasthan Royals": [], "Chennai Super Kings": []}}, {"home": "Punjab Kings", "away": "Gujarat Titans", "venue": "New Chandigarh", "winner": "Punjab Kings", "abandoned": False, "winner_runs": 165, "winner_overs_str": "19.1", "loser_runs": 162, "loser_overs_str": "20", "elo_before": {"Punjab Kings": 1506.5, "Gujarat Titans": 1512.4}, "form_before": {"Punjab Kings": [], "Gujarat Titans": []}}, {"home": "Lucknow Super Giants", "away": "Delhi Capitals", "venue": "Lucknow", "winner": "Delhi Capitals", "abandoned": False, "winner_runs": 145, "winner_overs_str": "17.1", "loser_runs": 141, "loser_overs_str": "20", "elo_before": {"Lucknow Super Giants": 1502.2, "Delhi Capitals": 1488.6}, "form_before": {"Lucknow Super Giants": [], "Delhi Capitals": []}}, {"home": "Kolkata Knight Riders", "away": "Sunrisers Hyderabad", "venue": "Kolkata", "winner": "Sunrisers Hyderabad", "abandoned": False, "winner_runs": 226, "winner_overs_str": "20", "loser_runs": 161, "loser_overs_str": "20", "elo_before": {"Kolkata Knight Riders": 1484.45, "Sunrisers Hyderabad": 1476.91}, "form_before": {"Kolkata Knight Riders": [0], "Sunrisers Hyderabad": [0]}}, {"home": "Chennai Super Kings", "away": "Punjab Kings", "venue": "Chennai", "winner": "Punjab Kings", "abandoned": False, "winner_runs": 210, "winner_overs_str": "18.4", "loser_runs": 209, "loser_overs_str": "20", "elo_before": {"Chennai Super Kings": 1469.47, "Punjab Kings": 1520.44}, "form_before": {"Chennai Super Kings": [0], "Punjab Kings": [1]}}, {"home": "Delhi Capitals", "away": "Mumbai Indians", "venue": "Delhi", "winner": "Delhi Capitals", "abandoned": False, "winner_runs": 164, "winner_overs_str": "18.1", "loser_runs": 162, "loser_overs_str": "20", "elo_before": {"Delhi Capitals": 1505.2, "Mumbai Indians": 1529.25}, "form_before": {"Delhi Capitals": [1], "Mumbai Indians": [1]}}, {"home": "Gujarat Titans", "away": "Rajasthan Royals", "venue": "Ahmedabad", "winner": "Rajasthan Royals", "abandoned": False, "winner_runs": 210, "winner_overs_str": "20", "loser_runs": 204, "loser_overs_str": "20", "elo_before": {"Gujarat Titans": 1498.46, "Rajasthan Royals": 1510.83}, "form_before": {"Gujarat Titans": [0], "Rajasthan Royals": [1]}}, {"home": "Sunrisers Hyderabad", "away": "Lucknow Super Giants", "venue": "Hyderabad", "winner": "Lucknow Super Giants", "abandoned": False, "winner_runs": 160, "winner_overs_str": "19.5", "loser_runs": 156, "loser_overs_str": "20", "elo_before": {"Sunrisers Hyderabad": 1493.26, "Lucknow Super Giants": 1485.6}, "form_before": {"Sunrisers Hyderabad": [0, 1], "Lucknow Super Giants": [0]}}, {"home": "Royal Challengers Bengaluru", "away": "Chennai Super Kings", "venue": "Bengaluru", "winner": "Royal Challengers Bengaluru", "abandoned": False, "winner_runs": 250, "winner_overs_str": "20", "loser_runs": 207, "loser_overs_str": "20", "elo_before": {"Royal Challengers Bengaluru": 1524.99, "Chennai Super Kings": 1456.38}, "form_before": {"Royal Challengers Bengaluru": [1], "Chennai Super Kings": [0, 0]}}, {"home": "Kolkata Knight Riders", "away": "Punjab Kings", "venue": "Kolkata", "winner": None, "abandoned": True, "winner_runs": 0, "winner_overs_str": "0.0", "loser_runs": 0, "loser_overs_str": "0.0", "elo_before": {"Kolkata Knight Riders": 1468.1, "Punjab Kings": 1533.53}, "form_before": {"Kolkata Knight Riders": [0, 0], "Punjab Kings": [1, 1]}}, {"home": "Rajasthan Royals", "away": "Mumbai Indians", "venue": "Guwahati", "winner": "Rajasthan Royals", "abandoned": False, "winner_runs": 150, "winner_overs_str": "11", "loser_runs": 123, "loser_overs_str": "11", "elo_before": {"Rajasthan Royals": 1520.42, "Mumbai Indians": 1512.45}, "form_before": {"Rajasthan Royals": [1, 1], "Mumbai Indians": [1, 0]}}, {"home": "Delhi Capitals", "away": "Gujarat Titans", "venue": "Delhi", "winner": "Gujarat Titans", "abandoned": False, "winner_runs": 210, "winner_overs_str": "20", "loser_runs": 209, "loser_overs_str": "20", "elo_before": {"Delhi Capitals": 1522.0, "Gujarat Titans": 1488.87}, "form_before": {"Delhi Capitals": [1, 1], "Gujarat Titans": [0, 0]}}, {"home": "Kolkata Knight Riders", "away": "Lucknow Super Giants", "venue": "Kolkata", "winner": "Lucknow Super Giants", "abandoned": False, "winner_runs": 182, "winner_overs_str": "20", "loser_runs": 181, "loser_overs_str": "20", "elo_before": {"Kolkata Knight Riders": 1468.1, "Lucknow Super Giants": 1495.29}, "form_before": {"Kolkata Knight Riders": [0, 0], "Lucknow Super Giants": [0, 1]}}, {"home": "Rajasthan Royals", "away": "Royal Challengers Bengaluru", "venue": "Guwahati", "winner": "Rajasthan Royals", "abandoned": False, "winner_runs": 202, "winner_overs_str": "18", "loser_runs": 201, "loser_overs_str": "20", "elo_before": {"Rajasthan Royals": 1536.05, "Royal Challengers Bengaluru": 1537.87}, "form_before": {"Rajasthan Royals": [1, 1, 1], "Royal Challengers Bengaluru": [1, 1]}}, {"home": "Punjab Kings", "away": "Sunrisers Hyderabad", "venue": "New Chandigarh", "winner": "Punjab Kings", "abandoned": False, "winner_runs": 223, "winner_overs_str": "18.5", "loser_runs": 219, "loser_overs_str": "20", "elo_before": {"Punjab Kings": 1533.53, "Sunrisers Hyderabad": 1483.57}, "form_before": {"Punjab Kings": [1, 1], "Sunrisers Hyderabad": [0, 1, 0]}}, {"home": "Chennai Super Kings", "away": "Delhi Capitals", "venue": "Chennai", "winner": "Chennai Super Kings", "abandoned": False, "winner_runs": 212, "winner_overs_str": "20", "loser_runs": 189, "loser_overs_str": "20", "elo_before": {"Chennai Super Kings": 1443.5, "Delhi Capitals": 1518.49}, "form_before": {"Chennai Super Kings": [0, 0, 0], "Delhi Capitals": [1, 1, 0]}}, {"home": "Lucknow Super Giants", "away": "Gujarat Titans", "venue": "Lucknow", "winner": "Gujarat Titans", "abandoned": False, "winner_runs": 165, "winner_overs_str": "18.4", "loser_runs": 164, "loser_overs_str": "20", "elo_before": {"Lucknow Super Giants": 1496.03, "Gujarat Titans": 1492.38}, "form_before": {"Lucknow Super Giants": [0, 1, 1], "Gujarat Titans": [0, 0, 1]}}, {"home": "Mumbai Indians", "away": "Royal Challengers Bengaluru", "venue": "Mumbai", "winner": "Royal Challengers Bengaluru", "abandoned": False, "winner_runs": 240, "winner_overs_str": "20", "loser_runs": 222, "loser_overs_str": "20", "elo_before": {"Mumbai Indians": 1496.82, "Royal Challengers Bengaluru": 1521.88}, "form_before": {"Mumbai Indians": [1, 0, 0], "Royal Challengers Bengaluru": [1, 1, 0]}}, {"home": "Sunrisers Hyderabad", "away": "Rajasthan Royals", "venue": "Hyderabad", "winner": "Sunrisers Hyderabad", "abandoned": False, "winner_runs": 216, "winner_overs_str": "20", "loser_runs": 159, "loser_overs_str": "20", "elo_before": {"Sunrisers Hyderabad": 1470.22, "Rajasthan Royals": 1552.04}, "form_before": {"Sunrisers Hyderabad": [0, 1, 0, 0], "Rajasthan Royals": [1, 1, 1, 1]}}, {"home": "Chennai Super Kings", "away": "Kolkata Knight Riders", "venue": "Chennai", "winner": "Chennai Super Kings", "abandoned": False, "winner_runs": 192, "winner_overs_str": "20", "loser_runs": 160, "loser_overs_str": "20", "elo_before": {"Chennai Super Kings": 1462.8, "Kolkata Knight Riders": 1467.36}, "form_before": {"Chennai Super Kings": [0, 0, 0, 1], "Kolkata Knight Riders": [0, 0, 9, 0]}}, {"home": "Royal Challengers Bengaluru", "away": "Lucknow Super Giants", "venue": "Bengaluru", "winner": "Royal Challengers Bengaluru", "abandoned": False, "winner_runs": 149, "winner_overs_str": "15.1", "loser_runs": 146, "loser_overs_str": "20", "elo_before": {"Royal Challengers Bengaluru": 1536.38, "Lucknow Super Giants": 1481.12}, "form_before": {"Royal Challengers Bengaluru": [1, 1, 0, 1], "Lucknow Super Giants": [0, 1, 1, 0]}}, {"home": "Mumbai Indians", "away": "Punjab Kings", "venue": "Mumbai", "winner": "Punjab Kings", "abandoned": False, "winner_runs": 198, "winner_overs_str": "16.3", "loser_runs": 195, "loser_overs_str": "20", "elo_before": {"Mumbai Indians": 1482.32, "Punjab Kings": 1546.88}, "form_before": {"Mumbai Indians": [1, 0, 0, 0], "Punjab Kings": [1, 1, 9, 1]}}, {"home": "Gujarat Titans", "away": "Kolkata Knight Riders", "venue": "Ahmedabad", "winner": "Gujarat Titans", "abandoned": False, "winner_runs": 181, "winner_overs_str": "19.4", "loser_runs": 180, "loser_overs_str": "20", "elo_before": {"Gujarat Titans": 1507.29, "Kolkata Knight Riders": 1451.16}, "form_before": {"Gujarat Titans": [0, 0, 1, 1], "Kolkata Knight Riders": [0, 0, 9, 0, 0]}}]  # END_COMMITTED_RESULTS


# ---------------------------------------------------------------------------
# NRR / overs helpers
# ---------------------------------------------------------------------------
def overs_to_float(overs_str):
    if isinstance(overs_str, (int, float)):
        return float(overs_str)
    try:
        parts = str(overs_str).split(".")
        whole = int(parts[0])
        balls = int(parts[1]) if len(parts) > 1 else 0
        return round(whole + balls / 6, 6)
    except:
        return 0.0


def _add_overs(existing_str, new_str):
    total_balls = round((overs_to_float(existing_str) + overs_to_float(new_str)) * 6)
    return f"{total_balls // 6}.{total_balls % 6}"


def calculate_nrr(team_data):
    rf = team_data["runs_for"]
    of = overs_to_float(team_data["overs_faced"])
    ra = team_data["runs_against"]
    ob = overs_to_float(team_data["overs_bowled"])
    if of == 0 or ob == 0:
        return 0.0
    return round((rf / of) - (ra / ob), 3)


def simulate_nrr_change(winner_weight, loser_weight):
    strength_diff = winner_weight - loser_weight
    base = np.clip(0.15 + 0.4 * strength_diff, 0.05, 0.50)
    noise = np.random.normal(0, 0.12)
    return round(np.clip(base + noise, -0.05, 0.80), 3)


# ---------------------------------------------------------------------------
# Elo update — margin-aware
# ---------------------------------------------------------------------------
def _update_elo_margin(winner, loser, winner_nrr_delta):
    wr = elo_ratings[winner]
    lr = elo_ratings[loser]
    expected_w = 1 / (1 + 10 ** ((lr - wr) / 400))
    margin_score = 0.5 + 0.5 * np.tanh(winner_nrr_delta / 0.4)
    elo_ratings[winner] = round(wr + ELO_K * (margin_score - expected_w), 2)
    elo_ratings[loser] = round(lr + ELO_K * ((1 - margin_score) - (1 - expected_w)), 2)


def get_elo_table():
    rows = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(rows, columns=["Team", "Elo Rating"])


def reset_elo_for_new_season(reversion=0.30):
    for team in elo_ratings:
        elo_ratings[team] = round(1500 + (1 - reversion) * (elo_ratings[team] - 1500), 1)
    _rewrite_source(
        updated_points_data, elo_ratings, recent_form,
        remaining_matches, MATCHES_COMMITTED, committed_results
    )
    print(f"✅ Elo reset for new season (reversion={reversion}):")
    for team, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"   {team}: {rating}")


# ---------------------------------------------------------------------------
# FILE REWRITER — FIXED TO PRESERVE INDENTATION/COLUMN ALIGNMENT
# ---------------------------------------------------------------------------
def _rewrite_source(new_points_data, new_elo, new_form, new_remaining, new_committed, new_committed_results):
    src_path = os.path.abspath(__file__)
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    pd_lines = "{\n"
    for k, v in new_points_data.items():
        pd_lines += f'    "{k}": {json.dumps(v)},\n'
    pd_lines += "}"
    source = re.sub(
        r'(updated_points_data\s*=\s*\{).*?(\n\})',
        lambda m: m.group(1) + "\n" + pd_lines[2:],
        source, flags=re.DOTALL
    )

    elo_lines = "{\n"
    for k, v in new_elo.items():
        pad = " " * max(1, 38 - len(k))
        elo_lines += f'    "{k}":{pad}{v},\n'
    elo_lines += "}"
    source = re.sub(
        r'(elo_ratings\s*=\s*\{).*?(\n\})',
        lambda m: m.group(1) + "\n" + elo_lines[2:],
        source, flags=re.DOTALL
    )

    rf_lines = "{\n"
    for k, v in new_form.items():
        pad = " " * max(1, 38 - len(k))
        rf_lines += f'    "{k}":{pad}{json.dumps(v)},\n'
    rf_lines += "}"
    source = re.sub(
        r'(recent_form\s*=\s*\{).*?(\n\})',
        lambda m: m.group(1) + "\n" + rf_lines[2:],
        source, flags=re.DOTALL
    )

    rm_lines = "[\n"
    for m in new_remaining:
        h_pad = " " * max(0, 28 - len(m["home"]))
        a_pad = " " * max(0, 28 - len(m["away"]))
        v_pad = " " * max(0, 16 - len(m["venue"]))
        rm_lines += f'    {{"home": "{m["home"]}",{h_pad}"away": "{m["away"]}",{a_pad}"venue": "{m["venue"]}",{v_pad}"result": None, "margin": None, "applied": False}},\n'
    rm_lines += "]"
    source = re.sub(
        r'(remaining_matches\s*=\s*\[).*?(\n\])',
        lambda m: m.group(1) + "\n" + rm_lines[2:],
        source, flags=re.DOTALL
    )

    source = re.sub(
        r'(MATCHES_COMMITTED\s*=\s*)\d+',
        lambda m: m.group(1) + str(new_committed),
        source
    )

    cr_json = json.dumps(new_committed_results, separators=(', ', ': '))
    cr_py = cr_json.replace(': true', ': True').replace(': false', ': False').replace(': null', ': None')
    source = re.sub(
        r'(committed_results\s*=\s*).*?(  # END_COMMITTED_RESULTS)',
        lambda m: m.group(1) + cr_py + '  # END_COMMITTED_RESULTS',
        source, flags=re.DOTALL
    )

    with open(src_path, "w", encoding="utf-8") as f:
        f.write(source)


# ---------------------------------------------------------------------------
# commit_result
# ---------------------------------------------------------------------------
def commit_result(home, away, winner, winner_runs, winner_overs_str,
                  loser_runs, loser_overs_str, abandoned=False):
    global MATCHES_COMMITTED, remaining_matches, committed_results

    new_remaining = [
        m for m in remaining_matches
        if not (m["home"] == home and m["away"] == away)
    ]
    if len(new_remaining) == len(remaining_matches):
        raise ValueError(f"Match '{home} vs {away}' not found in remaining_matches.")

    loser = away if winner == home else home

    snapshot = {
        "home": home,
        "away": away,
        "venue": next(m["venue"] for m in remaining_matches
                      if m["home"] == home and m["away"] == away),
        "winner": winner,
        "abandoned": abandoned,
        "winner_runs": winner_runs,
        "winner_overs_str": winner_overs_str,
        "loser_runs": loser_runs,
        "loser_overs_str": loser_overs_str,
        "elo_before": {home: elo_ratings[home], away: elo_ratings[away]},
        "form_before": {home: list(recent_form[home]), away: list(recent_form[away])},
    }

    if abandoned:
        for team in [home, away]:
            updated_points_data[team]["points"] += 1
            updated_points_data[team]["matches"] += 1
            recent_form[team].append(9)
            recent_form[team] = recent_form[team][-5:]
    else:
        wo = overs_to_float(winner_overs_str)
        lo = overs_to_float(loser_overs_str)

        updated_points_data[winner]["points"] += 2
        updated_points_data[winner]["matches"] += 1
        updated_points_data[winner]["runs_for"] += winner_runs
        updated_points_data[winner]["overs_faced"] = _add_overs(updated_points_data[winner]["overs_faced"],
                                                                winner_overs_str)
        updated_points_data[winner]["runs_against"] += loser_runs
        updated_points_data[winner]["overs_bowled"] = _add_overs(updated_points_data[winner]["overs_bowled"],
                                                                 loser_overs_str)

        updated_points_data[loser]["matches"] += 1
        updated_points_data[loser]["runs_for"] += loser_runs
        updated_points_data[loser]["overs_faced"] = _add_overs(updated_points_data[loser]["overs_faced"],
                                                               loser_overs_str)
        updated_points_data[loser]["runs_against"] += winner_runs
        updated_points_data[loser]["overs_bowled"] = _add_overs(updated_points_data[loser]["overs_bowled"],
                                                                winner_overs_str)

        nrr_delta = (winner_runs / wo - loser_runs / lo) if wo > 0 and lo > 0 else 0.0
        _update_elo_margin(winner, loser, nrr_delta)

        for team, res in [(winner, 1), (loser, 0)]:
            recent_form[team].append(res)
            recent_form[team] = recent_form[team][-5:]

    MATCHES_COMMITTED += 1
    remaining_matches = new_remaining
    committed_results = committed_results + [snapshot]

    _rewrite_source(
        updated_points_data, elo_ratings, recent_form,
        new_remaining, MATCHES_COMMITTED, committed_results
    )

    print(f"✅ Match #{MATCHES_COMMITTED} committed: {home} vs {away} → "
          f"{'Abandoned' if abandoned else winner}")


# ---------------------------------------------------------------------------
# decommit_last
# ---------------------------------------------------------------------------
def decommit_last():
    global MATCHES_COMMITTED, remaining_matches, committed_results

    if not committed_results:
        raise ValueError("No committed results to undo.")

    snap = committed_results[-1]
    home = snap["home"]
    away = snap["away"]
    winner = snap["winner"]
    loser = away if winner == home else home
    abandoned = snap["abandoned"]

    if abandoned:
        for team in [home, away]:
            updated_points_data[team]["points"] -= 1
            updated_points_data[team]["matches"] -= 1
    else:
        wr = snap["winner_runs"]
        lr = snap["loser_runs"]
        wo_str = snap["winner_overs_str"]
        lo_str = snap["loser_overs_str"]

        def _sub_overs(existing_str, sub_str):
            total_balls = round((overs_to_float(existing_str) - overs_to_float(sub_str)) * 6)
            return f"{total_balls // 6}.{total_balls % 6}"

        updated_points_data[winner]["points"] -= 2
        updated_points_data[winner]["matches"] -= 1
        updated_points_data[winner]["runs_for"] -= wr
        updated_points_data[winner]["overs_faced"] = _sub_overs(updated_points_data[winner]["overs_faced"], wo_str)
        updated_points_data[winner]["runs_against"] -= lr
        updated_points_data[winner]["overs_bowled"] = _sub_overs(updated_points_data[winner]["overs_bowled"], lo_str)

        updated_points_data[loser]["matches"] -= 1
        updated_points_data[loser]["runs_for"] -= lr
        updated_points_data[loser]["overs_faced"] = _sub_overs(updated_points_data[loser]["overs_faced"], lo_str)
        updated_points_data[loser]["runs_against"] -= wr
        updated_points_data[loser]["overs_bowled"] = _sub_overs(updated_points_data[loser]["overs_bowled"], wo_str)

        elo_ratings[home] = snap["elo_before"][home]
        elo_ratings[away] = snap["elo_before"][away]
        recent_form[home] = snap["form_before"][home]
        recent_form[away] = snap["form_before"][away]

    restored_match = {
        "home": home, "away": away, "venue": snap["venue"],
        "result": None, "margin": None, "applied": False
    }
    remaining_matches = [restored_match] + list(remaining_matches)
    committed_results = committed_results[:-1]
    MATCHES_COMMITTED -= 1

    _rewrite_source(
        updated_points_data, elo_ratings, recent_form,
        remaining_matches, MATCHES_COMMITTED, committed_results
    )

    print(f"↩️  Decommitted match #{MATCHES_COMMITTED + 1}: {home} vs {away}")


# ---------------------------------------------------------------------------
# What-if setter
# ---------------------------------------------------------------------------
def set_what_if_results(new_remaining_matches):
    global remaining_matches
    remaining_matches = new_remaining_matches


# ---------------------------------------------------------------------------
# Pre-match win probability
# ---------------------------------------------------------------------------
def get_win_probability(home, away, venue):
    elo_norm = {t: np.clip((elo_ratings[t] - 1350) / 300, 0.1, 0.9) for t in teams}
    form_scores = {t: get_form_score(t) for t in teams}
    nrr_scores = {t: np.clip((calculate_nrr(updated_points_data[t]) + 3) / 6, 0.0, 1.0) for t in teams}

    win_pcts = {
        t: (updated_points_data[t]["points"] + 2) / ((updated_points_data[t]["matches"] + 2) * 2)
        for t in teams
    }

    mp = max(d["matches"] for d in updated_points_data.values())
    form_weight = min(0.30, 0.06 * mp)
    elo_weight = max(0.45, 0.55 - 0.02 * mp)
    winpct_weight = 0.15
    nrr_weight = 1.0 - elo_weight - form_weight - winpct_weight

    raw = {
        t: max(0.05, elo_weight * elo_norm[t] + form_weight * form_scores[t] +
               winpct_weight * win_pcts[t] + nrr_weight * nrr_scores[t])
        for t in teams
    }
    total = sum(raw.values())
    hw = {t: raw[t] / total for t in teams}

    boost = get_home_boost(venue)
    sh = hw[home] * boost
    sa = hw[away]
    home_p = round(sh / (sh + sa) * 100, 1)
    return home_p, round(100 - home_p, 1)


# ---------------------------------------------------------------------------
# Shared helper: build Elo-weighted match probabilities
# ---------------------------------------------------------------------------
def _build_match_probs(pending):
    """Return a list of (home, away, venue, home_win_prob) for pending matches."""
    elo_norm = {t: np.clip((elo_ratings[t] - 1350) / 300, 0.1, 0.9) for t in teams}
    form_scores_local = {t: get_form_score(t) for t in teams}
    nrr_scores = {t: np.clip((calculate_nrr(updated_points_data[t]) + 3) / 6, 0.0, 1.0) for t in teams}
    win_pcts = {
        t: (updated_points_data[t]["points"] + 2) / ((updated_points_data[t]["matches"] + 2) * 2)
        for t in teams
    }
    mp = max(d["matches"] for d in updated_points_data.values())
    form_weight = min(0.30, 0.06 * mp)
    elo_weight = max(0.45, 0.55 - 0.02 * mp)
    winpct_weight = 0.15
    nrr_weight = 1.0 - elo_weight - form_weight - winpct_weight
    raw = {
        t: max(0.05, elo_weight * elo_norm[t] + form_weight * form_scores_local[t] +
               winpct_weight * win_pcts[t] + nrr_weight * nrr_scores[t])
        for t in teams
    }
    total_raw = sum(raw.values())
    hw = {t: raw[t] / total_raw for t in teams}

    result = []
    for m in pending:
        boost = get_home_boost(m.get("venue", ""))
        sh = hw[m["home"]] * boost
        sa = hw[m["away"]]
        result.append((m["home"], m["away"], m.get("venue", ""), sh / (sh + sa)))
    return result


# ---------------------------------------------------------------------------
# generate_random_scorecard — RETIRED, replaced by generate_constrained_scorecard
# (loser overs were not always 20; winner margin was not capped at 0–6 runs)
# ---------------------------------------------------------------------------
# def generate_random_scorecard(home, away, venue):
#     home_p, away_p = get_win_probability(home, away, venue)
#     winner = home if np.random.rand() < home_p / 100 else away
#     loser = away if winner == home else home
#
#     first_innings_runs = int(np.clip(np.random.normal(185, 30), 66, 287))
#     chase_won = np.random.rand() < 0.52
#
#     if chase_won:
#         loser_runs = first_innings_runs
#         loser_overs = "20.0"
#         if np.random.rand() < 0.40:
#             balls_remaining = np.random.randint(1, 18)
#             total_balls = 120 - balls_remaining
#             winner_overs = f"{total_balls // 6}.{total_balls % 6}"
#         else:
#             winner_overs = "20.0"
#         winner_runs = loser_runs + np.random.randint(1, 8)
#         winner_runs = loser_runs + np.random.randint(0, 5)
#     else:
#         winner_runs = first_innings_runs
#         winner_overs = "20.0"
#         if np.random.rand() < 0.35:
#             balls = np.random.randint(90, 119)
#             loser_overs = f"{balls // 6}.{balls % 6}"
#         else:
#             loser_overs = "20.0"
#         loser_runs = winner_runs - np.random.randint(5, 51)
#         loser_runs = max(loser_runs, 80)
#
#     return {
#         "winner": winner,
#         "loser": loser,
#         "winner_runs": int(winner_runs),
#         "winner_overs": winner_overs,
#         "loser_runs": int(loser_runs),
#         "loser_overs": loser_overs,
#     }


# ---------------------------------------------------------------------------
# Constrained scorecard — used by God Mode, Pathfinder & Randomise All
#   · Loser always plays full 20 overs
#   · Winner beats loser by exactly 0–6 runs
#   · Winner's overs & loser's runs follow the same distribution as above
# ---------------------------------------------------------------------------
def generate_constrained_scorecard(home, away, venue):
    home_p, away_p = get_win_probability(home, away, venue)
    winner = home if np.random.rand() < home_p / 100 else away
    loser = away if winner == home else home

    first_innings_runs = int(np.clip(np.random.normal(185, 30), 66, 287))
    chase_won = np.random.rand() < 0.52

    # FIX: Loser ALWAYS counts as 20.0 overs for accurate NRR penalty
    loser_overs = "20.0"

    if chase_won:
        # WINNER BATS SECOND (The Run Chase)
        loser_runs = first_innings_runs
        run_diff = np.random.randint(0, 7)
        winner_runs = loser_runs + run_diff
        
        if run_diff == 0:
            # FIX: If it is a tie/Super Over, the chasing team MUST have used all 20 overs
            winner_overs = "20.0"
        else:
            if np.random.rand() < 0.40:
                balls_remaining = np.random.randint(1, 18)
                total_balls = 120 - balls_remaining
                winner_overs = f"{total_balls // 6}.{total_balls % 6}"
            else:
                winner_overs = "20.0"
            
    else:
        # WINNER BATS FIRST (Defending a total)
        winner_runs = first_innings_runs
        winner_overs = "20.0" 
            
        loser_runs = winner_runs - np.random.randint(0, 51)
        loser_runs = max(loser_runs, 80)

    return {
        "winner": winner,
        "loser": loser,
        "winner_runs": int(winner_runs),
        "winner_overs": winner_overs,
        "loser_runs": int(loser_runs),
        "loser_overs": loser_overs,
    }





# ---------------------------------------------------------------------------
# What-if Baseline Generator
# ---------------------------------------------------------------------------
def get_what_if_baseline(matches):
    base_td = {
        team: {
            "points": updated_points_data[team]["points"],
            "matches": updated_points_data[team]["matches"],
            "runs_for": updated_points_data[team]["runs_for"],
            "overs_faced": overs_to_float(updated_points_data[team]["overs_faced"]),
            "runs_against": updated_points_data[team]["runs_against"],
            "overs_bowled": overs_to_float(updated_points_data[team]["overs_bowled"]),
        }
        for team in teams
    }

    pending_matches = []

    for match in matches:
        if match.get("applied") and match.get("result"):
            h, a, r = match["home"], match["away"], match["result"]
            if r == "Abandoned/No Result (1 point each)":
                for t in [h, a]:
                    base_td[t]["points"] += 1
                    base_td[t]["matches"] += 1
                continue

            w = r
            l = a if w == h else h

            base_td[w]["points"] += 2
            base_td[w]["matches"] += 1
            base_td[l]["matches"] += 1

            try:
                wr = match.get("runs", {}).get(w)
                lr = match.get("runs", {}).get(l)
                wo = overs_to_float(match.get("overs", {}).get(w, "0.0"))
                lo = overs_to_float(match.get("overs", {}).get(l, "0.0"))

                if None not in (wr, lr) and wo > 0 and lo > 0:
                    base_td[w]["runs_for"] += wr
                    base_td[w]["overs_faced"] += wo
                    base_td[w]["runs_against"] += lr
                    base_td[w]["overs_bowled"] += lo

                    base_td[l]["runs_for"] += lr
                    base_td[l]["overs_faced"] += lo
                    base_td[l]["runs_against"] += wr
                    base_td[l]["overs_bowled"] += wo
            except Exception:
                pass
        else:
            pending_matches.append(match)

    base_pts = {t: base_td[t]["points"] for t in teams}
    base_nrr = {t: calculate_nrr(base_td[t]) for t in teams}
    return base_pts, base_nrr, pending_matches, base_td


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def run_adjusted_simulation(num_simulations, what_if=False, override_matches=None):
    matches = override_matches if override_matches is not None else remaining_matches
    base_pts, base_nrr, pending, _ = get_what_if_baseline(matches)

    if not pending:
        sorted_t = sorted(teams, key=lambda t: (base_pts[t], base_nrr[t]), reverse=True)
        pts_5th = base_pts[sorted_t[4]]
        pts_3rd = base_pts[sorted_t[2]]
        rows = []
        for rank, t in enumerate(sorted_t):
            qualify = 100.0 if rank < 4 else 0.0
            top2 = 100.0 if rank < 2 else 0.0
            safe4 = 100.0 if base_pts[t] > pts_5th else 0.0
            safe2 = 100.0 if base_pts[t] > pts_3rd else 0.0
            rows.append((t, qualify, top2, safe4, safe2, None, None, base_pts[t], round(base_nrr[t], 3)))
        return pd.DataFrame(rows, columns=["Team", "Qualify %", "Top 2 %", "Safe by Points %", "Safe Top 2 %",
                                           "Still Possible %", "Top 2 Still Possible %", "Avg Final Points",
                                           "Avg Final NRR"])

    elo_norm = {t: np.clip((elo_ratings[t] - 1350) / 300, 0.1, 0.9) for t in teams}
    form_scores = {t: get_form_score(t) for t in teams}
    nrr_scores = {t: np.clip((calculate_nrr(updated_points_data[t]) + 3) / 6, 0.0, 1.0) for t in teams}
    win_pcts = {
        t: (updated_points_data[t]["points"] + 2) / ((updated_points_data[t]["matches"] + 2) * 2)
        for t in teams
    }

    mp = max(d["matches"] for d in updated_points_data.values())
    form_weight = min(0.30, 0.06 * mp)
    elo_weight = max(0.45, 0.55 - 0.02 * mp)
    winpct_weight = 0.15
    nrr_weight = 1.0 - elo_weight - form_weight - winpct_weight

    raw = {
        t: max(0.05, elo_weight * elo_norm[t] + form_weight * form_scores[t] +
               winpct_weight * win_pcts[t] + nrr_weight * nrr_scores[t])
        for t in teams
    }
    total = sum(raw.values())
    hw = {t: raw[t] / total for t in teams}

    top4_c = {t: 0 for t in teams}
    top2_c = {t: 0 for t in teams}
    top4_pts = {t: 0 for t in teams}
    top2_pts = {t: 0 for t in teams}
    cumulative_pts = {t: 0.0 for t in teams}
    cumulative_nrr = {t: 0.0 for t in teams}

    for _ in range(num_simulations):
        pts = base_pts.copy()
        nrrs = base_nrr.copy()

        for match in pending:
            h, a = match["home"], match["away"]
            boost = get_home_boost(match.get("venue", ""))
            sh = hw[h] * boost
            sa = hw[a]
            w, l = (h, a) if np.random.rand() < sh / (sh + sa) else (a, h)
            pts[w] += 2
            mg = simulate_nrr_change(hw[w], hw[l])
            nrrs[w] += mg
            nrrs[l] -= mg

        vals = sorted(pts.values(), reverse=True)
        f5 = vals[4]
        f3 = vals[2]

        st_sorted = sorted(teams, key=lambda t: (pts[t], nrrs[t]), reverse=True)
        for i, t in enumerate(st_sorted):
            if i < 4: top4_c[t] += 1
            if i < 2: top2_c[t] += 1
            if pts[t] > f5: top4_pts[t] += 1
            if pts[t] > f3: top2_pts[t] += 1
            cumulative_pts[t] += pts[t]
            cumulative_nrr[t] += nrrs[t]

    n = num_simulations
    q4 = {t: round(top4_c[t] / n * 100, 2) for t in teams}
    q2 = {t: round(top2_c[t] / n * 100, 2) for t in teams}
    c4 = {t: round(top4_pts[t] / n * 100, 2) for t in teams}
    c2 = {t: round(top2_pts[t] / n * 100, 2) for t in teams}
    avg_pts = {t: int(round(cumulative_pts[t] / n / 2) * 2) for t in teams}
    avg_nrr = {t: round(cumulative_nrr[t] / n, 3) for t in teams}

    sorted_t = sorted(teams, key=lambda t: q4[t], reverse=True)
    return pd.DataFrame(
        [(t, q4[t], q2[t], c4[t], c2[t], None, None, avg_pts[t], avg_nrr[t]) for t in sorted_t],
        columns=["Team", "Qualify %", "Top 2 %", "Safe by Points %",
                 "Safe Top 2 %", "Still Possible %", "Top 2 Still Possible %",
                 "Avg Final Points", "Avg Final NRR"]
    )


# ---------------------------------------------------------------------------
# Points table helpers
# ---------------------------------------------------------------------------
def get_current_points_table():
    table = []
    for team in teams:
        d = updated_points_data[team]
        of = overs_to_float(d["overs_faced"])
        ob = overs_to_float(d["overs_bowled"])
        nrr = round(d["runs_for"] / of - d["runs_against"] / ob, 3) if of > 0 and ob > 0 else 0.0
        table.append({"Team": team, "Matches": d["matches"], "Points": d["points"], "NRR": nrr})
    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))


def get_points_table_after_what_if(what_if_matches):
    _, _, _, base_td = get_what_if_baseline(what_if_matches)
    table = []
    for team, d in base_td.items():
        of = d["overs_faced"]
        ob = d["overs_bowled"]
        nrr = round(d["runs_for"] / of - d["runs_against"] / ob, 3) if of > 0 and ob > 0 else 0.0
        table.append({"Team": team, "Matches": d["matches"], "Points": d["points"], "NRR": nrr})
    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))


# ---------------------------------------------------------------------------
# Pure Math simulation worker — now also tracks safe4 / safe2
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pure Math simulation worker
# ---------------------------------------------------------------------------
def run_pure_math_worker(args):
    seed, sims, base_pts, pending = args
    np.random.seed(seed)
    top4 = {t: 0 for t in teams}
    top2 = {t: 0 for t in teams}
    safe4 = {t: 0 for t in teams}
    safe2 = {t: 0 for t in teams}

    for _ in range(sims):
        pts = base_pts.copy()
        for m in pending:
            pts[np.random.choice([m["home"], m["away"]])] += 2

        st_sorted = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        f4 = st_sorted[3][1]
        f2 = st_sorted[1][1]

        # FIX: Compare against 5th place (index 4) and 3rd place (index 2)
        f5_pts = st_sorted[4][1]
        f3_pts = st_sorted[2][1]

        for t in teams:
            if pts[t] > f5_pts:
                safe4[t] += 1
            if pts[t] > f3_pts:
                safe2[t] += 1

        above4 = [t for t, p in st_sorted if p > f4]
        tied4 = [t for t, p in st_sorted if p == f4]
        spots4 = 4 - len(above4)
        for t in above4: top4[t] += 1
        if spots4 > 0 and tied4:
            for t in tied4: top4[t] += spots4 / len(tied4)

        above2 = [t for t, p in st_sorted if p > f2]
        tied2 = [t for t, p in st_sorted if p == f2]
        spots2 = 2 - len(above2)
        for t in above2: top2[t] += 1
        if spots2 > 0 and tied2:
            for t in tied2: top2[t] += spots2 / len(tied2)

    return {"top4": top4, "top2": top2, "safe4": safe4, "safe2": safe2}


def run_pure_math_simulation_parallel(total_sims=10000, processes=4, override_matches=None):
    matches = override_matches if override_matches is not None else remaining_matches
    base_pts, base_nrr, pending, _ = get_what_if_baseline(matches)

    if not pending:
        st_sorted = sorted(teams, key=lambda t: (base_pts[t], base_nrr[t]), reverse=True)
        in4 = set(st_sorted[:4])
        in2 = set(st_sorted[:2])

        # FIX: Compare against 5th and 3rd at the end of the season
        f5_pts = base_pts[st_sorted[4]]
        f3_pts = base_pts[st_sorted[2]]
        return {
            "top4": {t: 100.0 if t in in4 else 0.0 for t in teams},
            "top2": {t: 100.0 if t in in2 else 0.0 for t in teams},
            "safe4": {t: 100.0 if base_pts[t] > f5_pts else 0.0 for t in teams},
            "safe2": {t: 100.0 if base_pts[t] > f3_pts else 0.0 for t in teams},
        }

    spc = total_sims // processes
    seeds = np.random.randint(0, 1_000_000_000, size=processes)
    with ThreadPoolExecutor(max_workers=processes) as ex:
        results = list(ex.map(run_pure_math_worker,
                              [(s, spc, base_pts, pending) for s in seeds]))

    combined4 = {t: sum(r["top4"][t] for r in results) for t in teams}
    combined2 = {t: sum(r["top2"][t] for r in results) for t in teams}
    combined_s4 = {t: sum(r["safe4"][t] for r in results) for t in teams}
    combined_s2 = {t: sum(r["safe2"][t] for r in results) for t in teams}

    return {
        "top4": {t: round(combined4[t] / total_sims * 100, 2) for t in teams},
        "top2": {t: round(combined2[t] / total_sims * 100, 2) for t in teams},
        "safe4": {t: round(combined_s4[t] / total_sims * 100, 2) for t in teams},
        "safe2": {t: round(combined_s2[t] / total_sims * 100, 2) for t in teams},
    }


# ---------------------------------------------------------------------------
# NEW: Tragic Status Helper
# ---------------------------------------------------------------------------
def calculate_tragic_status(pure_math_res, pending_count):
    """
    Assigns definitive status badges based purely on mathematical possibility.
    """
    status_map = {}
    for t in teams:
        possible = pure_math_res["top4"][t]

        if possible >= 100.0:
            status_map[t] = "✅ QUALIFIED"
        elif possible <= 0.0:
            status_map[t] = "❌ ELIMINATED"
        elif possible <= 7.5:
            status_map[t] = "⚠️ E1(Must Win)"
        elif possible <= 15.0:
            status_map[t] = "📉 E2(Struggling)"
        else:
            status_map[t] = "🏏 In Hunt"

    return status_map


# ---------------------------------------------------------------------------
# Parallel main simulation
# ---------------------------------------------------------------------------
def parallel_worker(args):
    seed, sims, what_if, matches = args
    np.random.seed(seed)
    return run_adjusted_simulation(sims, what_if=what_if, override_matches=matches)


def run_parallel_simulations(total_sims=10000, processes=4, override_matches=None):
    spc = total_sims // processes
    seeds = np.random.randint(0, 1_000_000_000, size=processes)
    mu = override_matches if override_matches is not None else remaining_matches
    with ThreadPoolExecutor(max_workers=processes) as ex:
        results = list(ex.map(parallel_worker,
                              [(s, spc, True, copy.deepcopy(mu)) for s in seeds]))
    final = pd.concat(results)
    g = final.groupby("Team").agg({
        "Qualify %": "mean",
        "Top 2 %": "mean",
        "Safe by Points %": "mean",
        "Safe Top 2 %": "mean",
        "Avg Final Points": "mean",
        "Avg Final NRR": "mean",
    }).reset_index()

    g["Avg Final Points"] = g["Avg Final Points"].apply(lambda x: int(round(x / 2) * 2))
    g["Avg Final NRR"] = g["Avg Final NRR"].round(3)

    return g.sort_values(by=["Qualify %", "Top 2 %", "Avg Final Points", "Avg Final NRR"],
                         ascending=[False, False, False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# NEW: Exact Combination Counter
# ---------------------------------------------------------------------------
def count_combination(team_combo, num_sims=50000, override_matches=None):
    """
    Calculate what % of Elo-weighted simulations result in *exactly* team_combo
    as the top 4 (using random NRR tie-breaking at the boundary).

    Args:
        team_combo : list/set of team names (2–4 teams)
        num_sims   : number of Monte Carlo iterations
        override_matches : optional what-if match list

    Returns:
        float — percentage (0–100)
    """
    matches = override_matches if override_matches is not None else remaining_matches
    base_pts, base_nrr, pending, _ = get_what_if_baseline(matches)
    combo_set = set(team_combo)

    if not pending:
        sorted_t = sorted(teams, key=lambda t: (base_pts[t], base_nrr[t]), reverse=True)
        actual_top4 = set(sorted_t[:4])
        return 100.0 if actual_top4 == combo_set else 0.0

    match_probs = _build_match_probs(pending)

    count = 0
    for _ in range(num_sims):
        pts = base_pts.copy()
        for h, a, _venue, prob in match_probs:
            winner = h if np.random.rand() < prob else a
            pts[winner] += 2

        sorted_by_pts = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        f4 = sorted_by_pts[3][1]

        above4 = [t for t, p in sorted_by_pts if p > f4]
        tied4  = [t for t, p in sorted_by_pts if p == f4]
        spots4 = 4 - len(above4)

        if spots4 <= 0 or not tied4:
            top4 = set(above4[:4])
        else:
            n_pick = min(spots4, len(tied4))
            chosen = np.random.choice(tied4, size=n_pick, replace=False).tolist()
            top4 = set(above4) | set(chosen)

        if top4 == combo_set:
            count += 1

    return round(count / num_sims * 100, 2)


# ---------------------------------------------------------------------------
# Core sorting logic (To be used in ALL simulation functions)
# ---------------------------------------------------------------------------
def _get_sorted_standings(pts_dict, nrr_dict):
    """
    Sorts teams by Points, then NRR, then a random tie-breaker.
    Ensures mathematical consistency even in perfect ties.
    """
    # Create a random shuffle for the ultimate tie-breaker
    tie_breaker_list = list(teams)
    np.random.seed()  # Ensure fresh randomness
    np.random.shuffle(tie_breaker_list)
    tb_rank = {team: i for i, team in enumerate(tie_breaker_list)}

    # Sort: Points (Primary), NRR (Secondary), tb_rank (Tertiary)
    return sorted(teams, key=lambda t: (pts_dict[t], nrr_dict[t], tb_rank[t]), reverse=True)


# ---------------------------------------------------------------------------
# UPDATED: God Mode & Pathfinder using main scorecard logic
# ---------------------------------------------------------------------------
def generate_forced_scenario(forced_top4, max_attempts=200_000, override_matches=None):
    matches = override_matches if override_matches is not None else remaining_matches
    base_pts, _, pending, base_td = get_what_if_baseline(matches)
    forced_set = set(forced_top4)
    match_probs = _build_match_probs(pending)

    for attempt in range(max_attempts):
        td = {t: dict(base_td[t]) for t in teams}
        match_results = []

        for h, a, venue, prob in match_probs:
            # Loaded Dice
            if h in forced_set and a not in forced_set:
                actual_p = 0.82
            elif a in forced_set and h not in forced_set:
                actual_p = 0.18
            else:
                actual_p = prob

            # Use constrained scorecard: loser always 20 overs, winner wins by 0–6 runs
            sc = generate_constrained_scorecard(h, a, venue)
            # Override winner based on loaded dice
            winner = h if np.random.rand() < actual_p else a
            loser = a if winner == h else h

            wr, lr = sc["winner_runs"], sc["loser_runs"]
            wo, lo = sc["winner_overs"], sc["loser_overs"]

            td[winner]["points"] += 2
            td[winner]["runs_for"] += wr
            td[winner]["overs_faced"] += overs_to_float(wo)
            td[winner]["runs_against"] += lr
            td[winner]["overs_bowled"] += overs_to_float(lo)

            td[loser]["runs_for"] += lr
            td[loser]["overs_faced"] += overs_to_float(lo)
            td[loser]["runs_against"] += wr
            td[loser]["overs_bowled"] += overs_to_float(wo)

            match_results.append({
                "Winner": winner, "winner_runs": wr, "winner_overs": wo,
                "loser_runs": lr, "loser_overs": lo
            })

        final_pts = {t: td[t]["points"] for t in teams}
        final_nrr = {t: calculate_nrr(td[t]) for t in teams}
        st_sorted = _get_sorted_standings(final_pts, final_nrr)

        # Realism Filter: Exactly 4 teams AND top team <= 22 points
        if set(st_sorted[:4]) == forced_set and max(final_pts.values()) <= 22:
            return match_results, final_pts, attempt + 1

    return None, None, max_attempts


def generate_single_team_route(target_team, target_rank, max_attempts=100_000):
    base_pts, _, pending, base_td = get_what_if_baseline(remaining_matches)
    match_probs = _build_match_probs(pending)

    for attempt in range(max_attempts):
        td = {t: dict(base_td[t]) for t in teams}
        results = []
        for h, a, venue, prob in match_probs:
            # Bias ONLY for the target team
            if h == target_team:
                actual_p = 0.85
            elif a == target_team:
                actual_p = 0.15
            else:
                actual_p = prob

            winner = h if np.random.rand() < actual_p else a
            loser = a if winner == h else h
            
            sc = generate_constrained_scorecard(h, a, venue)
            
            wr, lr = sc["winner_runs"], sc["loser_runs"]
            wo, lo = sc["winner_overs"], sc["loser_overs"]

            # FIX: Properly applying runs and overs for NRR tracking
            td[winner]["points"] += 2
            td[winner]["runs_for"] += wr
            td[winner]["overs_faced"] += overs_to_float(wo)
            td[winner]["runs_against"] += lr
            td[winner]["overs_bowled"] += overs_to_float(lo)

            td[loser]["runs_for"] += lr
            td[loser]["overs_faced"] += overs_to_float(lo)
            td[loser]["runs_against"] += wr
            td[loser]["overs_bowled"] += overs_to_float(wo)

            results.append({"Winner": winner, "winner_runs": wr, "winner_overs": wo,
                            "loser_runs": lr, "loser_overs": lo})

        final_pts = {t: td[t]["points"] for t in teams}
        final_nrr = {t: calculate_nrr(td[t]) for t in teams}
        st = _get_sorted_standings(final_pts, final_nrr)

        actual_rank = st.index(target_team) + 1
        success = (actual_rank == target_rank) if isinstance(target_rank, int) else (actual_rank <= 4)

        if success and max(final_pts.values()) <= 22:
            return results, final_pts, attempt + 1
            
    return None, None, max_attempts



# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def fancy_highlight_half_split(df):
    pct_cols = [
        "Qualify %", "Top 2 %",
        "Safe by Points %", "Safe Top 2 %",
        "Still Possible %", "Top 2 Still Possible %",
        "Math Safe by Pts %", "Math Safe Top 2 %",   # new columns
    ]

    def color_by_pct(val):
        if pd.isna(val): return ""
        try:
            val = float(val)
        except:
            return ""
        if val == 100.00: return "background-color: #301934; color: white; font-weight: bold"
        if val == 0.00:   return "background-color: #36454F; color: white"
        if 0.01 <= val <= 0.99: return "background-color: #580000; color: white"
        if 1.00 <= val <= 44.99:
            step = int(val // 5)
            cs = ["#880000", "#A02020", "#B03030", "#C04040", "#D05030",
                  "#E06030", "#EF7020", "#F88020", "#FF9020"]
            return f"background-color: {cs[min(step, len(cs) - 1)]}; color: white"
        if 45.00 <= val <= 50.00: return "background-color: #FFFF66; color: black"
        if 50.01 <= val <= 99.99:
            gi = int(245 - ((val - 50.01) / 49.98) * 150)
            rb = int(168 - ((val - 50.01) / 49.98) * 168)
            return f"background-color: #{'%02X%02X%02X' % (rb, gi, rb)}; color: black"
        return ""

    def color_avg_pts(val):
        if pd.isna(val): return ""
        try:
            val = float(val)
        except:
            return ""
        intensity = min(max((val - 8) / 10, 0), 1)
        r = int(20 + (1 - intensity) * 60)
        g = int(60 + (1 - intensity) * 80)
        b = int(120 + (1 - intensity) * 80)
        return f"background-color: rgb({r},{g},{b}); color: white"

    def color_avg_nrr(val):
        if pd.isna(val): return ""
        try:
            val = float(val)
        except:
            return ""
        if val > 0:
            intensity = min(val / 0.5, 1.0)
            g = int(120 + intensity * 100)
            return f"background-color: rgb(20,{g},40); color: white"
        elif val < 0:
            intensity = min(abs(val) / 0.5, 1.0)
            r = int(120 + intensity * 100)
            return f"background-color: rgb({r},20,20); color: white"
        return "background-color: #36454F; color: white"

    fmt = {c: "{:.2f}" for c in pct_cols if c in df.columns}
    if "Avg Final Points" in df.columns:
        fmt["Avg Final Points"] = "{:d}"
    if "Avg Final NRR" in df.columns:
        fmt["Avg Final NRR"] = "{:.3f}"
    styled = df.style.format(fmt)

    for col in pct_cols:
        if col in df.columns:
            styled = styled.map(color_by_pct, subset=col)
    if "Avg Final Points" in df.columns:
        styled = styled.map(color_avg_pts, subset=["Avg Final Points"])
    if "Avg Final NRR" in df.columns:
        styled = styled.map(color_avg_nrr, subset=["Avg Final NRR"])

    styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})
    styled = styled.set_properties(**{"text-align": "center", "vertical-align": "middle"})

    def color_team(val):
        if val in team_colors:
            c = team_colors[val]
            return f"background-color: {c['bg']}; color: {c['text']}"
        return ""

    styled = styled.map(color_team, subset=["Team"])
    return styled


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def run_full_simulation_and_prompt():
    df = run_parallel_simulations(10000, processes=16)
    pure_math = run_pure_math_simulation_parallel(10000, processes=16)

    df["Still Possible %"] = df["Team"].map(pure_math["top4"])
    df["Top 2 Still Possible %"] = df["Team"].map(pure_math["top2"])
    df["Math Safe by Pts %"] = df["Team"].map(pure_math["safe4"])
    df["Math Safe Top 2 %"]  = df["Team"].map(pure_math["safe2"])

    styled_df = fancy_highlight_half_split(df)
    print(df)

    suggested = f"m{MATCHES_COMMITTED}"
    user_input = input(f"Enter match ID (Enter for '{suggested}', or 'skip'): ").strip().lower()
    match_id = suggested if user_input == "" else (None if user_input == "skip" else user_input)

    if match_id:
        fmt = input("Save format? 'csv', 'excel', or 'both': ").strip().lower()
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs("results", exist_ok=True)
        if fmt in ["csv", "both"]: df.to_csv(f"results/post_{match_id}_results_{ts}.csv", index=False)
        if fmt in ["excel", "both"]: styled_df.to_excel(f"results/post_{match_id}_stylized_{ts}.xlsx", index=False)
        print("✅ Saved.")
    else:
        print("⚠️ Skipped.")



if __name__ == "__main__":
    run_full_simulation_and_prompt()