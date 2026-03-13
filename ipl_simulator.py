import os
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
# IPL 2026 — reset points table (season not started)
# ---------------------------------------------------------------------------
updated_points_data = {
    "Royal Challengers Bengaluru": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Punjab Kings": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Mumbai Indians": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Gujarat Titans": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Delhi Capitals": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Kolkata Knight Riders": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Lucknow Super Giants": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Sunrisers Hyderabad": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Rajasthan Royals": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    },
    "Chennai Super Kings": {
        "points": 0, "matches": 0,
        "runs_for": 0, "overs_faced": "0.0",
        "runs_against": 0, "overs_bowled": "0.0"
    }
}

# ---------------------------------------------------------------------------
# Elo ratings — seeded from 2025 season performance + squad changes for 2026
# Update via update_elo() after each real match is committed
# ---------------------------------------------------------------------------
elo_ratings = {
    "Royal Challengers Bengaluru": 1530,  # Defending champions
    "Gujarat Titans":              1510,
    "Mumbai Indians":              1505,
    "Punjab Kings":                1495,
    "Kolkata Knight Riders":       1490,
    "Delhi Capitals":              1480,
    "Lucknow Super Giants":        1470,
    "Sunrisers Hyderabad":         1465,
    "Rajasthan Royals":            1455,
    "Chennai Super Kings":         1450,
}

ELO_K = 32  # Higher K = faster adaptation, appropriate for a short tournament

def update_elo(winner, loser):
    """Call this each time you commit a real match result."""
    wr = elo_ratings[winner]
    lr = elo_ratings[loser]
    expected_w = 1 / (1 + 10 ** ((lr - wr) / 400))
    elo_ratings[winner] = round(elo_ratings[winner] + ELO_K * (1 - expected_w), 2)
    elo_ratings[loser]  = round(elo_ratings[loser]  + ELO_K * (0 - (1 - expected_w)), 2)

# ---------------------------------------------------------------------------
# Recent form — last 5 results per team (1 = win, 0 = loss), oldest first
# Update by appending to the list and trimming to last 5 after each match
# ---------------------------------------------------------------------------
recent_form = {
    "Royal Challengers Bengaluru": [],
    "Gujarat Titans":              [],
    "Mumbai Indians":              [],
    "Punjab Kings":                [],
    "Kolkata Knight Riders":       [],
    "Delhi Capitals":              [],
    "Lucknow Super Giants":        [],
    "Sunrisers Hyderabad":         [],
    "Rajasthan Royals":            [],
    "Chennai Super Kings":         [],
}

FORM_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30]  # oldest → newest

def update_recent_form(winner, loser):
    """Call alongside update_elo() after each real match."""
    for team, result in [(winner, 1), (loser, 0)]:
        recent_form[team].append(result)
        recent_form[team] = recent_form[team][-5:]  # keep last 5 only

def get_form_score(team):
    results = recent_form[team]
    if not results:
        return 0.5  # neutral prior when no games played yet
    w = FORM_WEIGHTS[-len(results):]  # align weights to available results
    total_w = sum(w)
    return sum(wi * r for wi, r in zip(w, results)) / total_w

# ---------------------------------------------------------------------------
# Venue-specific home advantage multipliers
# ---------------------------------------------------------------------------
home_advantage = {
    "MUMBAI":     1.08,
    "CHENNAI":    1.07,
    "KOLKATA":    1.06,
    "BENGALURU":  1.05,
    "HYDERABAD":  1.05,
    "AHMEDABAD":  1.04,
    "JAIPUR":     1.04,
    "DELHI":      1.03,
    "LUCKNOW":    1.03,
    "MULLANPUR":  1.03,
    "GUWAHATI":   1.02,  # Neutral-ish venue used by RR/CSK
    "DHARAMSALA": 1.03,
}

def get_home_boost(venue):
    return home_advantage.get(venue.upper().strip(), 1.03)

# ---------------------------------------------------------------------------
# IPL 2026 Schedule — first 20 matches from ESPNcricinfo
# Total matches in league stage = 70; remaining will be added progressively
# ---------------------------------------------------------------------------
remaining_matches = [
    {"home": "Royal Challengers Bengaluru", "away": "Sunrisers Hyderabad",  "venue": "Bengaluru",  "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians",              "away": "Kolkata Knight Riders", "venue": "Mumbai",     "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals",            "away": "Chennai Super Kings",   "venue": "Guwahati",   "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings",                "away": "Gujarat Titans",        "venue": "Mullanpur",  "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants",        "away": "Delhi Capitals",        "venue": "Lucknow",    "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders",       "away": "Sunrisers Hyderabad",   "venue": "Kolkata",    "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings",         "away": "Punjab Kings",          "venue": "Chennai",    "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals",              "away": "Mumbai Indians",        "venue": "Delhi",      "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans",              "away": "Rajasthan Royals",      "venue": "Ahmedabad",  "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad",         "away": "Lucknow Super Giants",  "venue": "Hyderabad",  "result": None, "margin": None, "applied": False},
    {"home": "Royal Challengers Bengaluru", "away": "Chennai Super Kings",   "venue": "Bengaluru",  "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders",       "away": "Punjab Kings",          "venue": "Kolkata",    "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals",            "away": "Mumbai Indians",        "venue": "Guwahati",   "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals",              "away": "Gujarat Titans",        "venue": "Delhi",      "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders",       "away": "Lucknow Super Giants",  "venue": "Kolkata",    "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals",            "away": "Royal Challengers Bengaluru", "venue": "Guwahati", "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings",                "away": "Sunrisers Hyderabad",   "venue": "Mullanpur",  "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings",         "away": "Delhi Capitals",        "venue": "Chennai",    "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants",        "away": "Gujarat Titans",        "venue": "Lucknow",    "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians",              "away": "Royal Challengers Bengaluru", "venue": "Mumbai", "result": None, "margin": None, "applied": False},
]

TOTAL_MATCHES = 80       # IPL 2026 expanded league stage
MATCHES_COMMITTED = 0   # increment this manually as you comment out real results

def set_what_if_results(new_remaining_matches):
    global remaining_matches
    remaining_matches = new_remaining_matches
    print("\n--- WHAT-IF MATCHES RECEIVED ---")
    for match in remaining_matches:
        if match.get("applied"):
            print(match)

# ---------------------------------------------------------------------------
# Improved NRR noise model
# ---------------------------------------------------------------------------
def simulate_nrr_change(winner_weight, loser_weight):
    strength_diff = winner_weight - loser_weight
    # Base reflects typical competitive T20 margin; scales with mismatch
    base = np.clip(0.15 + 0.4 * strength_diff, 0.05, 0.50)
    noise = np.random.normal(0, 0.12)
    raw = base + noise
    # Asymmetric clip: small negatives possible (narrow wins), cap blowouts at 0.8
    return round(np.clip(raw, -0.05, 0.80), 3)

def overs_to_float(overs_str):
    if isinstance(overs_str, (int, float)):
        return float(overs_str)
    try:
        parts = str(overs_str).split(".")
        whole = int(parts[0])
        balls = int(parts[1]) if len(parts) > 1 else 0
        return round(whole + balls / 6, 3)
    except:
        return 0.0

def calculate_nrr(team_data):
    rf = team_data["runs_for"]
    of = overs_to_float(team_data["overs_faced"])
    ra = team_data["runs_against"]
    ob = overs_to_float(team_data["overs_bowled"])
    if of == 0 or ob == 0:
        return 0.0
    return round((rf / of) - (ra / ob), 3)

# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def run_adjusted_simulation(num_simulations, what_if=False, override_matches=None):
    matches = override_matches if override_matches is not None else remaining_matches

    # --- Build hybrid strength weights ---
    # Elo component
    max_elo = max(elo_ratings.values())
    min_elo = min(elo_ratings.values())
    elo_norm = {
        t: (elo_ratings[t] - min_elo) / (max_elo - min_elo + 1e-9)
        for t in teams
    }

    # Recent form component
    form_scores = {t: get_form_score(t) for t in teams}

    # NRR component from current table
    nrr_scores = {}
    for team, data in updated_points_data.items():
        nrr = calculate_nrr(data)
        nrr_scores[team] = (nrr + 2) / 4  # normalise to ~0-1

    # Win % component
    win_pcts = {}
    for team, data in updated_points_data.items():
        m = data["matches"]
        win_pcts[team] = (data["points"] / (m * 2)) if m > 0 else 0.5

    # Blend: Elo carries most weight early, form grows as matches accumulate
    matches_played = max(d["matches"] for d in updated_points_data.values())
    form_weight  = min(0.30, 0.06 * matches_played)   # grows from 0 → 0.30
    elo_weight   = max(0.45, 0.75 - 0.03 * matches_played)  # shrinks 0.75 → 0.45
    winpct_weight = 0.15
    nrr_weight   = 1.0 - elo_weight - form_weight - winpct_weight

    raw_strength = {}
    for t in teams:
        raw_strength[t] = (
            elo_weight   * elo_norm[t] +
            form_weight  * form_scores[t] +
            winpct_weight * win_pcts[t] +
            nrr_weight   * nrr_scores[t]
        )

    total = sum(raw_strength.values())
    hybrid_weights = {t: raw_strength[t] / total for t in teams}

    # --- Apply What-if results to base data ---
    base_team_data = {
        team: {
            "points":      data["points"],
            "matches":     data["matches"],
            "runs_for":    data["runs_for"],
            "overs_faced": overs_to_float(data["overs_faced"]),
            "runs_against":data["runs_against"],
            "overs_bowled":overs_to_float(data["overs_bowled"]),
        }
        for team, data in updated_points_data.items()
    }

    for match in matches:
        if match.get("applied") and match.get("result") in [
            match["home"], match["away"], "Abandoned/No Result (1 point each)"
        ]:
            home   = match["home"]
            away   = match["away"]
            result = match["result"]

            if result == "Abandoned/No Result (1 point each)":
                base_team_data[home]["points"]  += 1
                base_team_data[away]["points"]  += 1
                base_team_data[home]["matches"] += 1
                base_team_data[away]["matches"] += 1
                continue

            winner = result
            loser  = away if winner == home else home

            try:
                wr = match["runs"][winner]
                wo = overs_to_float(match["overs"][winner])
                lr = match["runs"][loser]
                lo = overs_to_float(match["overs"][loser])
            except Exception as e:
                print(f"Invalid What-if format for match: {match}")
                continue

            base_team_data[winner]["points"]      += 2
            base_team_data[winner]["matches"]     += 1
            base_team_data[winner]["runs_for"]    += wr
            base_team_data[winner]["overs_faced"] += wo
            base_team_data[winner]["runs_against"] += lr
            base_team_data[winner]["overs_bowled"] += lo

            base_team_data[loser]["matches"]      += 1
            base_team_data[loser]["runs_for"]     += lr
            base_team_data[loser]["overs_faced"]  += lo
            base_team_data[loser]["runs_against"] += wr
            base_team_data[loser]["overs_bowled"] += wo

    base_points = {team: base_team_data[team]["points"] for team in teams}
    base_nrr    = {team: calculate_nrr(base_team_data[team]) for team in teams}

    top4_counts                  = {team: 0 for team in teams}
    top2_counts                  = {team: 0 for team in teams}
    top4_confirmed_points_only   = {team: 0 for team in teams}
    top2_confirmed_points_only   = {team: 0 for team in teams}
    cumulative_points            = {team: 0 for team in teams}
    cumulative_nrr               = {team: 0.0 for team in teams}

    for _ in range(num_simulations):
        points = base_points.copy()
        nrrs   = base_nrr.copy()

        for match in matches:
            if match.get("applied"):
                continue
            home   = match["home"]
            away   = match["away"]
            venue  = match.get("venue", "")

            # Home advantage applied here
            boost  = get_home_boost(venue)
            s_home = hybrid_weights[home] * boost
            s_away = hybrid_weights[away]

            prob_home_win = s_home / (s_home + s_away)
            winner, loser = (home, away) if np.random.rand() < prob_home_win else (away, home)

            points[winner] += 2
            margin = simulate_nrr_change(hybrid_weights[winner], hybrid_weights[loser])
            nrrs[winner] += margin
            nrrs[loser]  -= margin

        # Points-only sort for confirmed columns
        sorted_points_only = sorted(points.items(), key=lambda x: x[1], reverse=True)
        fifth_points = sorted_points_only[4][1]
        third_points = sorted_points_only[2][1]

        for team, pts in points.items():
            if pts > fifth_points:
                top4_confirmed_points_only[team] += 1
            if pts > third_points:
                top2_confirmed_points_only[team] += 1

        sorted_teams = sorted(teams, key=lambda t: (points[t], nrrs[t]), reverse=True)
        for team in sorted_teams[:4]:
            top4_counts[team] += 1
        for team in sorted_teams[:2]:
            top2_counts[team] += 1

        for team in teams:
            cumulative_points[team] += points[team]
            cumulative_nrr[team]    += nrrs[team]

    qualifications           = {team: round((top4_counts[team] / num_simulations) * 100, 2) for team in teams}
    top2_qualifications      = {team: round((top2_counts[team] / num_simulations) * 100, 2) for team in teams}
    top4_conf_pct            = {team: round((top4_confirmed_points_only[team] / num_simulations) * 100, 2) for team in teams}
    top2_conf_pct            = {team: round((top2_confirmed_points_only[team] / num_simulations) * 100, 2) for team in teams}

    sorted_by_qual = sorted(teams, key=lambda t: qualifications[t], reverse=True)

    results = pd.DataFrame([
        (team, qualifications[team], top2_qualifications[team],
         top4_conf_pct[team], top2_conf_pct[team], None)
        for team in sorted_by_qual
    ], columns=[
        "Team", "Top 4 (%)", "Top 2 (%)",
        "Top 4 Confirmed (%)", "Top 2 Confirmed (%)",
        "Top 4 Pure Math (%)"
    ])

    return results

# ---------------------------------------------------------------------------
# Points table helpers
# ---------------------------------------------------------------------------
def get_current_points_table():
    table = []
    for team in teams:
        data = updated_points_data[team]
        of = overs_to_float(data["overs_faced"])
        ob = overs_to_float(data["overs_bowled"])
        nrr = round((data["runs_for"] / of) - (data["runs_against"] / ob), 3) if of > 0 and ob > 0 else 0.0
        table.append({
            "Team":    team,
            "Points":  data["points"],
            "Matches": data["matches"],
            "NRR":     nrr
        })
    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))


def get_points_table_after_what_if(what_if_matches):
    team_data = {
        team: {
            "points":      updated_points_data[team]["points"],
            "matches":     updated_points_data[team]["matches"],
            "runs_for":    updated_points_data[team]["runs_for"],
            "overs_faced": overs_to_float(updated_points_data[team]["overs_faced"]),
            "runs_against":updated_points_data[team]["runs_against"],
            "overs_bowled":overs_to_float(updated_points_data[team]["overs_bowled"]),
        }
        for team in teams
    }

    for match in what_if_matches:
        if match.get("applied") and match.get("result") in [
            match["home"], match["away"], "Abandoned/No Result (1 point each)"
        ]:
            home   = match["home"]
            away   = match["away"]
            result = match["result"]

            if result == "Abandoned/No Result (1 point each)":
                team_data[home]["points"]  += 1
                team_data[away]["points"]  += 1
                team_data[home]["matches"] += 1
                team_data[away]["matches"] += 1
                continue

            winner = result
            loser  = away if winner == home else home

            wr = match["runs"].get(winner)
            lr = match["runs"].get(loser)
            wo = overs_to_float(match["overs"].get(winner))
            lo = overs_to_float(match["overs"].get(loser))

            if None in (wr, lr, wo, lo):
                continue

            team_data[winner]["points"]       += 2
            team_data[winner]["matches"]      += 1
            team_data[winner]["runs_for"]     += wr
            team_data[winner]["overs_faced"]  += wo
            team_data[winner]["runs_against"] += lr
            team_data[winner]["overs_bowled"] += lo

            team_data[loser]["matches"]       += 1
            team_data[loser]["runs_for"]      += lr
            team_data[loser]["overs_faced"]   += lo
            team_data[loser]["runs_against"]  += wr
            team_data[loser]["overs_bowled"]  += wo

    table = []
    for team, data in team_data.items():
        of  = data["overs_faced"]
        ob  = data["overs_bowled"]
        nrr = round(data["runs_for"] / of - data["runs_against"] / ob, 3) if of > 0 and ob > 0 else 0.0
        table.append({"Team": team, "Points": data["points"], "Matches": data["matches"], "NRR": nrr})

    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))

# ---------------------------------------------------------------------------
# Pure Math simulation (parallel)
# ---------------------------------------------------------------------------
def run_pure_math_worker(args):
    seed, sims, matches = args
    np.random.seed(seed)
    top4_counts = {team: 0 for team in teams}
    base_points = {team: updated_points_data[team]["points"] for team in teams}

    for match in matches:
        if match.get("applied") and match.get("result"):
            result = match["result"]
            if result == "Abandoned/No Result (1 point each)":
                base_points[match["home"]] += 1
                base_points[match["away"]] += 1
            elif result in base_points:
                base_points[result] += 2

    for _ in range(sims):
        points = base_points.copy()
        for match in matches:
            if match.get("applied") and match.get("result"):
                continue
            winner = np.random.choice([match["home"], match["away"]])
            points[winner] += 2

        sorted_teams       = sorted(points.items(), key=lambda x: x[1], reverse=True)
        fourth_place_pts   = sorted_teams[3][1]
        above = [t for t, p in sorted_teams if p > fourth_place_pts]
        tied  = [t for t, p in sorted_teams if p == fourth_place_pts]
        spots = 4 - len(above)

        for t in above:
            top4_counts[t] += 1
        if spots > 0 and tied:
            for t in tied:
                top4_counts[t] += spots / len(tied)

    return top4_counts


def run_pure_math_simulation_parallel(total_sims=10000, processes=4, override_matches=None):
    sims_per_core = total_sims // processes
    seeds   = np.random.randint(0, 1_000_000_000, size=processes)
    matches = override_matches if override_matches is not None else remaining_matches

    with ThreadPoolExecutor(max_workers=processes) as executor:
        results = list(executor.map(
            run_pure_math_worker,
            [(seed, sims_per_core, copy.deepcopy(matches)) for seed in seeds]
        ))

    combined = {team: 0 for team in teams}
    for partial in results:
        for team in teams:
            combined[team] += partial[team]

    return {team: round((combined[team] / total_sims) * 100, 2) for team in teams}

# ---------------------------------------------------------------------------
# Parallel main simulation
# ---------------------------------------------------------------------------
def parallel_worker(args):
    seed, sims, what_if, matches = args
    np.random.seed(seed)
    return run_adjusted_simulation(sims, what_if=what_if, override_matches=matches)


def run_parallel_simulations(total_sims=10000, processes=4, override_matches=None):
    sims_per_core  = total_sims // processes
    seeds          = np.random.randint(0, 1_000_000_000, size=processes)
    matches_to_use = override_matches if override_matches is not None else remaining_matches

    with ThreadPoolExecutor(max_workers=processes) as executor:
        results = list(executor.map(
            parallel_worker,
            [(seed, sims_per_core, True, copy.deepcopy(matches_to_use)) for seed in seeds]
        ))

    final_df = pd.concat(results)
    grouped  = final_df.groupby("Team").agg({
        "Top 4 (%)":           "mean",
        "Top 2 (%)":           "mean",
        "Top 4 Confirmed (%)": "mean",
        "Top 2 Confirmed (%)": "mean",
    }).reset_index()
    grouped = grouped.sort_values(
        by=["Top 4 (%)", "Top 2 (%)"], ascending=[False, False]
    ).reset_index(drop=True)
    return grouped

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def fancy_highlight_half_split(df):
    percentage_columns = [
        "Top 2 (%)", "Top 4 (%)", "Top 2 Confirmed (%)",
        "Top 4 Confirmed (%)", "Top 4 Pure Math (%)"
    ]

    def color_by_percentage(val):
        if pd.isna(val):
            return ""
        try:
            val = float(val)
        except:
            return ""
        if val == 100.00:
            return "background-color: #301934; color: white; font-weight: bold"
        if val == 0.00:
            return "background-color: #36454F; color: white"
        if 0.01 <= val <= 0.99:
            return "background-color: #580000; color: white"
        if 1.00 <= val <= 44.99:
            step = int(val // 5)
            red_values = [
                "#880000","#A02020","#B03030","#C04040","#D05030",
                "#E06030","#EF7020","#F88020","#FF9020"
            ]
            color = red_values[min(step, len(red_values) - 1)]
            return f"background-color: {color}; color: white"
        if 45.00 <= val <= 50.00:
            return "background-color: #FFFF66; color: black"
        if 50.01 <= val <= 99.99:
            green_intensity = int(245 - ((val - 50.01) / 49.98) * 150)
            red_blue = int(168 - ((val - 50.01) / 49.98) * 168)
            hex_color = '#{0:02X}{1:02X}{2:02X}'.format(red_blue, green_intensity, red_blue)
            return f"background-color: {hex_color}; color: black"
        return ""

    styled = df.style.format({
        "Top 2 (%)":           "{:.2f}",
        "Top 4 (%)":           "{:.2f}",
        "Top 2 Confirmed (%)": "{:.2f}",
        "Top 4 Confirmed (%)": "{:.2f}",
        "Top 4 Pure Math (%)": "{:.2f}",
    })

    for col in percentage_columns:
        styled = styled.map(color_by_percentage, subset=col)

    styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})
    styled = styled.set_properties(**{"text-align": "center", "vertical-align": "middle"})

    def color_team_cells(val):
        if val in team_colors:
            c = team_colors[val]
            return f"background-color: {c['bg']}; color: {c['text']}"
        return ""

    styled = styled.map(color_team_cells, subset=["Team"])
    return styled

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def run_full_simulation_and_prompt():
    df         = run_parallel_simulations(10000, processes=16)
    pure_math  = run_pure_math_simulation_parallel(10000, processes=16)
    df["Top 4 Pure Math (%)"] = df["Team"].map(pure_math)
    styled_df  = fancy_highlight_half_split(df)
    print(df)

    default_match_number = MATCHES_COMMITTED
    suggested_match_id   = f"m{default_match_number}"
    user_input = input(
        f"Enter match ID (press Enter for '{suggested_match_id}', or 'skip'): "
    ).strip().lower()

    match_id = suggested_match_id if user_input == "" else (None if user_input == "skip" else user_input)

    if match_id:
        format_input = input("Save format? 'csv', 'excel', or 'both': ").strip().lower()
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir   = "results"
        os.makedirs(output_dir, exist_ok=True)

        csv_path  = f"{output_dir}/post_{match_id}_results_{timestamp}.csv"
        xlsx_path = f"{output_dir}/post_{match_id}_stylized_{timestamp}.xlsx"

        if format_input in ["csv",   "both"]: df.to_csv(csv_path, index=False)
        if format_input in ["excel", "both"]: styled_df.to_excel(xlsx_path, index=False)

        print("\n✅ Saved:")
        if format_input in ["csv",   "both"]: print(f"  [CSV]   {csv_path}")
        if format_input in ["excel", "both"]: print(f"  [Excel] {xlsx_path}")
    else:
        print("\n⚠️ Skipped file saving.")


if __name__ == "__main__":
    run_full_simulation_and_prompt()