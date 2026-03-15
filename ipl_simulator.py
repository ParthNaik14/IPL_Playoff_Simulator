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
    "Royal Challengers Bengaluru": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Punjab Kings": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Mumbai Indians": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Gujarat Titans": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Delhi Capitals": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Kolkata Knight Riders": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Lucknow Super Giants": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Sunrisers Hyderabad": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Rajasthan Royals": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
    "Chennai Super Kings": {"points": 0, "matches": 0, "runs_for": 0, "overs_faced": "0.0", "runs_against": 0, "overs_bowled": "0.0"},
}

# ---------------------------------------------------------------------------
# Elo ratings — auto-updated by commit_result (margin-aware)
# ---------------------------------------------------------------------------
elo_ratings = {
    "Royal Challengers Bengaluru":           1530,
    "Gujarat Titans":                        1510,
    "Mumbai Indians":                        1505,
    "Punjab Kings":                          1495,
    "Kolkata Knight Riders":                 1490,
    "Delhi Capitals":                        1480,
    "Lucknow Super Giants":                  1470,
    "Sunrisers Hyderabad":                   1465,
    "Rajasthan Royals":                      1455,
    "Chennai Super Kings":                   1450,
}

ELO_K = 32

# ---------------------------------------------------------------------------
# Recent form — last 5 results, auto-updated by commit_result
# ---------------------------------------------------------------------------
recent_form = {
    "Royal Challengers Bengaluru":           [],
    "Gujarat Titans":                        [],
    "Mumbai Indians":                        [],
    "Punjab Kings":                          [],
    "Kolkata Knight Riders":                 [],
    "Delhi Capitals":                        [],
    "Lucknow Super Giants":                  [],
    "Sunrisers Hyderabad":                   [],
    "Rajasthan Royals":                      [],
    "Chennai Super Kings":                   [],
}

FORM_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30]  # oldest to newest

def get_form_score(team):
    results = recent_form[team]
    if not results:
        return 0.5
    w = FORM_WEIGHTS[-len(results):]
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
    "GUWAHATI":   1.02,
    "DHARAMSALA": 1.03,
}

def get_home_boost(venue):
    return home_advantage.get(venue.upper().strip(), 1.03)

# ---------------------------------------------------------------------------
# IPL 2026 Schedule — first 20 matches
# Completed matches are removed from this list automatically by commit_result()
# ---------------------------------------------------------------------------
remaining_matches = [
    {"home": "Royal Challengers Bengaluru", "away": "Sunrisers Hyderabad", "venue": "Bengaluru", "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians", "away": "Kolkata Knight Riders", "venue": "Mumbai", "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals", "away": "Chennai Super Kings", "venue": "Guwahati", "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings", "away": "Gujarat Titans", "venue": "Mullanpur", "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants", "away": "Delhi Capitals", "venue": "Lucknow", "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders", "away": "Sunrisers Hyderabad", "venue": "Kolkata", "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings", "away": "Punjab Kings", "venue": "Chennai", "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals", "away": "Mumbai Indians", "venue": "Delhi", "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans", "away": "Rajasthan Royals", "venue": "Ahmedabad", "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad", "away": "Lucknow Super Giants", "venue": "Hyderabad", "result": None, "margin": None, "applied": False},
    {"home": "Royal Challengers Bengaluru", "away": "Chennai Super Kings", "venue": "Bengaluru", "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders", "away": "Punjab Kings", "venue": "Kolkata", "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals", "away": "Mumbai Indians", "venue": "Guwahati", "result": None, "margin": None, "applied": False},
    {"home": "Delhi Capitals", "away": "Gujarat Titans", "venue": "Delhi", "result": None, "margin": None, "applied": False},
    {"home": "Kolkata Knight Riders", "away": "Lucknow Super Giants", "venue": "Kolkata", "result": None, "margin": None, "applied": False},
    {"home": "Rajasthan Royals", "away": "Royal Challengers Bengaluru", "venue": "Guwahati", "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings", "away": "Sunrisers Hyderabad", "venue": "Mullanpur", "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings", "away": "Delhi Capitals", "venue": "Chennai", "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants", "away": "Gujarat Titans", "venue": "Lucknow", "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians", "away": "Royal Challengers Bengaluru", "venue": "Mumbai", "result": None, "margin": None, "applied": False},
]

TOTAL_MATCHES     = 80  # IPL 2026 expanded league stage
MATCHES_COMMITTED = 0   # auto-incremented by commit_result()

# ---------------------------------------------------------------------------
# Committed results log — auto-managed, do not edit manually
# Each entry stores everything needed to fully reverse a commit
# ---------------------------------------------------------------------------
committed_results = []  # END_COMMITTED_RESULTS

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
    """Add two over strings preserving cricket ball notation."""
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
    base  = np.clip(0.15 + 0.4 * strength_diff, 0.05, 0.50)
    noise = np.random.normal(0, 0.12)
    return round(np.clip(base + noise, -0.05, 0.80), 3)

# ---------------------------------------------------------------------------
# Elo update — margin-aware
# ---------------------------------------------------------------------------
def _update_elo_margin(winner, loser, winner_nrr_delta):
    """
    Margin-aware Elo: a big win moves ratings more than a narrow one.
    winner_nrr_delta = (winner_runs/winner_overs) - (loser_runs/loser_overs)
    """
    wr = elo_ratings[winner]
    lr = elo_ratings[loser]
    expected_w   = 1 / (1 + 10 ** ((lr - wr) / 400))
    # tanh maps NRR delta smoothly: 0 → 0.5, 0.5 → ~0.77, 1.0 → ~0.93
    margin_score = 0.5 + 0.5 * np.tanh(winner_nrr_delta / 0.4)
    elo_ratings[winner] = round(wr + ELO_K * (margin_score - expected_w), 2)
    elo_ratings[loser]  = round(lr + ELO_K * ((1 - margin_score) - (1 - expected_w)), 2)

# ---------------------------------------------------------------------------
# FILE REWRITER — patches mutable blocks in ipl_simulator.py in-place
# ---------------------------------------------------------------------------
def _rewrite_source(new_points_data, new_elo, new_form, new_remaining, new_committed, new_committed_results):
    src_path = os.path.abspath(__file__)
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    # --- updated_points_data ---
    pd_lines = "{\n"
    for k, v in new_points_data.items():
        pd_lines += f'    "{k}": {json.dumps(v)},\n'
    pd_lines += "}"
    source = re.sub(
        r'(updated_points_data\s*=\s*\{).*?(\n\})',
        lambda m: m.group(1) + "\n" + pd_lines[2:],
        source, flags=re.DOTALL
    )

    # --- elo_ratings ---
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

    # --- recent_form ---
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

    # --- remaining_matches ---
    rm_lines = "[\n"
    for m in new_remaining:
        rm_lines += (
            f'    {{"home": "{m["home"]}", "away": "{m["away"]}", '
            f'"venue": "{m["venue"]}", "result": None, "margin": None, "applied": False}},\n'
        )
    rm_lines += "]"
    source = re.sub(
        r'(remaining_matches\s*=\s*\[).*?(\n\])',
        lambda m: m.group(1) + "\n" + rm_lines[2:],
        source, flags=re.DOTALL
    )

    # --- MATCHES_COMMITTED ---
    source = re.sub(
        r'(MATCHES_COMMITTED\s*=\s*)\d+',
        lambda m: m.group(1) + str(new_committed),
        source
    )

    # --- committed_results ---
    cr_json = json.dumps(new_committed_results, separators=(', ', ': '))
    cr_py   = cr_json.replace(': true', ': True').replace(': false', ': False').replace(': null', ': None')
    source = re.sub(
        r'(committed_results\s*=\s*).*?(  # END_COMMITTED_RESULTS)',
        lambda m: m.group(1) + cr_py + '  # END_COMMITTED_RESULTS',
        source, flags=re.DOTALL
    )

    with open(src_path, "w", encoding="utf-8") as f:
        f.write(source)

# ---------------------------------------------------------------------------
# commit_result — call from Streamlit UI after each real match
# ---------------------------------------------------------------------------
def commit_result(home, away, winner, winner_runs, winner_overs_str,
                  loser_runs, loser_overs_str, abandoned=False):
    """
    Permanently commits a real match result.
    - Updates points table, Elo, recent form
    - Logs entry to committed_results (enables decommit)
    - Removes match from remaining_matches
    - Increments MATCHES_COMMITTED
    - Rewrites ipl_simulator.py so state survives restarts and git pushes
    """
    global MATCHES_COMMITTED, remaining_matches, committed_results

    # Validate match exists
    new_remaining = [
        m for m in remaining_matches
        if not (m["home"] == home and m["away"] == away)
    ]
    if len(new_remaining) == len(remaining_matches):
        raise ValueError(f"Match '{home} vs {away}' not found in remaining_matches.")

    loser = away if winner == home else home

    # Snapshot state BEFORE applying changes — used by decommit to reverse
    snapshot = {
        "home":              home,
        "away":              away,
        "venue":             next(m["venue"] for m in remaining_matches
                                 if m["home"] == home and m["away"] == away),
        "winner":            winner,
        "abandoned":         abandoned,
        "winner_runs":       winner_runs,
        "winner_overs_str":  winner_overs_str,
        "loser_runs":        loser_runs,
        "loser_overs_str":   loser_overs_str,
        "elo_before":        {home: elo_ratings[home], away: elo_ratings[away]},
        "form_before":       {home: list(recent_form[home]), away: list(recent_form[away])},
    }

    if abandoned:
        for team in [home, away]:
            updated_points_data[team]["points"]  += 1
            updated_points_data[team]["matches"] += 1
    else:
        wo = overs_to_float(winner_overs_str)
        lo = overs_to_float(loser_overs_str)

        updated_points_data[winner]["points"]       += 2
        updated_points_data[winner]["matches"]      += 1
        updated_points_data[winner]["runs_for"]     += winner_runs
        updated_points_data[winner]["overs_faced"]   = _add_overs(updated_points_data[winner]["overs_faced"], winner_overs_str)
        updated_points_data[winner]["runs_against"] += loser_runs
        updated_points_data[winner]["overs_bowled"]  = _add_overs(updated_points_data[winner]["overs_bowled"], loser_overs_str)

        updated_points_data[loser]["matches"]       += 1
        updated_points_data[loser]["runs_for"]      += loser_runs
        updated_points_data[loser]["overs_faced"]    = _add_overs(updated_points_data[loser]["overs_faced"], loser_overs_str)
        updated_points_data[loser]["runs_against"]  += winner_runs
        updated_points_data[loser]["overs_bowled"]   = _add_overs(updated_points_data[loser]["overs_bowled"], winner_overs_str)

        nrr_delta = (winner_runs / wo - loser_runs / lo) if wo > 0 and lo > 0 else 0.0
        _update_elo_margin(winner, loser, nrr_delta)

        for team, res in [(winner, 1), (loser, 0)]:
            recent_form[team].append(res)
            recent_form[team] = recent_form[team][-5:]

    MATCHES_COMMITTED  += 1
    remaining_matches   = new_remaining
    committed_results   = committed_results + [snapshot]

    _rewrite_source(
        updated_points_data, elo_ratings, recent_form,
        new_remaining, MATCHES_COMMITTED, committed_results
    )

    print(f"✅ Match #{MATCHES_COMMITTED} committed: {home} vs {away} → "
          f"{'Abandoned' if abandoned else winner}")

# ---------------------------------------------------------------------------
# decommit_last — reverses the most recent commit_result call
# ---------------------------------------------------------------------------
def decommit_last():
    """
    Reverses the most recent committed result.
    - Restores points table, Elo, recent form to pre-commit state
    - Re-inserts match at the front of remaining_matches
    - Decrements MATCHES_COMMITTED
    - Rewrites ipl_simulator.py
    """
    global MATCHES_COMMITTED, remaining_matches, committed_results

    if not committed_results:
        raise ValueError("No committed results to undo.")

    snap = committed_results[-1]
    home     = snap["home"]
    away     = snap["away"]
    winner   = snap["winner"]
    loser    = away if winner == home else home
    abandoned = snap["abandoned"]

    if abandoned:
        for team in [home, away]:
            updated_points_data[team]["points"]  -= 1
            updated_points_data[team]["matches"] -= 1
    else:
        wr = snap["winner_runs"]
        lr = snap["loser_runs"]
        wo_str = snap["winner_overs_str"]
        lo_str = snap["loser_overs_str"]

        # Reverse points table — subtract overs using negative ball count
        def _sub_overs(existing_str, sub_str):
            total_balls = round((overs_to_float(existing_str) - overs_to_float(sub_str)) * 6)
            return f"{total_balls // 6}.{total_balls % 6}"

        updated_points_data[winner]["points"]       -= 2
        updated_points_data[winner]["matches"]      -= 1
        updated_points_data[winner]["runs_for"]     -= wr
        updated_points_data[winner]["overs_faced"]   = _sub_overs(updated_points_data[winner]["overs_faced"], wo_str)
        updated_points_data[winner]["runs_against"] -= lr
        updated_points_data[winner]["overs_bowled"]  = _sub_overs(updated_points_data[winner]["overs_bowled"], lo_str)

        updated_points_data[loser]["matches"]       -= 1
        updated_points_data[loser]["runs_for"]      -= lr
        updated_points_data[loser]["overs_faced"]    = _sub_overs(updated_points_data[loser]["overs_faced"], lo_str)
        updated_points_data[loser]["runs_against"]  -= wr
        updated_points_data[loser]["overs_bowled"]   = _sub_overs(updated_points_data[loser]["overs_bowled"], wo_str)

        # Restore Elo and recent form exactly from snapshot
        elo_ratings[home] = snap["elo_before"][home]
        elo_ratings[away] = snap["elo_before"][away]
        recent_form[home] = snap["form_before"][home]
        recent_form[away] = snap["form_before"][away]

    # Re-insert match at the front of remaining_matches
    restored_match = {
        "home": home, "away": away, "venue": snap["venue"],
        "result": None, "margin": None, "applied": False
    }
    remaining_matches  = [restored_match] + list(remaining_matches)
    committed_results  = committed_results[:-1]
    MATCHES_COMMITTED -= 1

    _rewrite_source(
        updated_points_data, elo_ratings, recent_form,
        remaining_matches, MATCHES_COMMITTED, committed_results
    )

    print(f"↩️  Decommitted match #{MATCHES_COMMITTED + 1}: {home} vs {away}")

# ---------------------------------------------------------------------------
# What-if setter (Streamlit sidebar)
# ---------------------------------------------------------------------------
def set_what_if_results(new_remaining_matches):
    global remaining_matches
    remaining_matches = new_remaining_matches

# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def run_adjusted_simulation(num_simulations, what_if=False, override_matches=None):
    matches = override_matches if override_matches is not None else remaining_matches

    max_elo = max(elo_ratings.values())
    min_elo = min(elo_ratings.values())
    elo_norm    = {t: (elo_ratings[t] - min_elo) / (max_elo - min_elo + 1e-9) for t in teams}
    form_scores = {t: get_form_score(t) for t in teams}
    nrr_scores  = {t: (calculate_nrr(updated_points_data[t]) + 2) / 4 for t in teams}
    win_pcts    = {
        t: (updated_points_data[t]["points"] / (updated_points_data[t]["matches"] * 2))
        if updated_points_data[t]["matches"] > 0 else 0.5
        for t in teams
    }

    mp            = max(d["matches"] for d in updated_points_data.values())
    form_weight   = min(0.30, 0.06 * mp)
    elo_weight    = max(0.45, 0.75 - 0.03 * mp)
    winpct_weight = 0.15
    nrr_weight    = 1.0 - elo_weight - form_weight - winpct_weight

    raw = {
        t: elo_weight * elo_norm[t] + form_weight * form_scores[t] +
           winpct_weight * win_pcts[t] + nrr_weight * nrr_scores[t]
        for t in teams
    }
    total = sum(raw.values())
    hw = {t: raw[t] / total for t in teams}  # hybrid weights

    # Base data after applying any committed what-if results
    base_td = {
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
            h, a, r = match["home"], match["away"], match["result"]
            if r == "Abandoned/No Result (1 point each)":
                for t in [h, a]:
                    base_td[t]["points"]  += 1
                    base_td[t]["matches"] += 1
                continue
            w = r; l = a if w == h else h
            try:
                wr = match["runs"][w]; wo = overs_to_float(match["overs"][w])
                lr = match["runs"][l]; lo = overs_to_float(match["overs"][l])
            except Exception:
                continue
            base_td[w]["points"] += 2; base_td[w]["matches"] += 1
            base_td[w]["runs_for"] += wr; base_td[w]["overs_faced"] += wo
            base_td[w]["runs_against"] += lr; base_td[w]["overs_bowled"] += lo
            base_td[l]["matches"] += 1
            base_td[l]["runs_for"] += lr; base_td[l]["overs_faced"] += lo
            base_td[l]["runs_against"] += wr; base_td[l]["overs_bowled"] += wo

    base_pts = {t: base_td[t]["points"] for t in teams}
    base_nrr = {t: calculate_nrr(base_td[t]) for t in teams}

    top4_c = {t: 0 for t in teams}; top2_c = {t: 0 for t in teams}
    top4_pts = {t: 0 for t in teams}; top2_pts = {t: 0 for t in teams}

    for _ in range(num_simulations):
        pts  = base_pts.copy()
        nrrs = base_nrr.copy()
        for match in matches:
            if match.get("applied"):
                continue
            h, a   = match["home"], match["away"]
            boost  = get_home_boost(match.get("venue", ""))
            sh     = hw[h] * boost
            sa     = hw[a]
            w, l   = (h, a) if np.random.rand() < sh / (sh + sa) else (a, h)
            pts[w] += 2
            mg = simulate_nrr_change(hw[w], hw[l])
            nrrs[w] += mg; nrrs[l] -= mg

        spo = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        f5  = spo[4][1]; f3 = spo[2][1]
        for t, p in pts.items():
            if p > f5: top4_pts[t] += 1
            if p > f3: top2_pts[t] += 1

        st = sorted(teams, key=lambda t: (pts[t], nrrs[t]), reverse=True)
        for t in st[:4]: top4_c[t] += 1
        for t in st[:2]: top2_c[t] += 1

    n = num_simulations
    q4  = {t: round(top4_c[t]   / n * 100, 2) for t in teams}
    q2  = {t: round(top2_c[t]   / n * 100, 2) for t in teams}
    c4  = {t: round(top4_pts[t] / n * 100, 2) for t in teams}
    c2  = {t: round(top2_pts[t] / n * 100, 2) for t in teams}

    sorted_t = sorted(teams, key=lambda t: q4[t], reverse=True)
    return pd.DataFrame(
        [(t, q4[t], q2[t], c4[t], c2[t], None) for t in sorted_t],
        columns=["Team","Top 4 (%)","Top 2 (%)","Top 4 Confirmed (%)","Top 2 Confirmed (%)","Top 4 Pure Math (%)"]
    )

# ---------------------------------------------------------------------------
# Points table helpers
# ---------------------------------------------------------------------------
def get_current_points_table():
    table = []
    for team in teams:
        d  = updated_points_data[team]
        of = overs_to_float(d["overs_faced"])
        ob = overs_to_float(d["overs_bowled"])
        nrr = round(d["runs_for"] / of - d["runs_against"] / ob, 3) if of > 0 and ob > 0 else 0.0
        table.append({"Team": team, "Points": d["points"], "Matches": d["matches"], "NRR": nrr})
    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))

def get_points_table_after_what_if(what_if_matches):
    td = {
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
            h, a, r = match["home"], match["away"], match["result"]
            if r == "Abandoned/No Result (1 point each)":
                for t in [h, a]:
                    td[t]["points"] += 1; td[t]["matches"] += 1
                continue
            w = r; l = a if w == h else h
            wr = match["runs"].get(w); lr = match["runs"].get(l)
            wo = overs_to_float(match["overs"].get(w))
            lo = overs_to_float(match["overs"].get(l))
            if None in (wr, lr) or wo == 0 or lo == 0:
                continue
            td[w]["points"] += 2; td[w]["matches"] += 1
            td[w]["runs_for"] += wr; td[w]["overs_faced"] += wo
            td[w]["runs_against"] += lr; td[w]["overs_bowled"] += lo
            td[l]["matches"] += 1
            td[l]["runs_for"] += lr; td[l]["overs_faced"] += lo
            td[l]["runs_against"] += wr; td[l]["overs_bowled"] += wo

    table = []
    for team, d in td.items():
        of = d["overs_faced"]; ob = d["overs_bowled"]
        nrr = round(d["runs_for"] / of - d["runs_against"] / ob, 3) if of > 0 and ob > 0 else 0.0
        table.append({"Team": team, "Points": d["points"], "Matches": d["matches"], "NRR": nrr})
    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))

# ---------------------------------------------------------------------------
# Pure Math simulation (parallel)
# ---------------------------------------------------------------------------
def run_pure_math_worker(args):
    seed, sims, matches = args
    np.random.seed(seed)
    top4 = {t: 0 for t in teams}
    bp   = {t: updated_points_data[t]["points"] for t in teams}
    for m in matches:
        if m.get("applied") and m.get("result"):
            r = m["result"]
            if r == "Abandoned/No Result (1 point each)":
                bp[m["home"]] += 1; bp[m["away"]] += 1
            elif r in bp:
                bp[r] += 2
    for _ in range(sims):
        pts = bp.copy()
        for m in matches:
            if m.get("applied") and m.get("result"):
                continue
            pts[np.random.choice([m["home"], m["away"]])] += 2
        st  = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        fp  = st[3][1]
        above = [t for t, p in st if p > fp]
        tied  = [t for t, p in st if p == fp]
        spots = 4 - len(above)
        for t in above: top4[t] += 1
        if spots > 0 and tied:
            for t in tied: top4[t] += spots / len(tied)
    return top4

def run_pure_math_simulation_parallel(total_sims=10000, processes=4, override_matches=None):
    spc     = total_sims // processes
    seeds   = np.random.randint(0, 1_000_000_000, size=processes)
    matches = override_matches if override_matches is not None else remaining_matches
    with ThreadPoolExecutor(max_workers=processes) as ex:
        results = list(ex.map(run_pure_math_worker,
                              [(s, spc, copy.deepcopy(matches)) for s in seeds]))
    combined = {t: sum(r[t] for r in results) for t in teams}
    return {t: round(combined[t] / total_sims * 100, 2) for t in teams}

# ---------------------------------------------------------------------------
# Parallel main simulation
# ---------------------------------------------------------------------------
def parallel_worker(args):
    seed, sims, what_if, matches = args
    np.random.seed(seed)
    return run_adjusted_simulation(sims, what_if=what_if, override_matches=matches)

def run_parallel_simulations(total_sims=10000, processes=4, override_matches=None):
    spc  = total_sims // processes
    seeds = np.random.randint(0, 1_000_000_000, size=processes)
    mu   = override_matches if override_matches is not None else remaining_matches
    with ThreadPoolExecutor(max_workers=processes) as ex:
        results = list(ex.map(parallel_worker,
                              [(s, spc, True, copy.deepcopy(mu)) for s in seeds]))
    final = pd.concat(results)
    g = final.groupby("Team").agg({
        "Top 4 (%)": "mean", "Top 2 (%)": "mean",
        "Top 4 Confirmed (%)": "mean", "Top 2 Confirmed (%)": "mean",
    }).reset_index()
    return g.sort_values(by=["Top 4 (%)", "Top 2 (%)"],
                         ascending=[False, False]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def fancy_highlight_half_split(df):
    pct_cols = ["Top 2 (%)", "Top 4 (%)", "Top 2 Confirmed (%)",
                "Top 4 Confirmed (%)", "Top 4 Pure Math (%)"]

    def color_by_pct(val):
        if pd.isna(val): return ""
        try: val = float(val)
        except: return ""
        if val == 100.00: return "background-color: #301934; color: white; font-weight: bold"
        if val == 0.00:   return "background-color: #36454F; color: white"
        if 0.01 <= val <= 0.99: return "background-color: #580000; color: white"
        if 1.00 <= val <= 44.99:
            step = int(val // 5)
            cs = ["#880000","#A02020","#B03030","#C04040","#D05030",
                  "#E06030","#EF7020","#F88020","#FF9020"]
            return f"background-color: {cs[min(step, len(cs)-1)]}; color: white"
        if 45.00 <= val <= 50.00: return "background-color: #FFFF66; color: black"
        if 50.01 <= val <= 99.99:
            gi = int(245 - ((val - 50.01) / 49.98) * 150)
            rb = int(168 - ((val - 50.01) / 49.98) * 168)
            return f"background-color: #{'%02X%02X%02X' % (rb, gi, rb)}; color: black"
        return ""

    styled = df.style.format({c: "{:.2f}" for c in pct_cols})
    for col in pct_cols:
        styled = styled.map(color_by_pct, subset=col)
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
    df        = run_parallel_simulations(10000, processes=16)
    pure_math = run_pure_math_simulation_parallel(10000, processes=16)
    df["Top 4 Pure Math (%)"] = df["Team"].map(pure_math)
    styled_df = fancy_highlight_half_split(df)
    print(df)

    suggested  = f"m{MATCHES_COMMITTED}"
    user_input = input(f"Enter match ID (Enter for '{suggested}', or 'skip'): ").strip().lower()
    match_id   = suggested if user_input == "" else (None if user_input == "skip" else user_input)

    if match_id:
        fmt = input("Save format? 'csv', 'excel', or 'both': ").strip().lower()
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs("results", exist_ok=True)
        if fmt in ["csv",   "both"]: df.to_csv(f"results/post_{match_id}_results_{ts}.csv", index=False)
        if fmt in ["excel", "both"]: styled_df.to_excel(f"results/post_{match_id}_stylized_{ts}.xlsx", index=False)
        print("✅ Saved.")
    else:
        print("⚠️ Skipped.")

if __name__ == "__main__":
    run_full_simulation_and_prompt()