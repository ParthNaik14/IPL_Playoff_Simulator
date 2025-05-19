import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

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

updated_points_data = {
    "Royal Challengers Bengaluru": {
        "points": 17, "matches": 12,
        "runs_for": 1938, "overs_faced": "205.1",
        "runs_against": 1863, "overs_bowled": "207.5"
    },
    "Punjab Kings": {
        "points": 17, "matches": 12,
        "runs_for": 2054, "overs_faced": "208.1",
        "runs_against": 2003, "overs_bowled": "211.2"
    },
    "Mumbai Indians": {
        "points": 14, "matches": 12,
        "runs_for": 2108, "overs_faced": "221.2",
        "runs_against": 1993, "overs_bowled": "238.1"
    },
    "Gujarat Titans": {
        "points": 18, "matches": 12,
        "runs_for": 2335, "overs_faced": "231.5",
        "runs_against": 2174, "overs_bowled": "234.2"
    },
    "Delhi Capitals": {
        "points": 13, "matches": 12,
        "runs_for": 2025, "overs_faced": "211.1",
        "runs_against": 2023, "overs_bowled": "216.5"
    },
    "Kolkata Knight Riders": {
        "points": 12, "matches": 13,
        "runs_for": 1827, "overs_faced": "207.4",
        "runs_against": 1797, "overs_bowled": "208.5"
    },
    "Lucknow Super Giants": {
        "points": 10, "matches": 12,
        "runs_for": 2270, "overs_faced": "235.4",
        "runs_against": 2347, "overs_bowled": "231.3"
    },
    "Sunrisers Hyderabad": {
        "points": 9, "matches": 12,
        "runs_for": 2010, "overs_faced": "215.3",
        "runs_against": 2094, "overs_bowled": "202.4"
    },
    "Rajasthan Royals": {
        "points": 6, "matches": 13,
        "runs_for": 2415, "overs_faced": "255.5",
        "runs_against": 2586, "overs_bowled": "255.0"
    },
    "Chennai Super Kings": {
        "points": 6, "matches": 12,
        "runs_for": 2024, "overs_faced": "238.2",
        "runs_against": 2126, "overs_bowled": "224.1"
    }
}





squad_strength = {
    "Mumbai Indians": 9.0, "Punjab Kings": 8.5, "Delhi Capitals": 9.0,
    "Royal Challengers Bengaluru": 8.5, "Gujarat Titans": 8.5, "Kolkata Knight Riders": 8.5,
    "Lucknow Super Giants": 8.0, "Rajasthan Royals": 7.5, "Sunrisers Hyderabad": 8.0,
    "Chennai Super Kings": 7.5
}

remaining_matches = [
    #{"home": "Sunrisers Hyderabad", "away": "Delhi Capitals", "venue": "Hyderabad", "result": None, "margin": None, "applied": False},
    #{"home": "Mumbai Indians", "away": "Gujarat Titans", "venue": "Mumbai", "result": None, "margin": None, "applied": False},
    #{"home": "Kolkata Knight Riders", "away": "Chennai Super Kings", "venue": "Kolkata", "result": None, "margin": None, "applied": False},
    #{"home": "Royal Challengers Bengaluru", "away": "Kolkata Knight Riders", "venue": "Bengaluru", "result": None, "margin": None, "applied": False},
    #{"home": "Rajasthan Royals", "away": "Punjab Kings", "venue": "Jaipur", "result": None, "margin": None, "applied": False},
    #{"home": "Delhi Capitals", "away": "Gujarat Titans", "venue": "Delhi", "result": None, "margin": None, "applied": False},
    #{"home": "Lucknow Super Giants", "away": "Sunrisers Hyderabad", "venue": "Lucknow", "result": None, "margin": None, "applied": False},
    {"home": "Chennai Super Kings", "away": "Rajasthan Royals", "venue": "Delhi", "result": None, "margin": None, "applied": False},
    {"home": "Mumbai Indians", "away": "Delhi Capitals", "venue": "Mumbai", "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans", "away": "Lucknow Super Giants", "venue": "Ahmedabad", "result": None, "margin": None, "applied": False},
    {"home": "Royal Challengers Bengaluru", "away": "Sunrisers Hyderabad", "venue": "Bengaluru", "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings", "away": "Delhi Capitals", "venue": "Jaipur", "result": None, "margin": None, "applied": False},
    {"home": "Gujarat Titans", "away": "Chennai Super Kings", "venue": "Ahmedabad", "result": None, "margin": None, "applied": False},
    {"home": "Sunrisers Hyderabad", "away": "Kolkata Knight Riders", "venue": "Delhi", "result": None, "margin": None, "applied": False},
    {"home": "Punjab Kings", "away": "Mumbai Indians", "venue": "Jaipur", "result": None, "margin": None, "applied": False},
    {"home": "Lucknow Super Giants", "away": "Royal Challengers Bengaluru", "venue": "Lucknow", "result": None, "margin": None, "applied": False},
]

def set_what_if_results(new_remaining_matches):
    global remaining_matches
    remaining_matches = new_remaining_matches

    # DEBUG: Show received What-if matches
    print("\n--- WHAT-IF MATCHES RECEIVED ---")
    for match in remaining_matches:
        if match.get("applied"):
            print(match)




def simulate_nrr_change(winner_strength, loser_strength):
    strength_diff = winner_strength - loser_strength
    base = np.clip(0.2 * strength_diff, -0.1, 0.25)
    noise = np.random.normal(0.05, 0.05)
    return round(np.clip(base + noise, -0.3, 0.3), 3)

def overs_to_float(overs_str):
    """
    Converts overs in 'xx.y' format to float.
    Example: '205.1' -> 205.1667
    """
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


def custom_even_round(x):
    whole = int(x)
    return int(np.floor(x)) if whole % 2 == 0 else int(np.ceil(x))

def adjust_final_points(points_dict, qualification_order):
    total = sum(points_dict.values())
    sorted_teams = qualification_order.copy()
    while total != 140:
        if total < 140:
            for team in sorted_teams:
                points_dict[team] += 2
                total += 2
                if total == 140:
                    break
        elif total > 140:
            for team in reversed(sorted_teams):
                if points_dict[team] > 2:
                    points_dict[team] -= 2
                    total -= 2
                    if total == 140:
                        break
    return points_dict


def run_adjusted_simulation(num_simulations, what_if=False, override_matches=None):
    matches = override_matches if override_matches is not None else remaining_matches

    max_rating = max(squad_strength.values())
    squad_weights = {team: val / max_rating for team, val in squad_strength.items()}

    strength_scores = {}
    for team, data in updated_points_data.items():
        win_pct = data["points"] / (data["matches"] * 2)
        nrr_score = (calculate_nrr(data) + 2) / 4
        strength_scores[team] = 0.7 * win_pct + 0.3 * nrr_score

    total_form_strength = sum(strength_scores.values())
    form_weights = {team: val / total_form_strength for team, val in strength_scores.items()}

    hybrid_strength = {
        team: 0.9 * form_weights[team] + 0.1 * squad_weights[team]
        for team in teams
    }
    total_hybrid = sum(hybrid_strength.values())
    hybrid_weights = {team: val / total_hybrid for team, val in hybrid_strength.items()}



    # Apply What-if match results
    # Create a copy of updated_points_data
    base_team_data = {
        team: {
            "points": data["points"],
            "matches": data["matches"],
            "runs_for": data["runs_for"],
            "overs_faced": overs_to_float(data["overs_faced"]),
            "runs_against": data["runs_against"],
            "overs_bowled": overs_to_float(data["overs_bowled"]),
        }
        for team, data in updated_points_data.items()
    }

    # Apply What-if match results
    for match in matches:
        if match.get("applied") and match.get("result") in [match["home"], match["away"],
                                                            "Abandoned/No Result (1 point each)"]:
            home = match["home"]
            away = match["away"]
            result = match["result"]

            if result == "Abandoned/No Result (1 point each)":
                base_team_data[home]["points"] += 1
                base_team_data[away]["points"] += 1
                base_team_data[home]["matches"] += 1
                base_team_data[away]["matches"] += 1
                continue

            winner = result
            loser = away if winner == home else home

            try:
                wr = match["runs"][winner]
                wo = overs_to_float(match["overs"][winner])
                lr = match["runs"][loser]
                lo = overs_to_float(match["overs"][loser])
            except Exception as e:
                print(f"Invalid What-if format for match: {match}")
                continue

            base_team_data[winner]["points"] += 2
            base_team_data[winner]["matches"] += 1
            base_team_data[winner]["runs_for"] += wr
            base_team_data[winner]["overs_faced"] = round(base_team_data[winner]["overs_faced"] + wo)
            base_team_data[winner]["runs_against"] += lr
            base_team_data[winner]["overs_bowled"] = round(base_team_data[winner]["overs_bowled"] + lo)

            base_team_data[loser]["matches"] += 1
            base_team_data[loser]["runs_for"] += lr
            base_team_data[loser]["overs_faced"] = round(base_team_data[loser]["overs_faced"] + lo)
            base_team_data[loser]["runs_against"] += wr
            base_team_data[loser]["overs_bowled"] = round(base_team_data[loser]["overs_bowled"] + wo, 3)

    # Use the modified data to compute base points and NRR
    base_points = {team: base_team_data[team]["points"] for team in teams}
    #for match in remaining_matches:
     #   if match.get("applied") and match.get("result") in [match["home"], match["away"]]:
      #      base_points[match["result"]] += 2  # add 2 points temporarily

    base_nrr = {team: calculate_nrr(base_team_data[team]) for team in teams}

    top4_counts = {team: 0 for team in teams}
    top2_counts = {team: 0 for team in teams}
    top4_confirmed_points_only = {team: 0 for team in teams}
    top2_confirmed_points_only = {team: 0 for team in teams}
    cumulative_points = {team: 0 for team in teams}
    cumulative_nrr = {team: 0.0 for team in teams}

    for _ in range(num_simulations):
        points = base_points.copy()
        nrrs = base_nrr.copy()

        for match in matches:
            if match.get("applied"):
                continue  # Skip already applied What-if results
            home = match["home"]
            away = match["away"]
            s_home = hybrid_weights[home] #* 1.05
            s_away = hybrid_weights[away] #* 0.95
            prob_home_win = s_home / (s_home + s_away)
            winner, loser = (home, away) if np.random.rand() < prob_home_win else (away, home)
            points[winner] += 2
            margin = simulate_nrr_change(hybrid_weights[winner], hybrid_weights[loser])
            nrrs[winner] += margin
            nrrs[loser] -= margin

        # Sort by points only for "confirmed %" columns
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
            cumulative_nrr[team] += nrrs[team]

    avg_points = {team: custom_even_round(cumulative_points[team] / num_simulations) for team in teams}
    avg_nrrs = {team: round(cumulative_nrr[team] / num_simulations, 3) for team in teams}
    qualifications = {team: round((top4_counts[team] / num_simulations) * 100, 2) for team in teams}
    top2_qualifications = {team: round((top2_counts[team] / num_simulations) * 100, 2) for team in teams}
    top4_confirmed_points_only_pct = {team: round((top4_confirmed_points_only[team] / num_simulations) * 100, 2) for team in teams}
    top2_confirmed_points_only_pct = {team: round((top2_confirmed_points_only[team] / num_simulations) * 100, 2) for team in teams}

    sorted_by_qual = sorted(teams, key=lambda t: qualifications[t], reverse=True)
    adjusted_points = adjust_final_points(avg_points.copy(), sorted_by_qual)

    results = pd.DataFrame([
        (
            team,
            qualifications[team],
            top2_qualifications[team],
            top4_confirmed_points_only_pct[team],
            top2_confirmed_points_only_pct[team],
            None,
            adjusted_points[team],
            avg_nrrs[team]
        )
        for team in sorted_by_qual
    ], columns=[
        "Team", "Top 4 (%)", "Top 2 (%)",
        "Top 4 Confirmed (%)", "Top 2 Confirmed (%)",
        "Top 4 Pure Math (%)", "Avg Final Points", "Avg Final NRR"
    ])

    return results

def get_current_points_table():
    table = []
    for team in teams:
        data = updated_points_data[team]
        runs_for = data["runs_for"]
        overs_faced = overs_to_float(data["overs_faced"])
        runs_against = data["runs_against"]
        overs_bowled = overs_to_float(data["overs_bowled"])

        if overs_faced == 0 or overs_bowled == 0:
            nrr = 0.0
        else:
            nrr = round((runs_for / overs_faced) - (runs_against / overs_bowled), 3)

        table.append({
            "Team": team,
            "Points": data["points"],
            "Matches": data["matches"],
            "NRR": nrr
        })

    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))


def get_points_table_after_what_if(what_if_matches):

    team_data = {
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

    for match in what_if_matches:
        if match.get("applied") and match.get("result") in [match["home"], match["away"],
                                                            "Abandoned/No Result (1 point each)"]:
            home = match["home"]
            away = match["away"]
            result = match["result"]

            if result == "Abandoned/No Result (1 point each)":
                team_data[home]["points"] += 1
                team_data[away]["points"] += 1
                team_data[home]["matches"] += 1
                team_data[away]["matches"] += 1
                continue

            winner = result
            loser = away if winner == home else home

            wr = match["runs"].get(winner)
            lr = match["runs"].get(loser)
            wo = overs_to_float(match["overs"].get(winner))
            lo = overs_to_float(match["overs"].get(loser))

            if wr is None or lr is None or wo is None or lo is None:
                continue

            team_data[winner]["points"] += 2
            team_data[winner]["matches"] += 1
            team_data[winner]["runs_for"] += wr
            team_data[winner]["overs_faced"] += wo
            team_data[winner]["runs_against"] += lr
            team_data[winner]["overs_bowled"] += lo

            team_data[loser]["matches"] += 1
            team_data[loser]["runs_for"] += lr
            team_data[loser]["overs_faced"] += lo
            team_data[loser]["runs_against"] += wr
            team_data[loser]["overs_bowled"] += wo

    # Compute NRR and build output table
    table = []
    for team, data in team_data.items():
        of = data["overs_faced"]
        ob = data["overs_bowled"]
        nrr = round((data["runs_for"] / of - data["runs_against"] / ob), 3) if of > 0 and ob > 0 else 0.0
        table.append({
            "Team": team,
            "Points": data["points"],
            "Matches": data["matches"],
            "NRR": nrr
        })

    return pd.DataFrame(sorted(table, key=lambda x: (x["Points"], x["NRR"]), reverse=True))




# --- Pure Math Simulation (Parallel) ---
def run_pure_math_worker(args):
    seed, sims, matches = args
    np.random.seed(seed)
    top4_counts = {team: 0 for team in teams}
    base_points = {team: updated_points_data[team]["points"] for team in teams}

    for match in matches:
        if match.get("applied") and match.get("result"):
            result = match["result"]
            if result == "Abandoned/No Result (1 point each)":
                home = match["home"]
                away = match["away"]
                base_points[home] += 1
                base_points[away] += 1
            else:
                base_points[result] += 2

    for _ in range(sims):
        points = base_points.copy()

        for match in matches:
            if match.get("applied") and match.get("result"):
                continue  # Skip What-if result
            home = match["home"]
            away = match["away"]
            winner = np.random.choice([home, away])
            points[winner] += 2

        sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
        fourth_place_points = sorted_teams[3][1]

        above = [t for t, p in sorted_teams if p > fourth_place_points]
        tied = [t for t, p in sorted_teams if p == fourth_place_points]
        spots = 4 - len(above)

        for t in above:
            top4_counts[t] += 1
        if spots > 0 and tied:
            for t in tied:
                top4_counts[t] += spots / len(tied)

    return top4_counts





def run_pure_math_simulation_parallel(total_sims=10000, processes=4, override_matches=None):

    sims_per_core = total_sims // processes
    seeds = np.random.randint(0, 1e9, size=processes)
    matches = override_matches if override_matches is not None else remaining_matches

    with ThreadPoolExecutor(max_workers=processes) as executor:
        results = list(executor.map(run_pure_math_worker, [(seed, sims_per_core, matches) for seed in seeds]))

    combined = {team: 0 for team in teams}

    all_applied = all(m.get("applied") and m.get("result") for m in matches)
    if all_applied:
        # Only one scenario: current points table = final table
        sorted_points = sorted(
            {team: updated_points_data[team]["points"] + sum(2 for m in matches if m["result"] == team) for team in
             teams}.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top4_teams = [team for team, _ in sorted_points[:4]]
        return {team: 100.0 if team in top4_teams else 0.0 for team in teams}

    for partial in results:
        for team in teams:
            combined[team] += partial[team]

    return {team: round((combined[team] / total_sims) * 100, 2) for team in teams}


# Parallel helpers
def parallel_worker(seed_and_sims_matches):
    seed, sims, what_if, matches = seed_and_sims_matches
    np.random.seed(seed)
    return run_adjusted_simulation(sims, what_if=what_if, override_matches=matches)





def run_parallel_simulations(total_sims=10000, processes=4, override_matches=None):
    sims_per_core = total_sims // processes
    seeds = np.random.randint(0, 1e9, size=processes)
    matches_to_use = override_matches if override_matches is not None else remaining_matches

    with ThreadPoolExecutor(max_workers=processes) as executor:
        results = list(
            executor.map(parallel_worker, [(seed, sims_per_core, True, matches_to_use.copy()) for seed in seeds]))

    final_df = pd.concat(results)
    grouped = final_df.groupby("Team").agg({
        "Top 4 (%)": "mean",
        "Top 2 (%)": "mean",
        "Top 4 Confirmed (%)": "mean",
        "Top 2 Confirmed (%)": "mean",
        "Avg Final Points": "mean",
        "Avg Final NRR": "mean"
    }).reset_index()
    grouped["Avg Final Points"] = (grouped["Avg Final Points"] / 2).round().astype(int) * 2
    grouped = grouped.sort_values(by=["Top 4 (%)", "Avg Final Points", "Avg Final NRR"], ascending=[False, False, False]).reset_index(drop=True)
    return grouped



def fancy_highlight_half_split(df):
    top_half = df.index <6
    bottom_half = df.index >=6

    df_styled = df.copy()
    positional_map = {5: 0.0, 6: 0.25, 7: 0.5, 8: 0.75, 9: 1.0}
    for col in ["Top 2 (%)", "Top 4 (%)", "Top 2 Confirmed (%)", "Top 4 Confirmed (%)", "Top 4 Pure Math (%)"]:
        for idx in positional_map:
            df_styled.at[idx, col] = positional_map[idx]

    styled = df.style.format({
        "Top 2 (%)": "{:.2f}", "Top 4 (%)": "{:.2f}",
        "Top 2 Confirmed (%)": "{:.2f}", "Top 4 Confirmed (%)": "{:.2f}",
        "Top 4 Pure Math (%)": "{:.2f}", "Avg Final NRR": "{:.3f}"
    })

    for col in ["Top 2 (%)", "Top 4 (%)", "Top 2 Confirmed (%)", "Top 4 Confirmed (%)", "Top 4 Pure Math (%)"]:
        styled = styled.background_gradient(cmap="Greens", subset=pd.IndexSlice[top_half, col]) \
                         .background_gradient(cmap="Reds", subset=pd.IndexSlice[bottom_half, col], gmap=df_styled[col])

    for col in ["Avg Final Points", "Avg Final NRR"]:
        styled = styled.background_gradient(cmap="Blues", subset=col)

    styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})
    styled = styled.set_properties(**{"text-align": "center", "vertical-align": "middle"})

    def color_team_cells(val):
        if val in team_colors:
            c = team_colors[val]
            return f"background-color: {c['bg']}; color: {c['text']}"#; font-weight: bold"
        return ""

    styled = styled.map(color_team_cells, subset=["Team"])


    return styled


# Run the simulation


def run_full_simulation_and_prompt():
    df = run_parallel_simulations(10000, processes=16)
    pure_math = run_pure_math_simulation_parallel(10000, processes=16)

    # Insert "Top 4 Pure Math (%)" just before "Avg Final Points"
    pure_math_series = df["Team"].map(pure_math)
    insert_position = df.columns.get_loc("Avg Final Points")
    df.insert(loc=insert_position, column="Top 4 Pure Math (%)", value=pure_math_series)

    styled_df = fancy_highlight_half_split(df)
    print(df)

    default_match_number = 70 - len(remaining_matches)
    suggested_match_id = f"m{default_match_number}"
    user_input = input(f"Enter match ID (press Enter to use suggested '{suggested_match_id}', or type 'skip' to skip saving): ").strip().lower()

    if user_input == "":
        match_id = suggested_match_id
    elif user_input == "skip":
        match_id = None
    else:
        match_id = user_input

    if match_id:
        format_input = input("Save format? Type 'csv', 'excel', or 'both': ").strip().lower()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        csv_path = f"{output_dir}/post_{match_id}_results_{timestamp}.csv"
        xlsx_path = f"{output_dir}/post_{match_id}_stylized_{timestamp}.xlsx"

        if format_input in ["csv", "both"]:
            df.to_csv(csv_path, index=False)
        if format_input in ["excel", "both"]:
            styled_df.to_excel(xlsx_path, index=False)

        print("\n✅ Saved:")
        if format_input in ["csv", "both"]:
            print(f"  [CSV]   {csv_path}")
        if format_input in ["excel", "both"]:
            print(f"  [Excel] {xlsx_path}")
    else:
        print("\n⚠️ Skipped file saving.")


if __name__ == "__main__":
    run_full_simulation_and_prompt()
