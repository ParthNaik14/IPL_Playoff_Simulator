import streamlit as st
import pandas as pd
from datetime import datetime
import io
import ipl_simulator as sim

def add_form_column(df):
    """Adds a styled HTML Form column to the points table."""
    def get_styled_result(res):
        if res == 1:
            return '<span style="color: #00FF00; font-weight: bold;">W</span>'
        elif res == 0:
            return '<span style="color: #FF4B4B; font-weight: bold;">L</span>'
        elif res == 9: # Draw 'NR' whenever the script sees our nickname '9'
            return '<span style="color: #808080; font-weight: bold;">NR</span>'
        return ""

    form_map = {
        team: " ".join([get_styled_result(res) for res in sim.recent_form.get(team, [])])
        for team in df["Team"]
    }
    df["Form (Oldest → Recent)"] = df["Team"].map(form_map)
    return df

# --- Page setup ---
st.set_page_config(page_title="IPL Playoff Simulator", layout="wide")
st.title("📊 IPL 2026 Playoff Qualification Simulator")

# ---------------------------------------------------------------------------
# Admin mode
# ---------------------------------------------------------------------------
try:
    admin_mode = st.secrets.get("admin_mode", False)
except Exception:
    admin_mode = False

query_params = st.query_params
if "reset" in query_params:
    reset_type = query_params["reset"]
    if reset_type == "everything":
        st.success("✅ Everything has been reset.")
    elif reset_type == "scenarios":
        st.success("✅ All What-if scenario inputs have been cleared.")
    elif reset_type == "committed":
        st.success("✅ Match committed successfully!")
    elif reset_type == "decommitted":
        st.success("↩️ Last match decommitted successfully.")
    st.query_params.clear()

# --- Sidebar: Simulation controls ---
total_simulations = st.sidebar.number_input(
    "Total Simulations", value=10000, step=2000, min_value=2000, max_value=14000605
)
processes = st.sidebar.slider("Parallel Processes", min_value=1, max_value=4, value=2, step=1)

# --- Persistent state ---
if "simulation_df" not in st.session_state:
    st.session_state.simulation_df = None
    st.session_state.styled_df = None
    st.session_state.match_number = None
    st.session_state.timestamp = None
    st.session_state.reset_id = 0  # Add this line

# ADD IT RIGHT HERE:
if "reset_id" not in st.session_state:
    st.session_state.reset_id = 0

# ---------------------------------------------------------------------------
# Column configuration
# (name, default_visible, session_key)
# Priority order is the ORDER in this list — used for dynamic sorting.
# ---------------------------------------------------------------------------
ALL_COL_CONFIGS = [
    ("Status", True,  "col_status"),
    ("Qualify %", True, "col_qualify"),
    ("Top 2 %", True, "col_top2"),
    ("Safe by Points %", True, "col_safe_pts"),
    ("Safe Top 2 %", True, "col_safe_t2"),
    ("Still Possible %", True, "col_still4"),
    ("Top 2 Still Possible %", True, "col_still2"),
    ("Math Safe by Pts %", True, "col_math_safe4"),
    ("Math Safe Top 2 %", True, "col_math_safe2"),
    ("Avg Final Points", False, "col_avg_pts"),
    ("Avg Final NRR", False, "col_avg_nrr"),
]

# Seed defaults into session state on first load
for _cn, _def, _key in ALL_COL_CONFIGS:
    if _key not in st.session_state:
        st.session_state[_key] = _def


def _clear_whatif_state():
    # Increment reset_id to force all widgets to redraw from scratch
    st.session_state.reset_id += 1

    st.session_state.what_if_applied = False
    st.session_state.god_mode_success = None
    st.session_state.pathfinder_success = None
    st.session_state.simulation_df = None
    st.session_state.styled_df = None


# --- Randomisation callbacks updated for versioned keys ---
def randomize_single_callback(idx, m_home, m_away, m_venue):
    sc = sim.generate_constrained_scorecard(m_home, m_away, m_venue)
    rid = st.session_state.reset_id
    st.session_state[f"whatif_result_{idx}_{rid}"] = sc["winner"]
    st.session_state[f"winner_runs_{idx}_{rid}"] = sc["winner_runs"]
    st.session_state[f"winner_overs_{idx}_{rid}"] = sc["winner_overs"]
    st.session_state[f"loser_runs_{idx}_{rid}"] = sc["loser_runs"]
    st.session_state[f"loser_overs_{idx}_{rid}"] = sc["loser_overs"]


def randomize_all_callback():
    rid = st.session_state.reset_id
    for idx, match in enumerate(sim.remaining_matches):
        sc = sim.generate_constrained_scorecard(match["home"], match["away"], match["venue"])
        st.session_state[f"whatif_result_{idx}_{rid}"] = sc["winner"]
        st.session_state[f"winner_runs_{idx}_{rid}"] = sc["winner_runs"]
        st.session_state[f"winner_overs_{idx}_{rid}"] = sc["winner_overs"]
        st.session_state[f"loser_runs_{idx}_{rid}"] = sc["loser_runs"]
        st.session_state[f"loser_overs_{idx}_{rid}"] = sc["loser_overs"]


# ============================================================
# ADMIN SECTION 1 — COMMIT REAL RESULT
# ============================================================
if admin_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏏 Commit Real Result")
    st.sidebar.caption("Permanently saves result to source. Push to git after committing.")

    if not sim.remaining_matches:
        st.sidebar.info("All matches committed.")
    else:
        commit_options = [
            f"Match {sim.MATCHES_COMMITTED + i + 1}: {m['home']} vs {m['away']}"
            for i, m in enumerate(sim.remaining_matches)
        ]
        selected_commit = st.sidebar.selectbox(
            "Select match to commit", options=commit_options, key="commit_match_select"
        )
        commit_idx = commit_options.index(selected_commit)
        commit_match = sim.remaining_matches[commit_idx]

        commit_abandoned = st.sidebar.checkbox("Abandoned / No Result (1 pt each)", key="commit_abandoned")

        if commit_abandoned:
            if st.sidebar.button("✅ Commit Abandoned"):
                try:
                    sim.commit_result(
                        home=commit_match["home"], away=commit_match["away"],
                        winner=None, winner_runs=0, winner_overs_str="0.0",
                        loser_runs=0, loser_overs_str="0.0", abandoned=True,
                    )
                    st.session_state.simulation_df = None
                    st.session_state.what_if_applied = False
                    _clear_whatif_state()
                    st.query_params["reset"] = "committed"
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        else:
            commit_winner = st.sidebar.selectbox(
                "Winner",
                options=[commit_match["home"], commit_match["away"]],
                key="commit_winner"
            )
            commit_loser = commit_match["away"] if commit_winner == commit_match["home"] else commit_match["home"]

            st.sidebar.markdown(f"**{commit_winner}** (winner)")
            cw_runs = st.sidebar.number_input("Runs", min_value=0, value=None, key="cw_runs")
            cw_overs = st.sidebar.text_input("Overs (e.g. 19.3)", key="cw_overs")
            st.sidebar.markdown(f"**{commit_loser}** (loser)")
            cl_runs = st.sidebar.number_input("Runs ", min_value=0, value=None, key="cl_runs")
            cl_overs = st.sidebar.text_input("Overs (e.g. 20.0)", key="cl_overs")

            all_filled = all([cw_runs is not None, cw_overs, cl_runs is not None, cl_overs])

            if st.sidebar.button("✅ Commit Result", disabled=not all_filled):
                try:
                    sim.commit_result(
                        home=commit_match["home"], away=commit_match["away"],
                        winner=commit_winner,
                        winner_runs=int(cw_runs), winner_overs_str=cw_overs,
                        loser_runs=int(cl_runs), loser_overs_str=cl_overs,
                        abandoned=False,
                    )
                    st.session_state.simulation_df = None
                    st.session_state.what_if_applied = False
                    _clear_whatif_state()
                    st.query_params["reset"] = "committed"
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

    # ============================================================
    # ADMIN SECTION 2 — DECOMMIT LAST RESULT
    # ============================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ↩️ Decommit Last Result")

    if not sim.committed_results:
        st.sidebar.caption("No committed results to undo.")
    else:
        last = sim.committed_results[-1]
        label = (
            f"Match {sim.MATCHES_COMMITTED}: "
            f"{last['home']} vs {last['away']} — "
            f"{'Abandoned' if last['abandoned'] else last['winner']}"
        )
        st.sidebar.caption(f"Will undo: **{label}**")

        if "confirm_decommit" not in st.session_state:
            st.session_state.confirm_decommit = False

        if not st.session_state.confirm_decommit:
            if st.sidebar.button("↩️ Undo Last Commit"):
                st.session_state.confirm_decommit = True
                st.rerun()
        else:
            st.sidebar.warning(f"Are you sure you want to undo **{label}**?")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Yes, undo"):
                    try:
                        sim.decommit_last()
                        st.session_state.simulation_df = None
                        st.session_state.what_if_applied = False
                        st.session_state.confirm_decommit = False
                        _clear_whatif_state()
                        st.query_params["reset"] = "decommitted"
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error: {e}")
                        st.session_state.confirm_decommit = False
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_decommit = False
                    st.rerun()

# ============================================================
# SECTION 3 — RUN SIMULATION
# ============================================================
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running simulations..."):
        df = sim.run_parallel_simulations(total_simulations, processes=processes)
        pure_math = sim.run_pure_math_simulation_parallel(total_simulations, processes=processes)


        # --- NEW: Get Tragic Status ---
        # KEEP THESE:
        pending_matches = len(sim.remaining_matches)
        status_map = sim.calculate_tragic_status(pure_math, pending_matches)
        df["Status"] = df["Team"].map(status_map)  # Make sure this line exists right after!
        # ------------------------------

        df["Still Possible %"] = df["Team"].map(pure_math["top4"])
        df["Top 2 Still Possible %"] = df["Team"].map(pure_math["top2"])
        df["Math Safe by Pts %"] = df["Team"].map(pure_math["safe4"])
        df["Math Safe Top 2 %"] = df["Team"].map(pure_math["safe2"])

        # Update this list to include "Status"
        df = df[[
            "Team", "Status", "Qualify %", "Top 2 %",
            "Safe by Points %", "Safe Top 2 %",
            "Still Possible %", "Top 2 Still Possible %",
            "Math Safe by Pts %", "Math Safe Top 2 %",
            "Avg Final Points", "Avg Final NRR",
        ]]
        df.index = range(1, len(df) + 1)

        st.session_state.simulation_df = df
        st.session_state.styled_df = sim.fancy_highlight_half_split(df.copy())
        st.session_state.match_number = sim.MATCHES_COMMITTED
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.session_state["what_if_applied"] = False

# ============================================================
# ADMIN SECTION 3 — RESET ELO FOR NEW SEASON
# ============================================================
if admin_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔁 Reset Elo (New Season)")
    st.sidebar.caption(
        "Pulls all Elo ratings 30% toward 1500 to account for auction "
        "changes and inter-season uncertainty."
    )
    if "confirm_elo_reset" not in st.session_state:
        st.session_state.confirm_elo_reset = False

    if not st.session_state.confirm_elo_reset:
        if st.sidebar.button("🔁 Reset Elo for New Season"):
            st.session_state.confirm_elo_reset = True
            st.rerun()
    else:
        st.sidebar.warning("This will compress all Elo ratings toward 1500 and save to file. Are you sure?")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Yes, reset"):
                try:
                    sim.reset_elo_for_new_season()
                    st.session_state.confirm_elo_reset = False
                    st.session_state.simulation_df = None
                    st.sidebar.success("✅ Elo reset and saved.")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
                    st.session_state.confirm_elo_reset = False
        with col2:
            if st.button("❌ Cancel "):
                st.session_state.confirm_elo_reset = False
                st.rerun()

# ============================================================
# SECTION 4 — RESET BUTTONS
# ============================================================
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Everything"):
    reset_matches = [
        {"home": m["home"], "away": m["away"], "venue": m["venue"],
         "result": None, "margin": None, "applied": False}
        for m in sim.remaining_matches
    ]
    sim.set_what_if_results(reset_matches)
    _clear_whatif_state()
    st.session_state.simulation_df = None
    st.session_state.styled_df = None
    st.session_state.match_number = sim.MATCHES_COMMITTED
    st.session_state.timestamp = None
    st.session_state.what_if_applied = False
    st.session_state.combo_result = None
    st.query_params["reset"] = "everything"
    st.rerun()

if st.sidebar.button("♻️ Reset What-if Scenarios"):
    _clear_whatif_state()
    st.query_params["reset"] = "scenarios"
    st.rerun()

# ============================================================
# SECTION 5 — WHAT-IF SCENARIO EDITOR & GOD MODE
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 What-if Scenario Editor")

# NEW: God Mode directly integrated into the What-If sidebar!
with st.sidebar.expander("🎬 God Mode (Script Writer)", expanded=False):
    st.markdown("Pick 4 teams to force into the Top 4.")
    rid = st.session_state.reset_id

    if st.session_state.get("god_mode_success"):
        st.success(f"✅ Timeline found in {st.session_state.pop('god_mode_success'):,} attempts!")

    forced_teams = st.multiselect("Force Top 4 Teams:", sim.teams, key=f"god_teams_{rid}")
    max_att = st.select_slider("Max Attempts", options=[10_000, 50_000, 100_000, 250_000, 500_000], value=100_000, key=f"max_att_{rid}")

    if st.button("✨ Auto-Fill Timeline", disabled=len(forced_teams) != 4, key=f"god_btn_{rid}", use_container_width=True):
        with st.spinner("Writing script..."):
            results, _, attempts = sim.generate_forced_scenario(forced_teams, max_att)
            if results:
                for idx, res in enumerate(results):
                    st.session_state[f"whatif_result_{idx}_{rid}"] = res["Winner"]
                    st.session_state[f"winner_runs_{idx}_{rid}"] = res["winner_runs"]
                    st.session_state[f"winner_overs_{idx}_{rid}"] = res["winner_overs"]
                    st.session_state[f"loser_runs_{idx}_{rid}"] = res["loser_runs"]
                    st.session_state[f"loser_overs_{idx}_{rid}"] = res["loser_overs"]
                st.session_state.god_mode_success = attempts
                st.rerun()

            if results is not None:
                # Results list perfectly matches the remaining_matches list index
                for idx, res in enumerate(results):
                    st.session_state[f"whatif_result_{idx}"] = res["Winner"]

                    # Fill the runs and overs instead of clearing them!
                    st.session_state[f"winner_runs_{idx}"] = res["winner_runs"]
                    st.session_state[f"winner_overs_{idx}"] = res["winner_overs"]
                    st.session_state[f"loser_runs_{idx}"] = res["loser_runs"]
                    st.session_state[f"loser_overs_{idx}"] = res["loser_overs"]

                st.session_state.god_mode_success = attempts
                st.rerun()
            else:
                st.error(f"❌ No valid timeline found where these 4 qualify cleanly after {max_att:,} attempts.")

# NEW: Single Team Pathfinder UI
# Updated Pathfinder UI with messaging
with st.sidebar.expander("🎯 Single Team Pathfinder", expanded=False):
    st.markdown("Force one specific team to hit a specific rank.")
    rid = st.session_state.reset_id

    if st.session_state.get("pathfinder_success"):
        st.success(f"✅ Route found in {st.session_state.pop('pathfinder_success'):,} attempts!")

    t_team = st.selectbox("Target Team:", sim.teams, key=f"path_team_{rid}")
    t_rank = st.selectbox("Target Finish:", ["Top 4", 1, 2, 3, 4], key=f"path_rank_{rid}")

    if st.button("🗺️ Find Route", key=f"path_btn_{rid}", use_container_width=True):
        rank_val = "top4" if t_rank == "Top 4" else t_rank
        with st.spinner("Mapping route..."):
            results, _, attempts = sim.generate_single_team_route(t_team, rank_val)
            if results:
                for idx, res in enumerate(results):
                    st.session_state[f"whatif_result_{idx}_{rid}"] = res["Winner"]
                    st.session_state[f"winner_runs_{idx}_{rid}"] = res["winner_runs"]
                    st.session_state[f"winner_overs_{idx}_{rid}"] = res["winner_overs"]
                    st.session_state[f"loser_runs_{idx}_{rid}"] = res["loser_runs"]
                    st.session_state[f"loser_overs_{idx}_{rid}"] = res["loser_overs"]
                st.session_state.pathfinder_success = attempts
                st.rerun()
            else:
                st.error("❌ No realistic route found for this finish.")

what_if_matches = []

for i, match in enumerate(sim.remaining_matches):
    match_number = sim.MATCHES_COMMITTED + i + 1
    home_p, away_p = sim.get_win_probability(match["home"], match["away"], match["venue"])

    # ADDED _{rid} to the expander key
    with st.sidebar.expander(
            f"📌 Match {match_number}: {match['home']} vs {match['away']}", expanded=False
    ):
        st.caption(
            f"🏠 **{match['home']}** {home_p}%  ·  "
            f"✈️ **{match['away']}** {away_p}%  ·  📍 {match['venue']}"
        )

        # UPDATED: Button key now has _{rid}
        st.button(
            "🎲 Random Result",
            key=f"rand_{i}_{rid}",
            on_click=randomize_single_callback,
            args=(i, match["home"], match["away"], match["venue"])
        )

        # UPDATED: Selectbox key now has _{rid}
        winner = st.selectbox(
            "Select Result",
            options=["", match["home"], match["away"], "Abandoned/No Result (1 point each)"],
            key=f"whatif_result_{i}_{rid}"
        )

        match_dict = {
            "home": match["home"], "away": match["away"], "venue": match["venue"],
            "result": None, "applied": False, "runs": {}, "overs": {}
        }

        if winner:
            if winner == "Abandoned/No Result (1 point each)":
                match_dict["result"] = "Abandoned/No Result (1 point each)"
                match_dict["applied"] = True
            else:
                loser = match["away"] if winner == match["home"] else match["home"]

                # UPDATED: All input keys now have _{rid}
                winner_runs = st.number_input(f"{winner} Runs", min_value=0, value=None, key=f"winner_runs_{i}_{rid}")
                winner_overs = st.text_input(f"{winner} Overs (e.g., 19.3)", key=f"winner_overs_{i}_{rid}")
                loser_runs = st.number_input(f"{loser} Runs", min_value=0, value=None, key=f"loser_runs_{i}_{rid}")
                loser_overs = st.text_input(f"{loser} Overs (e.g., 20.0)", key=f"loser_overs_{i}_{rid}")

                match_dict["result"] = winner
                match_dict["applied"] = True
                match_dict["runs"] = {winner: winner_runs, loser: loser_runs}
                match_dict["overs"] = {winner: winner_overs, loser: loser_overs}

        what_if_matches.append(match_dict)

# Randomise All + Apply buttons
st.sidebar.markdown("---")
st.sidebar.button("🎲 Randomise All Remaining Matches", on_click=randomize_all_callback)

if st.sidebar.button("✅ Apply What-if Scenarios", type="primary", use_container_width=True):
    sim.set_what_if_results(what_if_matches)
    st.session_state.what_if_applied = True
    with st.spinner("Applying What-if and rerunning simulations..."):
        # 1. Run the simulations
        df = sim.run_parallel_simulations(total_simulations, processes=processes, override_matches=what_if_matches)
        pure_math = sim.run_pure_math_simulation_parallel(total_simulations, processes=processes,
                                                          override_matches=what_if_matches)

        pending_matches = len([m for m in what_if_matches if not m.get("applied")])
        status_map = sim.calculate_tragic_status(pure_math, pending_matches)
        df["Status"] = df["Team"].map(status_map)

        # 3. Map the rest of the columns
        df["Still Possible %"] = df["Team"].map(pure_math["top4"])
        df["Top 2 Still Possible %"] = df["Team"].map(pure_math["top2"])
        df["Math Safe by Pts %"] = df["Team"].map(pure_math["safe4"])
        df["Math Safe Top 2 %"] = df["Team"].map(pure_math["safe2"])

        # 4. CRITICAL: Make sure "Status" is included in this list
        df = df[[
            "Team", "Status", "Qualify %", "Top 2 %",
            "Safe by Points %", "Safe Top 2 %",
            "Still Possible %", "Top 2 Still Possible %",
            "Math Safe by Pts %", "Math Safe Top 2 %",
            "Avg Final Points", "Avg Final NRR",
        ]]
        df.index = range(1, len(df) + 1)
        st.session_state.simulation_df = df
        st.session_state.match_number = sim.MATCHES_COMMITTED
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.session_state["what_if_applied"] = True
    st.success("✅ What-if simulation applied successfully!")


# ============================================================
# HELPER — Points table styler
# ============================================================
def style_points_table(df):
    styled = df.style.format({"NRR": "{:.3f}"})
    styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})
    styled = styled.set_properties(**{"text-align": "center"})

    # NEW: Updated Status color coding logic
    def status_color(val):
        val_str = str(val)
        if "QUALIFIED" in val_str or "SAFE" in val_str:
            return "color: #00FF00; font-weight: bold;"  # Bright Green
        if "ELIMINATED" in val_str or "OUT" in val_str:
            return "color: #FF4B4B; font-weight: bold;"  # Red
        if "E1" in val_str:
            return "color: #FF4500; font-weight: bold;"  # Deep Orange/Red-Orange
        if "E2" in val_str:
            return "color: #FFA500; font-weight: bold;"  # Warning Orange
        if "In Hunt" in val_str:
            return "color: #1E90FF; font-weight: bold;"  # Dodger Blue
        return ""

    if "Status" in df.columns:
        styled = styled.map(status_color, subset=["Status"])

    def team_color(val):
        if val in sim.team_colors:
            c = sim.team_colors[val]
            return f"background-color: {c['bg']}; color: {c['text']}"
        return ""

    styled = styled.map(team_color, subset=["Team"])
    return styled


def style_elo_table(df):
    styled = df.style.format({"Elo Rating": "{:.1f}"})
    styled = styled.set_properties(**{"text-align": "center"})

    def team_color(val):
        if val in sim.team_colors:
            c = sim.team_colors[val]
            return f"background-color: {c['bg']}; color: {c['text']}"
        return ""

    styled = styled.map(team_color, subset=["Team"])

    def elo_bar(val):
        try:
            val = float(val)
            intensity = (val - 1480) / 40
            intensity = min(max(intensity, 0), 1)
            r = int(20 + (1 - intensity) * 60)
            g = int(60 + (1 - intensity) * 80)
            b = int(120 + (1 - intensity) * 80)
            return f"background-color: rgb({r},{g},{b}); color: white"
        except:
            return ""

    styled = styled.map(elo_bar, subset=["Elo Rating"])
    return styled


# ============================================================
# Explainers
# ============================================================
STATUS_EXPLAINER = """
### 🏷️ Qualification Status Explained
The **Status** badge gives you an instant read on a team's playoff survival. It is driven strictly by mathematical possibilities (ignoring team strength or NRR).

* **✅ QUALIFIED:** Mathematically guaranteed to make the Top 4. No combination of remaining results can knock them out.
* **🏏 In Hunt:** Actively in the playoff race. They survive in more than 15% of all possible future mathematical scenarios.
* **📉 E2 (Elimination Tier 2):** Struggling. They survive in 15% or less of scenarios. They likely need to win almost all remaining games and get lucky with other results.
* **⚠️ E1 (Must Win):** On the absolute brink. They survive in 7.5% or less of scenarios. One more loss, or even a rival team winning a separate match, will likely end their season.
* **❌ ELIMINATED:** Mathematically eliminated. Zero chance of making the playoffs, even if they win their remaining games.
"""
COLUMN_EXPLAINER = """
### 📖 How to read the simulation results

#### 🏷️ Status Badges Explained
The **Status** column gives you an instant read on a team's playoff survival. It is driven strictly by the **Still Possible %** column (pure mathematical scenarios, ignoring team strength or NRR).

* **✅ QUALIFIED / ❌ ELIMINATED:** The regular season is over (Match 70), and the final Top 4 is locked.
* **⭐ SAFE:** Mathematically guaranteed to make the Top 4. No combination of remaining results can knock them out (Still Possible = 100%).
* **🏏 In Hunt:** Actively in the playoff race. They survive in more than 35% of all possible future mathematical scenarios.
* **📉 E2 (Elimination Tier 2):** Struggling. They survive in less than 15% of scenarios. They likely need to win all remaining games and get lucky with other results.
* **⚠️ E1 (Must Win):** On the absolute brink. They survive in less than 7.5% of scenarios. One more loss, or even a rival team winning a separate match, will likely end their season.
* **❌ OUT:** Mathematically eliminated. Zero chance of making the playoffs, even if they win their remaining games (Still Possible = 0%).

---

The simulator runs **10,000 Monte Carlo simulations** of the remaining matches. Each simulation randomly determines winners based on team strength, home advantage, and current form, then ranks all 10 teams. Here's what each column means:

| Column | What it measures | Parameters used |
|--------|-----------------|-----------------|
| **Qualify %** | % of simulations where this team finished in the top 4. The primary qualification metric. Uses NRR to break points ties. | Elo rating · Recent form (last 5 games) · Win % · NRR · Home advantage |
| **Top 2 %** | % of simulations where this team finished top 2, earning a direct final berth. Uses NRR to break points ties. | Same as Qualify % |
| **Safe by Points %** | % of simulations where this team accumulated enough wins to finish top 4 **strictly on points alone** — even if NRR went against them. Uses the same strength-weighted win probabilities as Qualify %, but only checks the points standings at the end (NRR ignored for final rank). Will always be ≤ Qualify %. | Same win probability model as Qualify % · Final ranking by points only |
| **Safe Top 2 %** | Same as Safe by Points %, but for a top 2 finish on points alone. | Same win probability model as Top 2 % · Final ranking by points only |
| **Still Possible %** | Mathematical ceiling for top 4. If every remaining game is a **pure 50/50 coin flip** regardless of team strength, how often does this team finish top 4? When this hits 0%, top 4 is mathematically impossible. | Equal 50/50 win probability · Points only (no NRR) |
| **Top 2 Still Possible %** | Same as Still Possible %, but for a top 2 finish. When this hits 0%, a direct final berth is mathematically impossible. | Equal 50/50 win probability · Points only (no NRR) |
| **Math Safe by Pts %** | % of 50/50 simulations where the team guarantees top 4 strictly on points (dominating the 5th place team). | Pure 50/50 · Points-only strict dominance |
| **Math Safe Top 2 %** | Same as Math Safe by Pts %, but for top 2 strict dominance (dominating the 3rd place team). | Pure 50/50 · Points-only strict dominance |
| **Avg Final Points** | Average points this team finishes on across all simulations, rounded to nearest even number. **Most meaningful mid-season** once several matches have been played. | Same as Qualify % |
| **Avg Final NRR** | Average NRR this team finishes on across all simulations (3 decimal places). Most useful when multiple teams are projected on the same points — NRR decides who qualifies. Green = positive, red = negative. | Same as Qualify % |

> **Win probability model:** Each match probability = `team_strength / (team_strength + opponent_strength)`, where strength is a weighted blend of Elo (45–55%), recent form (0–30%), win % (15%), and NRR (remaining %). Home advantage adds a venue-specific multiplier (1.03–1.08×).
"""

# ============================================================
# MAIN CONTENT
# ============================================================
if st.session_state.simulation_df is not None:

    with st.expander("ℹ️ How to read the Qualification Status", expanded=False):
        st.markdown(STATUS_EXPLAINER)

    # --- Points table ---
    if st.session_state.get("what_if_applied"):
        st.subheader("📋 What-if Points Table (After Applied Matches)")
        whatif_table = sim.get_points_table_after_what_if(what_if_matches)
        if "Status" in st.session_state.simulation_df.columns:
            status_dict = dict(zip(st.session_state.simulation_df["Team"], st.session_state.simulation_df["Status"]))
            whatif_table.insert(1, "Status", whatif_table["Team"].map(status_dict))
        whatif_table.index = range(1, len(whatif_table) + 1)
        st.markdown(style_points_table(whatif_table).to_html(escape=False), unsafe_allow_html=True)
    else:
        st.markdown("### 📌 Current IPL Points Table")
        current_table = pd.DataFrame(sim.get_current_points_table())
        current_table = add_form_column(current_table)
        if "Status" in st.session_state.simulation_df.columns:
            status_dict = dict(zip(st.session_state.simulation_df["Team"], st.session_state.simulation_df["Status"]))
            current_table.insert(1, "Status", current_table["Team"].map(status_dict))
        current_table.index = range(1, len(current_table) + 1)
        st.markdown(style_points_table(current_table).to_html(escape=False), unsafe_allow_html=True)

    st.markdown("---")

    # --- Simulation results heading ---
    applied_count = sum(1 for m in what_if_matches if m.get("applied"))
    if st.session_state.get("what_if_applied"):
        simulated_match_number = sim.MATCHES_COMMITTED + applied_count
        st.subheader(f"📝 What-if Simulation Results — Post Match {simulated_match_number:02d}")
    else:
        st.subheader(f"📝 Simulation Results — Post Match {sim.MATCHES_COMMITTED:02d}")

    # ============================================================
    # THE FIX: DYNAMIC COLUMN TOGGLES & SORTING
    # ============================================================
    with st.expander("⚙️ Column Settings & Dynamic Sorting", expanded=False):
        st.markdown(
            "Select columns to display. **Drag and drop the tags to change the sorting priority.** (Avg Final Points and NRR are always the final tiebreakers).")

        all_col_names = [cn for cn, _d, _k in ALL_COL_CONFIGS]
        default_cols = [cn for cn, default, _k in ALL_COL_CONFIGS if default]

        selected_cols = st.multiselect(
            "Visible Columns & Sort Order:",
            options=all_col_names,
            default=default_cols,
            key="dynamic_col_multiselect"
        )

    # --- Column explainer ---
    with st.expander("ℹ️ How to read this table", expanded=False):
        st.markdown(COLUMN_EXPLAINER)

    # --- Build sorted display dataframe ---
    full_df = st.session_state.simulation_df.copy()

    if selected_cols:
        # 1. Build sorting priority, but EXCLUDE 'Status' from being used as a sort key
        sort_priority = [c for c in selected_cols if c != "Status"]

        # If the user somehow only selected "Status", give it a fallback sort
        if not sort_priority:
            sort_priority = ["Qualify %"]

        # 2. Always append the ultimate tie-breakers invisibly
        if "Avg Final Points" not in sort_priority: sort_priority.append("Avg Final Points")
        if "Avg Final NRR" not in sort_priority: sort_priority.append("Avg Final NRR")

        # 3. Sort dynamically
        full_df_sorted = full_df.sort_values(by=sort_priority, ascending=[False] * len(sort_priority))
        full_df_sorted.index = range(1, len(full_df_sorted) + 1)

        # 4. Display only what was selected
        display_cols = ["Team"] + [c for c in selected_cols if c != "Status"]
        # Deduplicate columns just in case
        display_cols = list(dict.fromkeys(display_cols))

        display_df = full_df_sorted[display_cols].copy()

        styled = sim.fancy_highlight_half_split(display_df)
        st.session_state.styled_df = styled
        st.markdown(styled.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please select at least one column to display the table.")

    st.markdown("---")

    # ============================================================
    # Exact Combination Tracker (Remains in main content area)
    # ============================================================
    with st.expander("🎯 Exact Combination Tracker", expanded=False):
        st.markdown(
            "Select **2–4 teams** and find out how often **exactly those teams** "
            "appear together in the Top 4 across Elo-weighted simulations."
        )

        combo_teams = st.multiselect(
            "Select teams",
            sim.teams,
            key="combo_teams_select",
            help="Select 2–4 teams. The tracker counts simulations where those are the *exact* Top 4 (or a subset — see note).",
        )

        combo_sims = st.select_slider(
            "Simulations",
            options=[10_000, 25_000, 50_000, 100_000],
            value=50_000,
            key="combo_sims_slider",
        )

        if 2 <= len(combo_teams) <= 4:
            note = (
                "**Note:** When fewer than 4 teams are selected, the tracker checks whether "
                "those teams are all *inside* the top 4 together (not necessarily as the exclusive top 4)."
                if len(combo_teams) < 4 else
                "The tracker checks whether **exactly** these 4 teams form the complete top 4."
            )
            st.caption(note)

            if st.button("🔍 Calculate Combination Probability", key="calc_combo_btn"):
                override = what_if_matches if st.session_state.get("what_if_applied") else None
                with st.spinner(f"Running {combo_sims:,} simulations…"):
                    # For <4 teams: count simulations where all combo teams are in top 4
                    # (not necessarily exclusively). We do this by passing combo_set;
                    # if len < 4 the function will almost never match exactly 4, so we
                    # use a wrapper that checks subset instead.
                    if len(combo_teams) == 4:
                        prob = sim.count_combination(combo_teams, combo_sims, override)
                    else:
                        # Subset check — run custom inline logic
                        import numpy as np

                        matches_src = override if override else sim.remaining_matches
                        base_pts, base_nrr, pending, _ = sim.get_what_if_baseline(matches_src)
                        combo_set = set(combo_teams)

                        if not pending:
                            sorted_t = sorted(sim.teams, key=lambda t: (base_pts[t], base_nrr[t]), reverse=True)
                            actual_top4 = set(sorted_t[:4])
                            prob = 100.0 if combo_set.issubset(actual_top4) else 0.0
                        else:
                            match_probs = sim._build_match_probs(pending)
                            hit = 0
                            for _ in range(combo_sims):
                                pts = base_pts.copy()
                                for h, a, _v, p in match_probs:
                                    winner = h if np.random.rand() < p else a
                                    pts[winner] += 2
                                sp = sorted(pts.items(), key=lambda x: x[1], reverse=True)
                                f4 = sp[3][1]
                                above4 = [t for t, p2 in sp if p2 > f4]
                                tied4 = [t for t, p2 in sp if p2 == f4]
                                spots4 = 4 - len(above4)
                                if spots4 > 0 and tied4:
                                    chosen = np.random.choice(tied4, size=min(spots4, len(tied4)),
                                                              replace=False).tolist()
                                    top4 = set(above4) | set(chosen)
                                else:
                                    top4 = set(above4[:4])
                                if combo_set.issubset(top4):
                                    hit += 1
                            prob = round(hit / combo_sims * 100, 2)

                    st.session_state.combo_result = {
                        "teams": combo_teams,
                        "prob": prob,
                        "sims": combo_sims,
                        "exact": len(combo_teams) == 4,
                    }

        elif len(combo_teams) > 4:
            st.warning("Please select at most 4 teams.")
        else:
            st.caption("Select at least 2 teams to begin.")

        if "combo_result" in st.session_state and st.session_state.combo_result:
            cr = st.session_state.combo_result
            teams_str = " + ".join(cr["teams"])
            label_type = "exactly these 4 teams as the Top 4" if cr["exact"] else "all of these teams inside the Top 4"
            st.metric(
                label=f"Probability of {label_type}",
                value=f"{cr['prob']:.2f}%",
                help=f"Based on {cr['sims']:,} Elo-weighted simulations.",
            )
            if cr["prob"] == 0.0:
                st.error("🚫 This combination appears impossible given current standings and remaining schedule.")
            elif cr["prob"] < 1.0:
                st.warning("⚠️ Very unlikely — less than 1% of simulations.")
            elif cr["prob"] > 80.0:
                st.success("✅ Highly likely — more than 80% of simulations.")

    st.markdown("---")

    # --- Elo ratings ---
    with st.expander("📊 Current Team Elo Ratings", expanded=False):
        st.markdown("""
**How Elo ratings are calculated:**
 
Pre-season ratings are seeded from two components blended together:
- **90% squad quality** — each team's expected XI is rated across four dimensions (batting impact, bowling impact, experience, T20 reputation) on a 0–10 scale. The top 11 players are aggregated with diminishing returns (12th man contributes less than the 1st), then normalised across all teams.
- **10% last season finish** — 2025 final league standings converted to a 0–1 score (1st = 1.0, 10th = 0.0), giving a small but meaningful nudge to teams that performed well recently.
 
The combined score is then normalised to a **1485–1515 range** — a deliberately narrow spread reflecting pre-season uncertainty. A 30-point gap between strongest and weakest translates to roughly a 54–46 win probability per match.
 
**After each match**, Elo updates via a margin-aware formula:
 
`new_rating = old_rating + K × (actual_score − expected_score)`
 
Where K = 32, expected score is derived from the pre-match Elo gap, and actual score is a smooth function of NRR margin (a big win moves ratings more than a narrow one). This means ratings self-calibrate over the season — form matters more as matches accumulate.
        """)
        elo_df = sim.get_elo_table()
        elo_df.index = range(1, len(elo_df) + 1)
        st.markdown(style_elo_table(elo_df).to_html(escape=False), unsafe_allow_html=True)

    # --- Download buttons ---
    prefix = "whatif" if st.session_state.get("what_if_applied") else "post"
    match_id = (
        f"m{sim.MATCHES_COMMITTED + applied_count}"
        if st.session_state.get("what_if_applied")
        else f"m{sim.MATCHES_COMMITTED}"
    )
    timestamp = st.session_state.timestamp
    csv_filename = f"{prefix}_{match_id}_results_{timestamp}.csv"
    excel_filename = f"{prefix}_{match_id}_stylized_{timestamp}.xlsx"

    csv_buffer = io.StringIO()
    st.session_state.simulation_df.to_csv(csv_buffer, index=False)

    excel_buffer = io.BytesIO()
    if st.session_state.styled_df is not None:
        st.session_state.styled_df.to_excel(excel_buffer, engine="openpyxl", index=False)
    excel_buffer.seek(0)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ Download CSV", data=csv_buffer.getvalue(),
                           file_name=csv_filename, mime="text/csv")
    with dl2:
        st.download_button("⬇️ Download Excel", data=excel_buffer,
                           file_name=excel_filename,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ============================================================
    # SECTION 6: The Playoff Bracket (Only appears at Match 70)
    # ============================================================
    import numpy as np

    applied_count = sum(1 for m in what_if_matches if m.get("applied")) if st.session_state.get(
        "what_if_applied") else 0

    if sim.MATCHES_COMMITTED + applied_count == 70:
        st.markdown("---")


        # --- NEW HELPER: Generates a realistic T20 Scorecard ---
        def generate_scorecard(winner, loser):
            winner_bats_first = np.random.rand() > 0.5

            if winner_bats_first:
                # Winner sets a target and defends it
                w_runs = int(np.random.normal(190, 25))  # Avg score 190
                w_wkts = np.random.randint(3, 9)

                margin = np.random.randint(4, 35)  # Wins by 4 to 35 runs
                l_runs = w_runs - margin
                l_wkts = np.random.randint(6, 11)  # 10 means all out
                l_ov = "20.0" if l_wkts < 10 else f"{np.random.randint(16, 19)}.{np.random.randint(1, 6)}"

                l_wkts_str = "All Out" if l_wkts == 10 else f"{l_wkts}"
                w_wkts_str = f"{w_wkts}"

                return f"**{winner}** {w_runs}/{w_wkts_str} (20.0) defeated **{loser}** {l_runs}/{l_wkts_str} ({l_ov}) by {margin} runs."
            else:
                # Loser sets a target, Winner chases it down
                l_runs = int(np.random.normal(175, 20))
                l_wkts = np.random.randint(5, 10)

                w_runs = l_runs + np.random.randint(1, 6)  # Chase target with a boundary/single
                w_wkts = np.random.randint(2, 8)
                w_ov = f"{np.random.randint(17, 19)}.{np.random.randint(1, 6)}"  # Winning in the death overs

                l_wkts_str = "All Out" if l_wkts == 10 else f"{l_wkts}"
                wickets_won_by = 10 - w_wkts

                return f"**{loser}** {l_runs}/{l_wkts_str} (20.0) lost to **{winner}** {w_runs}/{w_wkts} ({w_ov}) by {wickets_won_by} wickets."


        # -------------------------------------------------------

        head_col1, head_col2 = st.columns([3, 1])
        with head_col1:
            st.header("🏆 IPL 2026 Playoff Bracket")
        with head_col2:
            if st.button("🎲 Re-Simulate Playoffs", use_container_width=True):
                st.rerun()

        st.balloons()

        # Safely pull and sort the Top 4 directly from the base simulation data
        safe_df = st.session_state.simulation_df.sort_values(
            by=["Avg Final Points", "Avg Final NRR"],
            ascending=[False, False]
        )
        top4 = safe_df['Team'].tolist()[:4]

        st.markdown(f"**The Elite Four:** 1. {top4[0]} | 2. {top4[1]} | 3. {top4[2]} | 4. {top4[3]}")

        col1, col2 = st.columns(2)

        # --- QUALIFIER 1 ---
        with col1:
            st.subheader("🥇 Qualifier 1")
            st.info(f"**{top4[0]}** vs **{top4[1]}**")
            p1, p2 = sim.get_win_probability(top4[0], top4[1], "Neutral")

            roll = np.random.rand()
            q1_winner = top4[0] if roll < (p1 / 100.0) else top4[1]
            q1_loser = top4[1] if q1_winner == top4[0] else top4[0]

            upset_tag = "🔥 (Upset!)" if (q1_winner == top4[0] and p1 < 50) or (
                        q1_winner == top4[1] and p2 < 50) else ""
            st.write(f"📈 Matchup Odds: {top4[0]} ({p1:.1f}%) | {top4[1]} ({p2:.1f}%)")
            st.write(f"🏏 **Score:** {generate_scorecard(q1_winner, q1_loser)}")
            st.markdown(f"*{q1_winner} advances directly to the Final. {upset_tag}*")

        # --- ELIMINATOR ---
        with col2:
            st.subheader("🥉 Eliminator")
            st.warning(f"**{top4[2]}** vs **{top4[3]}**")
            p3, p4 = sim.get_win_probability(top4[2], top4[3], "Neutral")

            roll = np.random.rand()
            el_winner = top4[2] if roll < (p3 / 100.0) else top4[3]

            upset_tag = "🔥 (Upset!)" if (el_winner == top4[2] and p3 < 50) or (
                        el_winner == top4[3] and p4 < 50) else ""
            st.write(f"📈 Matchup Odds: {top4[2]} ({p3:.1f}%) | {top4[3]} ({p4:.1f}%)")
            st.write(f"🏏 **Score:** {generate_scorecard(el_winner, top4[2] if el_winner == top4[3] else top4[3])}")
            st.markdown(f"*{el_winner} survives and moves to Qualifier 2. {upset_tag}*")

        st.markdown("---")

        col3, col4 = st.columns(2)

        # --- QUALIFIER 2 ---
        with col3:
            st.subheader("🥈 Qualifier 2")
            st.info(f"**{q1_loser}** vs **{el_winner}**")
            p_q2_1, p_q2_2 = sim.get_win_probability(q1_loser, el_winner, "Neutral")

            roll = np.random.rand()
            q2_winner = q1_loser if roll < (p_q2_1 / 100.0) else el_winner

            upset_tag = "🔥 (Upset!)" if (q2_winner == q1_loser and p_q2_1 < 50) or (
                        q2_winner == el_winner and p_q2_2 < 50) else ""
            st.write(f"📈 Matchup Odds: {q1_loser} ({p_q2_1:.1f}%) | {el_winner} ({p_q2_2:.1f}%)")
            st.write(
                f"🏏 **Score:** {generate_scorecard(q2_winner, q1_loser if q2_winner == el_winner else el_winner)}")
            st.markdown(f"*{q2_winner} books the final ticket. {upset_tag}*")

        # --- THE FINAL ---
        with col4:
            st.subheader("🏆 The Grand Final")
            st.success(f"**{q1_winner}** vs **{q2_winner}**")
            p_f1, p_f2 = sim.get_win_probability(q1_winner, q2_winner, "Neutral")

            roll = np.random.rand()
            champion = q1_winner if roll < (p_f1 / 100.0) else q2_winner
            runner_up = q1_winner if champion == q2_winner else q2_winner

            st.write(f"📈 Matchup Odds: {q1_winner} ({p_f1:.1f}%) | {q2_winner} ({p_f2:.1f}%)")
            st.write(f"🏏 **Score:** {generate_scorecard(champion, runner_up)}")

        st.markdown(f"<h2 style='text-align: center; color: gold;'>👑 Projected Champion: {champion} 👑</h2>",
                    unsafe_allow_html=True)

# ============================================================
# PRE-SIMULATION VIEW
# ============================================================
else:
    st.markdown("### 📌 Current IPL Points Table")
    current_table = pd.DataFrame(sim.get_current_points_table())
    current_table = add_form_column(current_table)
    current_table.index = range(1, len(current_table) + 1)
    st.markdown(style_points_table(current_table).to_html(escape=False), unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("📊 Current Team Elo Ratings", expanded=False):
        st.markdown("""
**How Elo ratings are calculated:**
 
Pre-season ratings are seeded from two components blended together:
- **90% squad quality** — each team's expected XI is rated across four dimensions (batting impact, bowling impact, experience, T20 reputation) on a 0–10 scale. The top 11 players are aggregated with diminishing returns (12th man contributes less than the 1st), then normalised across all teams.
- **10% last season finish** — 2025 final league standings converted to a 0–1 score (1st = 1.0, 10th = 0.0), giving a small but meaningful nudge to teams that performed well recently.
 
The combined score is then normalised to a **1485–1515 range** — a deliberately narrow spread reflecting pre-season uncertainty. A 30-point gap between strongest and weakest translates to roughly a 54–46 win probability per match.
 
**After each match**, Elo updates via a margin-aware formula:
 
`new_rating = old_rating + K × (actual_score − expected_score)`
 
Where K = 32, expected score is derived from the pre-match Elo gap, and actual score is a smooth function of NRR margin (a big win moves ratings more than a narrow one). This means ratings self-calibrate over the season — form matters more as matches accumulate.
        """)
        elo_df = sim.get_elo_table()
        elo_df.index = range(1, len(elo_df) + 1)
        st.markdown(style_elo_table(elo_df).to_html(escape=False), unsafe_allow_html=True)

    st.info(
        "Click **Run Simulation** in the sidebar to generate qualification probabilities, "
        "or use the **What-if Scenario Editor** to simulate upcoming matches."
    )