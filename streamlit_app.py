import streamlit as st
import pandas as pd
from datetime import datetime
import io
import ipl_simulator as sim

# --- Page setup ---
st.set_page_config(page_title="IPL Playoff Simulator", layout="wide")
st.title("📊 IPL 2026 Playoff Qualification Simulator")

# ---------------------------------------------------------------------------
# Admin mode — only True when secrets.toml contains admin_mode = true
# Never true on public Streamlit Cloud deployment (no secrets file there)
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
        st.success(f"✅ Match committed successfully!")
    elif reset_type == "decommitted":
        st.success(f"↩️ Last match decommitted successfully.")
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


def _clear_whatif_state():
    for i in range(len(sim.remaining_matches) + 1):
        st.session_state.pop(f"whatif_result_{i}", None)
        st.session_state.pop(f"winner_runs_{i}", None)
        st.session_state.pop(f"winner_overs_{i}", None)
        st.session_state.pop(f"loser_runs_{i}", None)
        st.session_state.pop(f"loser_overs_{i}", None)


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

        # Two-step confirmation to prevent accidental decommits
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
if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner("Running simulations... please wait."):
        df = sim.run_parallel_simulations(total_simulations, processes=processes)
        pure_math = sim.run_pure_math_simulation_parallel(total_simulations, processes=processes)
        df["Still Possible %"] = df["Team"].map(pure_math["top4"])
        df["Top 2 Still Possible %"] = df["Team"].map(pure_math["top2"])
        df = df[["Team", "Qualify %", "Top 2 %", "Safe by Points %",
                 "Safe Top 2 %", "Still Possible %", "Top 2 Still Possible %",
                 "Avg Final Points", "Avg Final NRR"]]
        df.index = range(1, len(df) + 1)
        styled_df = sim.fancy_highlight_half_split(df.copy())
        st.session_state.simulation_df = df
        st.session_state.styled_df = styled_df
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
        "changes and inter-season uncertainty. Use once per season before "
        "updating squad-based ratings."
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
    st.query_params["reset"] = "everything"
    st.rerun()

if st.sidebar.button("♻️ Reset What-if Scenarios"):
    _clear_whatif_state()
    st.query_params["reset"] = "scenarios"
    st.rerun()

# ============================================================
# SECTION 5 — WHAT-IF SCENARIO EDITOR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 What-if Scenario Editor")
what_if_matches = []

for i, match in enumerate(sim.remaining_matches):
    match_number = sim.MATCHES_COMMITTED + i + 1
    with st.sidebar.expander(
            f"📌 Match {match_number}: {match['home']} vs {match['away']}", expanded=False
    ):
        winner = st.selectbox(
            "Select Result",
            options=["", match["home"], match["away"], "Abandoned/No Result (1 point each)"],
            key=f"whatif_result_{i}"
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
                winner_runs = st.number_input(f"{winner} Runs", min_value=0, value=None, key=f"winner_runs_{i}")
                winner_overs = st.text_input(f"{winner} Overs (e.g., 19.3)", key=f"winner_overs_{i}")
                loser_runs = st.number_input(f"{loser} Runs", min_value=0, value=None, key=f"loser_runs_{i}")
                loser_overs = st.text_input(f"{loser} Overs (e.g., 20.0)", key=f"loser_overs_{i}")
                match_dict["result"] = winner
                match_dict["applied"] = True
                match_dict["runs"] = {winner: winner_runs, loser: loser_runs}
                match_dict["overs"] = {winner: winner_overs, loser: loser_overs}
        what_if_matches.append(match_dict)

if st.sidebar.button("✅ Apply What-if Scenarios"):
    sim.set_what_if_results(what_if_matches)
    st.session_state.what_if_applied = True
    with st.spinner("Applying What-if and rerunning simulations..."):
        df = sim.run_parallel_simulations(
            total_simulations, processes=processes, override_matches=what_if_matches
        )
        pure_math = sim.run_pure_math_simulation_parallel(
            total_simulations, processes=processes, override_matches=what_if_matches
        )
        df["Still Possible %"] = df["Team"].map(pure_math["top4"])
        df["Top 2 Still Possible %"] = df["Team"].map(pure_math["top2"])
        df = df[["Team", "Qualify %", "Top 2 %", "Safe by Points %",
                 "Safe Top 2 %", "Still Possible %", "Top 2 Still Possible %",
                 "Avg Final Points", "Avg Final NRR"]]
        df.index = range(1, len(df) + 1)
        styled_df = sim.fancy_highlight_half_split(df.copy())
        st.session_state.simulation_df = df
        st.session_state.styled_df = styled_df
        st.session_state.match_number = sim.MATCHES_COMMITTED
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.session_state["what_if_applied"] = True
    st.success("✅ What-if simulation applied successfully!")


# ============================================================
# MAIN CONTENT
# ============================================================
def style_points_table(df):
    styled = df.style.format({"NRR": "{:.3f}"})
    styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})
    styled = styled.set_properties(**{"text-align": "center"})

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


COLUMN_EXPLAINER = """
### 📖 How to read the simulation results

The simulator runs **10,000 Monte Carlo simulations** of the remaining matches. Each simulation randomly determines winners based on team strength, home advantage, and current form, then ranks all 10 teams. Here's what each column means:

| Column | What it measures | Parameters used |
|--------|-----------------|-----------------|
| **Qualify %** | % of simulations where this team finished in the top 4. The primary qualification metric. Uses NRR to break points ties. | Elo rating · Recent form (last 5 games) · Win % · NRR · Home advantage |
| **Top 2 %** | % of simulations where this team finished top 2, earning a direct final berth. Uses NRR to break points ties. | Same as Qualify % |
| **Safe by Points %** | % of simulations where this team accumulated enough wins to finish top 4 **strictly on points alone** — even if NRR went against them. Uses the same strength-weighted win probabilities as Qualify %, but only checks the points standings at the end (NRR ignored for final rank). Will always be ≤ Qualify %. | Same win probability model as Qualify % · Final ranking by points only |
| **Safe Top 2 %** | Same as Safe by Points %, but for a top 2 finish on points alone. | Same win probability model as Top 2 % · Final ranking by points only |
| **Still Possible %** | Mathematical ceiling for top 4. If every remaining game is a **pure 50/50 coin flip** regardless of team strength, how often does this team finish top 4? When this hits 0%, top 4 is mathematically impossible. | Equal 50/50 win probability · Points only (no NRR) |
| **Top 2 Still Possible %** | Same as Still Possible %, but for a top 2 finish. When this hits 0%, a direct final berth is mathematically impossible. | Equal 50/50 win probability · Points only (no NRR) |
| **Avg Final Points** | Average points this team finishes on across all simulations, rounded to nearest even number. **Most meaningful mid-season** once several matches have been played — early in the season this only reflects points from the currently loaded schedule. | Same as Qualify % |
| **Avg Final NRR** | Average NRR this team finishes on across all simulations (3 decimal places). Most useful when multiple teams are projected on the same points — NRR decides who qualifies. Green = positive, red = negative. | Same as Qualify % |

> **Win probability model:** Each match probability = `team_strength / (team_strength + opponent_strength)`, where strength is a weighted blend of Elo (45–55%), recent form (0–30%), win % (15%), and NRR (remaining %). Home advantage adds a venue-specific multiplier (1.03–1.08×).
"""

if st.session_state.simulation_df is not None:

    if st.session_state.get("what_if_applied"):
        st.subheader("📋 What-if Points Table (After Applied Matches)")
        whatif_table = sim.get_points_table_after_what_if(what_if_matches)
        whatif_table.index = range(1, len(whatif_table) + 1)
        st.markdown(style_points_table(whatif_table).to_html(escape=False), unsafe_allow_html=True)
    else:
        st.markdown("### 📌 Current IPL Points Table")
        current_table = pd.DataFrame(sim.get_current_points_table())
        current_table.index = range(1, len(current_table) + 1)
        st.markdown(style_points_table(current_table).to_html(escape=False), unsafe_allow_html=True)

    st.markdown("---")

    applied_count = sum(1 for m in what_if_matches if m.get("applied"))
    if st.session_state.get("what_if_applied"):
        simulated_match_number = sim.MATCHES_COMMITTED + applied_count
        st.subheader(f"📝 What-if Simulation Results - Post Match {simulated_match_number:02d}")
    else:
        st.subheader(f"📝 Simulation Results - Post Match {sim.MATCHES_COMMITTED:02d}")

    # Column explainer
    with st.expander("ℹ️ How to read this table", expanded=False):
        st.markdown(COLUMN_EXPLAINER)

    # Toggles for optional columns
    col_toggle1, col_toggle2 = st.columns(2)
    with col_toggle1:
        show_avg_pts = st.checkbox(
            "Show Avg Final Points",
            value=False,
            help="Average points projected to finish on. Most meaningful mid-season once several matches have been played."
        )
    with col_toggle2:
        show_avg_nrr = st.checkbox(
            "Show Avg Final NRR",
            value=False,
            help="Average NRR projected to finish on. Most useful for teams clustered on the same points where NRR decides qualification."
        )

    if st.session_state.get("styled_df") is not None:
        display_df = st.session_state.simulation_df.copy()
        cols_to_drop = []
        if not show_avg_pts and "Avg Final Points" in display_df.columns:
            cols_to_drop.append("Avg Final Points")
        if not show_avg_nrr and "Avg Final NRR" in display_df.columns:
            cols_to_drop.append("Avg Final NRR")
        if cols_to_drop:
            display_df = display_df.drop(columns=cols_to_drop)
        styled = sim.fancy_highlight_half_split(display_df)
        st.markdown(styled.to_html(escape=False), unsafe_allow_html=True)

    st.markdown("---")

    # Elo ratings table
    with st.expander("📊 Current Team Elo Ratings", expanded=False):
        st.caption(
            "Elo ratings represent each team's estimated strength. They are seeded from "
            "squad quality (90%) and 2025 season finish (10%) before the tournament, then "
            "updated automatically after each committed match result based on the margin of victory."
        )
        elo_df = sim.get_elo_table()
        elo_df.index = range(1, len(elo_df) + 1)
        st.markdown(style_elo_table(elo_df).to_html(escape=False), unsafe_allow_html=True)

    # Download buttons
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
    st.session_state.styled_df.to_excel(excel_buffer, engine='openpyxl', index=False)
    excel_buffer.seek(0)

    st.download_button("⬇️ Download CSV", data=csv_buffer.getvalue(),
                       file_name=csv_filename, mime="text/csv")
    st.download_button("⬇️ Download Excel", data=excel_buffer,
                       file_name=excel_filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.markdown("### 📌 Current IPL Points Table")
    current_table = pd.DataFrame(sim.get_current_points_table())
    current_table.index = range(1, len(current_table) + 1)
    st.markdown(style_points_table(current_table).to_html(escape=False), unsafe_allow_html=True)

    st.markdown("---")

    # Show Elo ratings pre-simulation too — useful context before season starts
    with st.expander("📊 Pre-season Team Elo Ratings", expanded=False):
        st.caption(
            "Elo ratings represent each team's estimated strength, seeded from "
            "squad quality (90%) and 2025 season finish (10%). They update after "
            "each match based on result and margin of victory."
        )
        elo_df = sim.get_elo_table()
        elo_df.index = range(1, len(elo_df) + 1)
        st.markdown(style_elo_table(elo_df).to_html(escape=False), unsafe_allow_html=True)

    st.info(
        "Click **Run Simulation** in the sidebar to generate qualification probabilities, "
        "or use the **What-if Scenario Editor** to simulate upcoming matches."
    )