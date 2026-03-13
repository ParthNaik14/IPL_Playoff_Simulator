import streamlit as st
import pandas as pd
from datetime import datetime
import io
import ipl_simulator as sim

# --- Page setup ---
st.set_page_config(page_title="IPL Playoff Simulator", layout="wide")
st.title("📊 IPL 2026 Playoff Qualification Simulator")

query_params = st.query_params
if "reset" in query_params:
    reset_type = query_params["reset"]
    if reset_type == "everything":
        st.success("✅ Everything has been reset.")
    elif reset_type == "scenarios":
        st.success("✅ All What-if scenario inputs have been cleared.")
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


# --- Run Simulations ---
if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner("Running simulations... please wait."):
        df = sim.run_parallel_simulations(total_simulations, processes=processes)
        pure_math = sim.run_pure_math_simulation_parallel(total_simulations, processes=processes)
        df["Top 4 Pure Math (%)"] = df["Team"].map(pure_math)
        df.index = range(1, len(df) + 1)
        styled_df = sim.fancy_highlight_half_split(df.copy())

        st.session_state.simulation_df = df
        st.session_state.styled_df = styled_df
        st.session_state.match_number = sim.MATCHES_COMMITTED  # ✅ fixed
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.session_state["what_if_applied"] = False


# --- Reset Buttons ---
if st.sidebar.button("🔄 Reset Everything"):
    reset_matches = [
        {
            "home": match["home"],
            "away": match["away"],
            "venue": match["venue"],
            "result": None,
            "margin": None,
            "applied": False
        }
        for match in sim.remaining_matches
    ]
    sim.set_what_if_results(reset_matches)

    for i in range(len(sim.remaining_matches)):
        st.session_state[f"whatif_result_{i}"] = ""
        st.session_state.pop(f"winner_runs_{i}", None)
        st.session_state.pop(f"winner_overs_{i}", None)
        st.session_state.pop(f"loser_runs_{i}", None)
        st.session_state.pop(f"loser_overs_{i}", None)

    st.session_state.simulation_df = None
    st.session_state.styled_df = None
    st.session_state.match_number = sim.MATCHES_COMMITTED  # ✅ fixed
    st.session_state.timestamp = None
    st.session_state.what_if_applied = False

    st.query_params["reset"] = "everything"
    st.rerun()


if st.sidebar.button("♻️ Reset What-if Scenarios"):
    for i in range(len(sim.remaining_matches)):
        st.session_state[f"whatif_result_{i}"] = ""
        st.session_state.pop(f"winner_runs_{i}", None)
        st.session_state.pop(f"winner_overs_{i}", None)
        st.session_state.pop(f"loser_runs_{i}", None)
        st.session_state.pop(f"loser_overs_{i}", None)

    st.query_params["reset"] = "scenarios"
    st.rerun()


# --- What-if Scenario Editor ---
st.sidebar.markdown("### 🧪 What-if Scenario Editor")
what_if_matches = []

for i, match in enumerate(sim.remaining_matches):
    match_number = sim.MATCHES_COMMITTED + i + 1  # ✅ fixed: counts from real committed matches
    with st.sidebar.expander(
        f"📌 Match {match_number}: {match['home']} vs {match['away']}", expanded=False
    ):
        winner = st.selectbox(
            "Select Result",
            options=["", match["home"], match["away"], "Abandoned/No Result (1 point each)"],
            key=f"whatif_result_{i}"
        )

        match_dict = {
            "home":   match["home"],
            "away":   match["away"],
            "venue":  match["venue"],
            "result": None,
            "applied": False,
            "runs":   {},
            "overs":  {}
        }

        if winner:
            if winner == "Abandoned/No Result (1 point each)":
                match_dict["result"] = "Abandoned/No Result (1 point each)"
                match_dict["applied"] = True
            else:
                loser = match["away"] if winner == match["home"] else match["home"]

                winner_runs  = st.number_input(f"{winner} Runs",  min_value=0, value=None, key=f"winner_runs_{i}")
                winner_overs = st.text_input(f"{winner} Overs (e.g., 19.3)", key=f"winner_overs_{i}")
                loser_runs   = st.number_input(f"{loser} Runs",   min_value=0, value=None, key=f"loser_runs_{i}")
                loser_overs  = st.text_input(f"{loser} Overs (e.g., 20.0)",  key=f"loser_overs_{i}")

                match_dict["result"]  = winner
                match_dict["applied"] = True
                match_dict["runs"]    = {winner: winner_runs,  loser: loser_runs}
                match_dict["overs"]   = {winner: winner_overs, loser: loser_overs}

        what_if_matches.append(match_dict)


# --- Apply What-if ---
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
        df["Top 4 Pure Math (%)"] = df["Team"].map(pure_math)
        df.index = range(1, len(df) + 1)
        styled_df = sim.fancy_highlight_half_split(df.copy())

        st.session_state.simulation_df = df
        st.session_state.styled_df = styled_df
        st.session_state.match_number = sim.MATCHES_COMMITTED  # ✅ fixed
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    st.session_state["what_if_applied"] = True
    st.success("✅ What-if simulation applied successfully!")


# --- Display Results ---
if st.session_state.simulation_df is not None:

    if st.session_state.get("what_if_applied"):
        st.subheader("📋 What-if Points Table (After Applied Matches)")
        whatif_table = sim.get_points_table_after_what_if(what_if_matches)
        whatif_table.index = range(1, len(whatif_table) + 1)

        def style_points_table(df):
            styled = df.style.format({"NRR": "{:.3f}"})
            styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})
            styled = styled.set_properties(**{"text-align": "center"})
            def team_color(val):
                if val in sim.team_colors:
                    bg = sim.team_colors[val]['bg']
                    fg = sim.team_colors[val]['text']
                    return f"background-color: {bg}; color: {fg}"
                return ""
            styled = styled.map(team_color, subset=["Team"])
            return styled

        st.markdown(style_points_table(whatif_table).to_html(escape=False), unsafe_allow_html=True)

    if not st.session_state.get("what_if_applied"):
        st.markdown("### 📌 Current IPL Points Table")
        current_table = pd.DataFrame(sim.get_current_points_table())
        current_table.index = range(1, len(current_table) + 1)

        def style_points_table(df):
            styled = df.style.format({"NRR": "{:.3f}"})
            styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})
            styled = styled.set_properties(**{"text-align": "center"})
            def color_team_cells(val):
                if val in sim.team_colors:
                    c = sim.team_colors[val]
                    return f"background-color: {c['bg']}; color: {c['text']}"
                return ""
            styled = styled.map(color_team_cells, subset=["Team"])
            return styled

        st.markdown(style_points_table(current_table).to_html(escape=False), unsafe_allow_html=True)

    # Applied What-if summary (margin-based, from old flow — kept for compatibility)
    applied_matches = [
        match for match in sim.remaining_matches
        if match.get("result") in [match["home"], match["away"]] and match.get("margin") is not None
    ]
    if applied_matches:
        st.markdown("### 🎯 Applied What-if Scenarios")
        data = []
        for match in applied_matches:
            winner = match["result"]
            loser  = match["away"] if winner == match["home"] else match["home"]
            margin = f"{match['margin']:+.3f}"
            data.append({
                "Match":  f"{match['home']} vs {match['away']}",
                "Winner": winner,
                "Margin": margin
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    # --- Simulation results heading ---
    applied_count = sum(1 for match in what_if_matches if match.get("applied"))

    if st.session_state.get("what_if_applied"):
        # ✅ Fixed: base is MATCHES_COMMITTED, add what-if applied on top
        simulated_match_number = sim.MATCHES_COMMITTED + applied_count
        st.subheader(f"📝 What-if Simulation Results - Post Match {simulated_match_number:02d}")
    else:
        st.subheader(f"📝 Simulation Results - Post Match {sim.MATCHES_COMMITTED:02d}")

    # Styled results table
    if st.session_state.get("styled_df") is not None:
        st.markdown(st.session_state.styled_df.to_html(escape=False), unsafe_allow_html=True)

    # --- Download buttons ---
    prefix   = "whatif" if st.session_state.get("what_if_applied") else "post"
    match_id = (
        f"m{sim.MATCHES_COMMITTED + applied_count}"
        if st.session_state.get("what_if_applied")
        else f"m{sim.MATCHES_COMMITTED}"
    )
    timestamp      = st.session_state.timestamp
    csv_filename   = f"{prefix}_{match_id}_results_{timestamp}.csv"
    excel_filename = f"{prefix}_{match_id}_stylized_{timestamp}.xlsx"

    csv_buffer = io.StringIO()
    st.session_state.simulation_df.to_csv(csv_buffer, index=False)

    excel_buffer = io.BytesIO()
    st.session_state.styled_df.to_excel(excel_buffer, engine='openpyxl', index=False)
    excel_buffer.seek(0)

    st.download_button(
        "⬇️ Download CSV", data=csv_buffer.getvalue(),
        file_name=csv_filename, mime="text/csv"
    )
    st.download_button(
        "⬇️ Download Excel", data=excel_buffer,
        file_name=excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info(
        "Click **Run Simulation** from the sidebar to run simulations of already completed matches "
        "and **Apply What-If Scenarios** to run what if simulations."
    )