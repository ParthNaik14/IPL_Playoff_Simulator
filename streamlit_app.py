import streamlit as st
import pandas as pd
from datetime import datetime
import io
import ipl_simulator as sim  # your simulator logic file

# --- Page setup ---
st.set_page_config(page_title="IPL Playoff Simulator", layout="wide")
st.title("📊 IPL 2025 Playoff Qualification Simulator")

# --- Sidebar: Simulation controls ---
total_simulations = st.sidebar.number_input("Total Simulations", value=10000, step=2000, min_value=2000, max_value=14000605)
processes = st.sidebar.slider("Parallel Processes", min_value=2, max_value=128, value=16, step=2)

# --- Persistent state ---
if "simulation_df" not in st.session_state:
    st.session_state.simulation_df = None
    st.session_state.styled_df = None
    st.session_state.match_number = None
    st.session_state.timestamp = None



# --- What-if Scenario Editor ---
st.sidebar.markdown("### 🧪 What-if Scenario Editor")
what_if_matches = []

# Load the current match data from backend
for i, match in enumerate(sim.remaining_matches):
    match_number = 70 - len(sim.remaining_matches) + i + 1
    with st.sidebar.expander(f"📌 Match {match_number}: {match['home']} vs {match['away']}", expanded=False):
        winner = st.selectbox(
            "Select Winner",
            options=["", match["home"], match["away"]],
            key=f"whatif_result_{i}"
        )

        # Default structure
        match_dict = {
            "home": match["home"],
            "away": match["away"],
            "venue": match["venue"],
            "result": None,
            "applied": False,
            "runs": {},
            "overs": {}
        }

        if winner:
            loser = match["away"] if winner == match["home"] else match["home"]

            # Winner inputs
            winner_runs = st.number_input(f"{winner} Runs", min_value=0, value=None,  key=f"winner_runs_{i}")
            winner_overs = st.text_input(f"{winner} Overs (e.g., 19.3)", key=f"winner_overs_{i}")

            # Loser inputs
            loser_runs = st.number_input(f"{loser} Runs", min_value=0, value=None, key=f"loser_runs_{i}")
            loser_overs = st.text_input(f"{loser} Overs (e.g., 20.0)", key=f"loser_overs_{i}")

            # Fill in result
            match_dict["result"] = winner
            match_dict["applied"] = True
            match_dict["runs"] = {
                winner: winner_runs,
                loser: loser_runs
            }
            match_dict["overs"] = {
                winner: winner_overs,
                loser: loser_overs
            }

        what_if_matches.append(match_dict)
        # DEBUG print to inspect What-if match structure
        #st.sidebar.write("DEBUG: What-if Matches", what_if_matches)
        # Debug output
       # st.sidebar.code(match_dict, language="json")

# Apply What-if Scenario
if st.sidebar.button("✅ Apply What-if Scenarios"):
    sim.set_what_if_results(what_if_matches)
    st.session_state.what_if_applied = True  # 👈 NEW: Track that it's a What-if

    with st.spinner("Applying What-if and rerunning simulations..."):
        df = sim.run_parallel_simulations(total_simulations, processes=processes, override_matches=what_if_matches)


        pure_math = sim.run_pure_math_simulation_parallel(total_simulations, processes=processes, override_matches=what_if_matches)


        insert_idx = df.columns.get_loc("Avg Final Points")
        df.insert(loc=insert_idx, column="Top 4 Pure Math (%)", value=df["Team"].map(pure_math))
        df.index = range(1, len(df) + 1)
        styled_df = sim.fancy_highlight_half_split(df.copy())

        st.session_state.simulation_df = df
        st.session_state.styled_df = styled_df
        st.session_state.match_number = 70 - len(sim.remaining_matches)
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    st.session_state["what_if_applied"] = True
    st.success("✅ What-if simulation applied successfully!")




# Reset What-if Scenario
if st.sidebar.button("🔄 Reset What-if Scenarios"):
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
    st.warning("What-if results have been reset.")




# --- Run Simulations ---
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulations... please wait."):

        # Run simulations
        df = sim.run_parallel_simulations(total_simulations, processes=processes)
        pure_math = sim.run_pure_math_simulation_parallel(total_simulations, processes=processes)

        # Insert "Top 4 Pure Math (%)" before Avg Final Points
        insert_idx = df.columns.get_loc("Avg Final Points")
        df.insert(loc=insert_idx, column="Top 4 Pure Math (%)", value=df["Team"].map(pure_math))

        # Fix index from 1 to 10
        df.index = range(1, len(df) + 1)

        # Apply styled formatting
        styled_df = sim.fancy_highlight_half_split(df.copy())

        # Store results in session
        st.session_state.simulation_df = df
        st.session_state.styled_df = styled_df
        st.session_state.match_number = 70 - len(sim.remaining_matches)
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.session_state["what_if_applied"] = False

if st.sidebar.button("♻️ Reset What-if Results"):
    # Reset all what-if overrides in backend
    new_matches = []
    for match in sim.remaining_matches:
        reset_match = match.copy()
        reset_match["result"] = None
        reset_match["margin"] = None
        reset_match["applied"] = False
        new_matches.append(reset_match)
    sim.set_what_if_results(new_matches)

    # Clear result state
    st.session_state.simulation_df = None
    st.session_state.styled_df = None
    st.session_state.match_number = 70 - len(sim.remaining_matches)
    st.session_state.timestamp = None
    st.success("✅ All What-if results have been cleared.")

# --- Display Results ---
if st.session_state.simulation_df is not None:

    # --- Display Results ---
    if st.session_state.simulation_df is not None:

        if st.session_state.get("what_if_applied"):
            st.subheader("📋 What-if Points Table (After Applied Matches)")
            whatif_table = sim.get_points_table_after_what_if(what_if_matches)
            whatif_table.index = range(1, len(whatif_table) + 1)


            # Apply styling
            def style_points_table(df):
                styled = df.style.format({
                    "NRR": "{:.3f}"
                })

                # Gradient for NRR
                #styled = styled.background_gradient(cmap="Blues", subset=["NRR"])

                # Bold top 4
                styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})

                # Center all cells
                styled = styled.set_properties(**{"text-align": "center"})

                # Color team names
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


            def style_points_table(df):
                styled = df.style.format({
                    "NRR": "{:.3f}"
                })

                # Gradient for NRR
                # styled = styled.background_gradient(cmap="Blues", subset=["NRR"])

                # Bold top 4
                styled = styled.set_properties(subset=pd.IndexSlice[:4, :], **{"font-weight": "bold"})

                # Center all cells
                styled = styled.set_properties(**{"text-align": "center"})

                def color_team_cells(val):
                    if val in sim.team_colors:
                        c = sim.team_colors[val]
                        return f"background-color: {c['bg']}; color: {c['text']}"
                    return ""

                styled = styled.map(color_team_cells, subset=["Team"])
                return styled


            st.markdown(style_points_table(current_table).to_html(escape=False), unsafe_allow_html=True)

    # Show active What-if scenario summary (if any applied)
    applied_matches = [
        match for match in sim.remaining_matches
        if match.get("result") in [match["home"], match["away"]] and match.get("margin") is not None
    ]

    if applied_matches:
        st.markdown("### 🎯 Applied What-if Scenarios")
        data = []
        for match in applied_matches:
            winner = match["result"]
            loser = match["away"] if winner == match["home"] else match["home"]
            margin = f"{match['margin']:+.3f}"
            data.append({
                "Match": f"{match['home']} vs {match['away']}",
                "Winner": winner,
                "Margin": margin
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)



    if st.session_state.get("what_if_applied"):
        applied_count = sum(1 for match in sim.remaining_matches if match.get("applied"))
        simulated_match_number = st.session_state.match_number + applied_count
        st.subheader(f"📝 What-if Simulation Results - Post Match {simulated_match_number:02d}")
    else:
        st.subheader(f"📝 Simulation Results - Post Match {st.session_state.match_number:02d}")

    # Toggle to show/hide advanced metrics
    show_advanced = st.checkbox("Show Avg Final Points & NRR", value=False)

    styled_df = st.session_state.styled_df

    # Reset styles before conditionally hiding columns
    styled_df = styled_df.set_table_styles([])

    if not show_advanced:
        hidden_styles = []
        for col in ["Avg Final Points", "Avg Final NRR"]:
            try:
                col_idx = styled_df.data.columns.get_loc(col)
                hidden_styles.extend([
                    {"selector": f"th.col{col_idx}", "props": [("display", "none")]},
                    {"selector": f"td.col{col_idx}", "props": [("display", "none")]}
                ])
            except KeyError:
                pass
        styled_df = styled_df.set_table_styles(hidden_styles, overwrite=False)

    # Show styled HTML table
    st.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)

    # --- Download Buttons ---
    csv_buffer = io.StringIO()
    st.session_state.simulation_df.to_csv(csv_buffer, index=False)

    # Determine prefix
    prefix = "whatif" if st.session_state.get("what_if_applied") else "post"
    match_id = f"m{simulated_match_number}" if st.session_state.get("what_if_applied") else f"m{st.session_state.match_number}"
    timestamp = st.session_state.timestamp
    csv_filename = f"{prefix}_{match_id}_results_{timestamp}.csv"
    excel_filename = f"{prefix}_{match_id}_stylized_{timestamp}.xlsx"

    # Prepare CSV
    csv_buffer = io.StringIO()
    st.session_state.simulation_df.to_csv(csv_buffer, index=False)

    # Prepare Excel
    excel_buffer = io.BytesIO()
    st.session_state.styled_df.to_excel(excel_buffer, engine='openpyxl', index=False)
    excel_buffer.seek(0)

    # Download buttons
    st.download_button("⬇️ Download CSV", data=csv_buffer.getvalue(), file_name=csv_filename, mime="text/csv")
    st.download_button("⬇️ Download Excel", data=excel_buffer, file_name=excel_filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


else:
    st.info("Click **Run Simulation** from the sidebar to run simulations of already completed matches and **Apply What-If Scenarios** to run what if simulations.")
