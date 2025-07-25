import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

im = Image.open("favicon.ico")

# ---- Page Configuration ----
st.set_page_config(
    page_title="India's Gender Gap Dashboard",
    page_icon=im,
    layout="wide"
)

# ---- Data ----
# Hardcoded global data from the provided table for the year 2025
global_data = {
    'Final_Score': {'india_score': 0.644, 'india_rank': 131, 'top_score': 0.935, 'top_country': 'Iceland', 'lowest_score': 0.568, 'lowest_country': 'Sudan'},
    'Economic_Score': {'india_score': 0.407, 'india_rank': 144, 'top_score': 0.873, 'top_country': 'Botswana', 'lowest_score': 0.313, 'lowest_country': 'Sudan'},
    'Education_Score': {'india_score': 0.971, 'india_rank': 110, 'top_score': 1.0, 'top_country': '35 Countries', 'lowest_score': 0.649, 'lowest_country': 'DRC'},
    'Health_Score': {'india_score': 0.954, 'india_rank': 143, 'top_score': 0.98, 'top_country': '17 Countries', 'lowest_score': 0.934, 'lowest_country': 'Azerbaijan'},
    'Political_Score': {'india_score': 0.245, 'india_rank': 69, 'top_score': 0.954, 'top_country': 'Iceland', 'lowest_score': 0.006, 'lowest_country': 'Vanuatu'},
    'LFPR_FM': {'india_score': 0.459, 'india_rank': 136, 'top_score': 0.991, 'top_country': 'Burundi', 'lowest_score': 0.201, 'lowest_country': 'Iran'},
    'Wage_FM': {'india_score': 0.541, 'india_rank': 117, 'top_score': 0.931, 'top_country': 'Albania', 'lowest_score': 0.459, 'lowest_country': 'Chad'},
    # Add other indicators here if you have their global data
}


# ---- Data Loading and Processing ----
def load_and_process_data(path, domains_dict):
    """Loads, preprocesses, calculates domain scores, and ranks all data."""
    main_df = pd.read_csv(path, skipinitialspace=True)
    main_df.columns = main_df.columns.str.strip()
    rename_map = {
        "States/UTs": "State", "Ratio of female to male Labour Force Participation Rate (LFPR) (15-59 years)": "LFPR_FM", "Average Wage Salary Earnings F/M": "Wage_FM", "Women/Men ratio in managerial positions": "Managerial_FM", "Ratio of female workers to male workers working as Professionals and Technical Workers": "ProfTech_FM", "Female to Male Ratio of Literacy Rate of 7 and above (age)": "Literacy_FM", "Female to Male Ratio of Enrolment in primary education": "EnrollPrimary_FM", "Female to Male Ratio of Enrolment in secondary education": "EnrollSecondary_FM", "Female to Male Ratio of Enrolment in higher education": "EnrollTertiary_FM", "Sex ratio at birth (F/M)": "SexBirth_FM", "Gender Ratio (Life Expectancy)": "LifeExpectancy_FM", "F/M Ratio of seats held in Panchayat Raj Institutions (PRIs)": "PRISeats_FM", "F/M Lok Sabha": "LokSabha_FM", "Gender Ratio (Days of CM)": "DaysCM_FM", "Final Score": "Final_Score"
    }
    main_df.rename(columns=rename_map, inplace=True)

    df_states = main_df[main_df["State"] != "Average"].copy()
    # Separate the 'Average' row to use its values later
    df_average = main_df[main_df["State"] == "Average"].copy()

    # Calculate scores for both dataframes
    for df_to_process in [df_states, df_average]:
        for domain, indicators in domains_dict.items():
            df_to_process[f'{domain}_Score'] = df_to_process[indicators].mean(axis=1)

    score_columns = ['Final_Score'] + [f'{d}_Score' for d in domains_dict]
    all_indicators = [ind for inds in domains_dict.values() for ind in inds]

    # Calculate ranks only for the states dataframe
    for col in score_columns + all_indicators:
        df_states[f'{col}_Rank'] = df_states[col].rank(method='min', ascending=False).astype(int)

    return df_states, df_average

# --- Dictionaries & Lists ---
domain_to_indicators = {
    "Economic": ["LFPR_FM", "Wage_FM", "Managerial_FM", "ProfTech_FM"],
    "Education": ["Literacy_FM", "EnrollPrimary_FM", "EnrollSecondary_FM", "EnrollTertiary_FM"],
    "Health": ["SexBirth_FM", "LifeExpectancy_FM"],
    "Political": ["PRISeats_FM", "LokSabha_FM", "DaysCM_FM"]
}

# --- Load Data ---
df, df_avg = load_and_process_data("gggi_states.csv", domain_to_indicators)
all_states = sorted(df["State"].unique())

# --- UI Layout ---
st.title("🇮🇳 India Gender Gap Index Dashboard")

tab1, tab2 = st.tabs(["📊 Dashboard", "📖 Methodology"])

# --- DASHBOARD TAB ---
with tab1:
    # ---- Sidebar ----
    st.sidebar.title("Dashboard Controls ⚙️")
    view_mode = st.sidebar.selectbox("Select View", ["National Analysis", "State Analysis", "State Comparison"])

    # --- Main Panel (Dashboard) ---
    if view_mode == "National Analysis":
        st.sidebar.header("National View Options")
        domain_options = ["Overall"] + list(domain_to_indicators.keys())
        selected_domain_nat = st.sidebar.selectbox("Select Domain", domain_options, key="national_domain")

        selected_indicator_nat = None
        if selected_domain_nat != "Overall":
            indicator_options = ["Domain Overall"] + domain_to_indicators[selected_domain_nat]
            selected_indicator_nat = st.sidebar.selectbox("Select Indicator", indicator_options, key="national_indicator")

        sort_order = st.sidebar.radio("Sort states by", ["Rank", "A-Z"], horizontal=True, key="national_sort")

        # --- NEW: Global Perspective Section ---
        st.header("Global & National Perspective")

        # Determine the metric key for fetching data based on selection
        metric_key = "Final_Score"
        display_name = "Overall Score"
        if selected_indicator_nat and selected_indicator_nat != "Domain Overall":
            metric_key = selected_indicator_nat
            display_name = selected_indicator_nat
        elif selected_domain_nat != "Overall":
            metric_key = f"{selected_domain_nat}_Score"
            display_name = f"{selected_domain_nat} Domain Score"

        st.markdown(f"Comparing performance for **{display_name.replace('_', ' ')}**")

        nat_avg_score = df_avg.iloc[0][metric_key] if not df_avg.empty and metric_key in df_avg.columns else 0
        global_info = global_data.get(metric_key)

        cols = st.columns(3)
        with cols[0]:
            st.metric("National Average (from this dataset)", f"{nat_avg_score:.3f}")

        if global_info:
            with cols[1]:
                st.metric("India's Global Score (2025)", f"{global_info['india_score']:.3f}", help=f"Global Rank: {global_info['india_rank']}")
            with cols[2]:
                st.metric(f"Global Top Score (2025)", f"{global_info['top_score']:.3f}", help=f"Top Country: {global_info['top_country']}")
        else:
            cols[1].info("No direct global comparison data for this specific indicator.")

        st.markdown("---")
        st.header("National Rankings")

        # --- Simplified Ranking Display (Map is Removed) ---
        if selected_indicator_nat and selected_indicator_nat != "Domain Overall":
            st.subheader(f"State Rankings for Indicator: {selected_indicator_nat}")
            rank_df = df.sort_values(by=selected_indicator_nat if sort_order == "Rank" else "State", ascending=(sort_order == "A-Z"))
            display_df = rank_df[['State', selected_indicator_nat, f'{selected_indicator_nat}_Rank']].rename(columns={f'{selected_indicator_nat}_Rank': 'Rank'})
            st.dataframe(display_df, use_container_width=True)

        elif selected_domain_nat != "Overall":
            score_col = f'{selected_domain_nat}_Score'
            st.subheader(f"State Rankings for {selected_domain_nat} Domain")
            rank_df = df.sort_values(by=score_col if sort_order == "Rank" else "State", ascending=(sort_order == "A-Z"))
            fig_bar = px.bar(rank_df, x='State', y=score_col, title=f"State Rankings: {selected_domain_nat} Score")
            st.plotly_chart(fig_bar, use_container_width=True)

        else: # Default Overall view
            st.subheader("Overall State Rankings (Final Score)")
            rank_df = df.sort_values(by="Final_Score" if sort_order == "Rank" else "State", ascending=(sort_order == "A-Z"))
            fig_bar = px.bar(rank_df, x='State', y='Final_Score', title="State Rankings: Final Score")
            st.plotly_chart(fig_bar, use_container_width=True)

    elif view_mode == "State Analysis":
        st.sidebar.header("State View Options")
        selected_state = st.sidebar.selectbox("Select State/UT", all_states, index=all_states.index("Delhi"))
        state_data = df[df['State'] == selected_state].iloc[0]
        st.header(f"Deep-Dive Analysis for: {selected_state}")
        st.markdown("---")
        st.subheader("Performance Scores & National Ranks")
        total_states = len(all_states)
        cols = st.columns(5)
        score_metrics = { "Overall Score": "Final_Score", "Economic Score": "Economic_Score", "Education Score": "Education_Score", "Health Score": "Health_Score", "Political Score": "Political_Score" }
        for i, (label, col_name) in enumerate(score_metrics.items()):
            cols[i].metric(
                label=label, value=f"{state_data[col_name]:.3f}",
                help=f"The national rank for this score is {state_data[f'{col_name}_Rank']} out of {total_states} states."
            )
            cols[i].markdown(f"**Rank: `{state_data[f'{col_name}_Rank']} / {total_states}`**")
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("All-Indicator Radar Profile")
            all_inds = [ind for inds in domain_to_indicators.values() for ind in inds]
            values = state_data[all_inds].values.flatten().tolist()
            fig_radar = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=all_inds + [all_inds[0]], fill='toself', name=selected_state))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])), showlegend=False, margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_radar, use_container_width=True)
        with col2:
            st.subheader("Indicator-Level Breakdown")
            indicator_details = []
            for domain, indicators in domain_to_indicators.items():
                for indicator in indicators:
                    indicator_details.append({"Domain": domain, "Indicator": indicator, "Value": f"{state_data[indicator]:.3f}", "Rank": f"{state_data[f'{indicator}_Rank']} / {total_states}"})
            breakdown_df = pd.DataFrame(indicator_details)
            st.dataframe(breakdown_df, height=450, use_container_width=True)

    elif view_mode == "State Comparison":
        st.sidebar.header("Comparison Options")
        selected_states = st.sidebar.multiselect("Select States to Compare", all_states, default=["Delhi", "Kerala", "Uttar Pradesh"])
        domain_options_comp = ["Overall Score"] + list(domain_to_indicators.keys())
        selected_domain_comp = st.sidebar.selectbox("Select Comparison Level", domain_options_comp, key="comp_domain")
        selected_indicator_comp = None
        if selected_domain_comp != "Overall Score":
            indicator_options_comp = ["Domain Overall"] + domain_to_indicators[selected_domain_comp]
            selected_indicator_comp = st.sidebar.selectbox("Select Indicator", indicator_options_comp, key="comp_indicator")
        st.header(f"Comparing: {', '.join(selected_states)}")
        if not selected_states:
            st.warning("Please select at least two states to compare.")
        else:
            comp_df = df[df['State'].isin(selected_states)]
            title = ""; y_col = ""
            if selected_indicator_comp and selected_indicator_comp != "Domain Overall":
                title = f"Comparison for Indicator: {selected_indicator_comp}"; y_col = selected_indicator_comp
            elif selected_domain_comp != "Overall Score":
                title = f"Comparison for Domain: {selected_domain_comp}"; y_col = f"{selected_domain_comp}_Score"
            else:
                title = "Comparison of Overall Final Scores"; y_col = "Final_Score"
            st.subheader(title)
            fig_comp_bar = px.bar(comp_df, x='State', y=y_col, color='State', title=title)
            st.plotly_chart(fig_comp_bar, use_container_width=True)
            st.markdown("---")
            st.subheader("All-Indicator Radar Comparison")
            fig_comp_radar = go.Figure()
            all_inds_comp = [ind for inds in domain_to_indicators.values() for ind in inds]
            labels_comp = all_inds_comp + [all_inds_comp[0]]
            for state in selected_states:
                row = df[df["State"] == state].iloc[0]
                values = [row[ind] for ind in all_inds_comp]
                values += [values[0]]
                fig_comp_radar.add_trace(go.Scatterpolar(r=values, theta=labels_comp, fill='toself', name=state))
            fig_comp_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])), showlegend=True, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_comp_radar, use_container_width=True)

# --- METHODOLOGY TAB ---
with tab2:
    st.header("Methodology")
    st.markdown("---")
    st.subheader("Indicator Mapping and Data Sources")
    with st.expander("Labour-force participation rate [%]"):
        st.markdown("- **Indicator Used:** Ratio of female to male Labour Force Participation Rate (LFPR) (15-59 years)\n- **Data Source:** NII 2023-2024.")
    with st.expander("Wage equality for similar work & Estimated earned income"):
        st.markdown("- **Indicator Used:** Average Wage/Salary Earnings\n- **Data Source:** India Stat.\n- **Notes:** We took one year data (four quarters: Jul-Sep 2023; Oct-Dec 2023; Jan-Mar 2024; Apr-Jun 2024) to consider variations from seasonality. The two GGGI indicators were clubbed into one.")
    with st.expander("Legislators, senior officials and managers [%]"):
        st.markdown("- **Indicator Used:** Women/Men ratio in managerial positions\n- **Data Source:** NII 2023-2024.\n- **Notes:** Original data was Women(W) per Thousand Men, converted via W/(1000-W). For missing states, the average of others was used. Covers only managers.")
    with st.expander("Professional and technical workers [%]"):
        st.markdown("- **Indicator Used:** Ratio of female workers to male workers working as Professionals and Technical Workers\n- **Data Source:** NII 2023-2024.\n- **Notes:** Value in percent was divided by 100.")
    with st.expander("Literacy rate [%]"):
        st.markdown("- **Indicator Used:** Female to Male Ratio of Literacy Rate of 7 and above (age)\n- **Data Source:** India Stat (July 2023 to June 2024 Data).\n- **Notes:** We chose the '7 and above' age category.")
    with st.expander("Enrolment in primary education [%]"):
        st.markdown("- **Indicator Used:** Female to Male Ratio of Enrolment in primary education\n- **Data Source:** UDISE Plus (2021-22).\n- **Notes:** The average F/M ratio of Pre-Primary, Primary, and Upper Primary was computed.")
    with st.expander("Enrolment in secondary education [%]"):
        st.markdown("- **Indicator Used:** Female to Male Ratio of Enrolment in secondary education\n- **Data Source:** UDISE Plus (2021-22).")
    with st.expander("Enrolment in tertiary education [%]"):
        st.markdown("- **Indicator Used:** Female to Male Ratio of Enrolment in higher education\n- **Data Source:** UDISE Plus (2021-22).")
    with st.expander("Sex ratio at birth [%]"):
        st.markdown("- **Indicator Used:** Sex Ratio at Birth (F/M)\n- **Data Source:** NII 2023-2024.\n- **Notes:** Original value (Females per 1000 males) was divided by 1000.")
    with st.expander("Healthy life expectancy [years]"):
        st.markdown("- **Indicator Used:** Gender Ratio (Life Expectancy)\n- **Data Source:** SRS Based Abridged Life Tables 2018-22.\n- **Notes:** Life Expectancy was used as a proxy. For missing states, the average was used.")
    with st.expander("Women in parliament [%]"):
        st.markdown("- **Indicator Used:** F/M Ratio of seats held in Panchayat Raj Institutions (PRIs)\n- **Data Source:** NII 2023-2024.\n- **Notes:** Original value (% of seats held by women) was converted to a F/M ratio.")
    with st.expander("Women in ministerial positions [%]"):
        st.markdown("- **Indicator Used:** F/M Ratio of seats held in Lok Sabha\n- **Data Source:** Sansad.in (Downloaded 20 July 2025).\n- **Notes:** For each state, the ratio of female to male members was computed.")
    with st.expander("Years with a female head of state (last 50 years)"):
        st.markdown("- **Indicator Used:** Gender Ratio (Days of CM)\n- **Notes:** Calculated as Days(female CM) / Days(male CM).")

    st.markdown("---")
    st.subheader("Index Calculation")
    st.markdown("The GGGI uses the following weights for its indicators. In our index, we have clubbed indicators for 'Wage equality' and 'Estimated earned income' into one, summing their weights.")
    weights_data = {'Subindex': ['Economic Participation and Opportunity', '', '', '', '', 'Educational Attainment', '', '', '', 'Health and Survival', '', 'Political Empowerment', '', ''], 'Indicator': ['Labour-force participation rate', 'Wage equality for similar work (survey)', 'Estimated earned income', 'Legislators, senior officials and managers', 'Professional and technical workers', 'Literacy rate', 'Enrolment in primary education', 'Enrolment in secondary education', 'Enrolment in tertiary education', 'Sex ratio at birth', 'Healthy life expectancy', 'Women in parliament', 'Women in ministerial positions', 'Years with a female head of state (last 50 years)'], 'Weight': [0.199, 0.310, 0.221, 0.149, 0.121, 0.191, 0.459, 0.230, 0.121, 0.693, 0.307, 0.310, 0.247, 0.443]}
    weights_df = pd.DataFrame(weights_data)
    st.table(weights_df)