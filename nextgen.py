import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(layout='wide')
st.title('Will My Money Last Three Generations?')

with st.sidebar:
    with st.expander("💰 Investment Assumptions", expanded=True):
        investment_amount = st.number_input(
            'Current Investment Amount (ZAR)',
            min_value=0,
            value=30_000_000,
            step=1_000_000,
            format="%d",
            help="The lump sum you are starting with."
        )
        real_return = st.selectbox(
            'Expected Real Return per Annum (%)', list(range(0, 9)),
            help="Annual return after inflation."
        ) / 100
        volatility = st.slider(
            'Volatility (%)', 0, 30, 12,
            help="Year-to-year fluctuation in returns."
        ) / 100
        num_simulations = st.number_input(
            'Number of Simulations', min_value=100, value=1000, step=100,
            help="More simulations = more accuracy, but slower."
        )

    with st.expander("👨‍👩‍👧‍👦 Family Assumptions", expanded=True):
        num_children = st.number_input(
            'Number of 2nd Gen Children', min_value=1, value=2, step=1,
            help="How many children in the 2nd generation?"
        )
        second_gen_ages = []
        for i in range(num_children):
            age = st.number_input(
                f"Starting Age for 2nd Gen Child {i+1}",
                min_value=0, max_value=120, value=40,
                help="Current age of this 2nd generation child."
            )
            second_gen_ages.append(age)

        num_grandchildren = st.number_input(
            'Number of 3rd Gen Children (total)', min_value=0, value=4, step=1,
            help="Total number of 3rd generation children (grandchildren)."
        )
        third_gen_ages = []
        for i in range(num_grandchildren):
            gc_age = st.number_input(
                f"Starting Age for 3rd Gen Child {i+1}",
                min_value=0, max_value=120, value=10,
                help="Current age of this 3rd generation child."
            )
            third_gen_ages.append(gc_age)

        annual_draw_per_child = st.number_input(
            'Annual Drawdown per Child (ZAR)', min_value=0, value=300_000, step=50_000, format="%d",
            help="How much each descendant withdraws per year."
        )
        life_expectancy = st.number_input(
            'Life Expectancy (years)', min_value=50, max_value=120, value=85,
            help="Used to estimate generational length."
        )

with st.form("run_simulation"):
    submitted = st.form_submit_button("Run Simulation")
    if submitted:
        # Build a list of all descendants with their starting ages
        descendants = []
        # 2nd gen
        for i in range(num_children):
            descendants.append({
                "gen": 2,
                "start_age": second_gen_ages[i]
            })
        # 3rd gen
        for i in range(num_grandchildren):
            descendants.append({
                "gen": 3,
                "start_age": third_gen_ages[i]
            })

        # Determine simulation years: until the youngest descendant reaches life expectancy
        min_start_age = min([d["start_age"] for d in descendants]) if descendants else 0
        sim_years = int(max(life_expectancy - min_start_age, 1))

        # For each year, calculate how many descendants are still alive (age < life_expectancy)
        def get_active_draws(year):
            count = 0
            for d in descendants:
                if d["start_age"] + year < life_expectancy:
                    count += 1
            return count

        results = []
        for _ in range(num_simulations):
            value = investment_amount
            for year in range(sim_years):
                active_draws = get_active_draws(year)
                yearly_draw = annual_draw_per_child * active_draws
                rand_return = np.random.normal(real_return, volatility)
                value = value * (1 + rand_return) - yearly_draw
                if value <= 0:
                    results.append(year)
                    break
            else:
                results.append(sim_years)

        avg_years = np.mean(results)
        gen_duration = int(life_expectancy - 30)
        gen_count = avg_years // gen_duration

        st.metric(label="Average Years Fund Lasts", value=f"{avg_years:,.1f} years")
        st.metric(label="Estimated Generations Covered", value=f"{int(gen_count):,}")

        # Portfolio paths for plotting and download
        portfolio_paths = []
        for _ in range(num_simulations):
            path = []
            value = investment_amount
            for year in range(sim_years):
                active_draws = get_active_draws(year)
                yearly_draw = annual_draw_per_child * active_draws
                rand_return = np.random.normal(real_return, volatility)
                value = value * (1 + rand_return) - yearly_draw
                value = max(value, 0)
                path.append(value)
            portfolio_paths.append(path)

        portfolio_paths = np.array(portfolio_paths)
        median_path = np.median(portfolio_paths, axis=0)
        p10 = np.percentile(portfolio_paths, 10, axis=0)
        p90 = np.percentile(portfolio_paths, 90, axis=0)

        final_values = portfolio_paths[:, -1]
        survival_rate = np.mean(final_values > 0) * 100
        median_final = np.median(final_values)
        p10_final = np.percentile(final_values, 10)
        p90_final = np.percentile(final_values, 90)
        worst_final = np.min(final_values)

        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)

        col1.metric("Median Final Portfolio", f"R {median_final:,.0f}")
        col2.metric("10th–90th Range", f"R {p10_final:,.0f} – R {p90_final:,.0f}")
        col3.metric("Survival Rate", f"{survival_rate:.1f}%")

        st.caption(f"🧯 In the worst-case simulation, the fund ended with R {worst_final:,.0f}.")

        # --- Interactive Plotly Graph ---
        st.subheader("📈 Interactive Portfolio Value Simulation")
        years = list(range(1, sim_years + 1))
        fig = go.Figure()

        # Show a sample of simulation paths
        for i in np.random.choice(num_simulations, size=min(10, num_simulations), replace=False):
            fig.add_trace(go.Scatter(
                x=years, y=portfolio_paths[i],
                mode='lines',
                line=dict(color='lightgray', width=1),
                opacity=0.6,
                showlegend=False
            ))

        # Median and percentile bands
        fig.add_trace(go.Scatter(
            x=years, y=median_path,
            mode='lines',
            name='Median Portfolio Value',
            line=dict(color='blue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=years, y=p90,
            mode='lines',
            name='90th Percentile',
            line=dict(color='rgba(0,0,255,0.2)', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=years, y=p10,
            mode='lines',
            name='10th Percentile',
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.15)',
            line=dict(color='rgba(0,0,255,0.2)', width=0),
            showlegend=False
        ))

        fig.update_layout(
            xaxis_title='Years',
            yaxis_title='Portfolio Value (ZAR)',
            title='Monte Carlo Simulation: Portfolio Value Over Time',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state['portfolio_paths'] = portfolio_paths
        st.session_state['years'] = years
        st.session_state['num_simulations'] = num_simulations

# Place this outside the form, after the form block
if 'portfolio_paths' in st.session_state and 'years' in st.session_state:
    st.subheader("⬇️ Download Simulation Data")
    df = pd.DataFrame(
        st.session_state['portfolio_paths'].T,
        columns=[f"Simulation {i+1}" for i in range(st.session_state['num_simulations'])]
    )
    df.insert(0, "Year", st.session_state['years'])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Portfolio Paths as CSV",
        data=csv,
        file_name='portfolio_simulation_results.csv',
        mime='text/csv'
    )

st.markdown("---")
st.subheader("📘 Methodology & Key Findings")
st.markdown("""
### Methodology

This simulation uses a Monte Carlo approach to estimate how long a lump sum investment can sustain annual drawdowns across multiple generations.

- The model starts with a defined **initial investment amount**, which is subject to **annual real returns** drawn from a normal distribution based on the specified mean return and volatility.
- Each year, a **fixed drawdown per descendant** is subtracted from the portfolio, but only for those descendants who are alive (age < life expectancy).
- The simulation runs for the duration of one generational cycle (calculated as life expectancy minus 30 years).
- The process is repeated across **multiple simulations** to capture a range of possible outcomes.
- All drawdowns begin immediately and remain constant unless modified. Birth years are not staggered in this model.

### Key Findings

The key outputs of the model include:

- The **average number of years** the fund lasts.
- The **estimated number of generations** it supports.
- The **median final portfolio value**, along with the **10th-90th percentile range**.
- A **survival rate**, indicating how often the portfolio lasted the full simulation period.
""")