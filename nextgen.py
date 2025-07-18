import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(layout='wide', page_icon="💸")
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'>🌳 Will My Money Last Three Generations? 🌳</h1>",
    unsafe_allow_html=True,
)

# --- Main Inputs on Page for Mobile Friendliness ---
with st.expander("💰 Investment Assumptions", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        investment_amount = st.number_input(
            'Initial Investment',
            min_value=0,
            value=30_000_000,
            step=1_000_000,
            format="%d",
            help="The lump sum you are starting with."
        )
    with c2:
        real_return = st.selectbox(
            'Expected Real Return (%)', list(range(0, 9)),
            help="Annual return after inflation."
        ) / 100
    with c3:
        volatility = st.slider(
            'Volatility (%)', 0, 30, 12,
            help="Year-to-year fluctuation in returns."
        ) / 100
    with c4:
        num_simulations = st.number_input(
            'Simulations', min_value=100, value=1000, step=100,
            help="More simulations = more accuracy, but slower."
        )

with st.expander("👨‍👩‍👧‍👦 Family Setup", expanded=True):
    st.markdown("##### 2nd Generation (Children)")
    col1, col2 = st.columns(2)
    with col1:
        num_children = st.number_input(
            'Number of Children', min_value=1, value=2, step=1,
            help="How many children in the 2nd generation?"
        )
    with col2:
        second_gen_ages = []
        for i in range(num_children):
            age = st.number_input(
                f"Age of Child {i+1}",
                min_value=0, max_value=120, value=40,
                help="Current age of this 2nd generation child.",
                key=f"2nd_gen_age_{i}"
            )
            second_gen_ages.append(age)

    st.markdown("##### 3rd Generation (Grandchildren)")
    col3, col4 = st.columns(2)
    with col3:
        num_grandchildren = st.number_input(
            'Number of Grandchildren', min_value=0, value=4, step=1,
            help="Total number of 3rd generation children."
        )
    with col4:
        third_gen_ages = []
        for i in range(num_grandchildren):
            gc_age = st.number_input(
                f"Age of Grandchild {i+1}",
                min_value=0, max_value=120, value=10,
                help="Current age of this 3rd generation child.",
                key=f"3rd_gen_age_{i}"
            )
            third_gen_ages.append(gc_age)

    annual_draw_per_child = st.number_input(
        'Annual Drawdown per Person', min_value=0, value=300_000, step=50_000, format="%d",
        help="How much each descendant withdraws per year."
    )
    life_expectancy = st.number_input(
        'Life Expectancy (years)', min_value=50, max_value=120, value=85,
        help="Used to estimate generational length."
    )

st.markdown("---")

with st.form("run_simulation"):
    st.markdown("### 🎲 Run Your Simulation")
    submitted = st.form_submit_button("Simulate!")
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
            return sum(1 for d in descendants if d["start_age"] + year < life_expectancy)

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
        final_values = portfolio_paths[:, -1]
        avg_years = np.mean(results)

        # Calculate generations
        generations = 0
        if num_children > 0:
            generations += 1  # 2nd gen exists
        if num_grandchildren > 0:
            generations += 1  # 3rd gen exists

        # --- Results Section ---
        st.markdown("## 🏁 Results")
        if np.all(final_values > 0):
            st.success(f"🎉 The fund survived all drawdowns and lasted for all **{generations} generations**!")
            st.metric(label="Average Years Fund Lasts", value=f"{sim_years:,.1f} years")
            st.metric(label="Estimated Generations Covered", value=f"{generations}")
        else:
            st.warning("⚠️ The fund was depleted before all drawdowns ended.")
            st.metric(label="Average Years Fund Lasts", value=f"{avg_years:,.1f} years")
            st.metric(label="Estimated Generations Covered", value=f"{avg_years / sim_years * generations:.1f}")

        median_path = np.median(portfolio_paths, axis=0)
        p10 = np.percentile(portfolio_paths, 10, axis=0)
        p90 = np.percentile(portfolio_paths, 90, axis=0)

        survival_rate = np.mean(final_values > 0) * 100
        median_final = np.median(final_values)
        p10_final = np.percentile(final_values, 10)
        p90_final = np.percentile(final_values, 90)
        worst_final = np.min(final_values)

        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Median Final Portfolio", f"{median_final:,.0f}")
        col2.metric("10th–90th Range", f"{p10_final:,.0f} – {p90_final:,.0f}")
        col3.metric("Survival Rate", f"{survival_rate:.1f}%")
        st.caption(f"🧯 In the worst-case simulation, the fund ended with {worst_final:,.0f}.")

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
            line=dict(color='#2E8B57', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=years, y=p90,
            mode='lines',
            name='90th Percentile',
            line=dict(color='rgba(46,139,87,0.2)', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=years, y=p10,
            mode='lines',
            name='10th Percentile',
            fill='tonexty',
            fillcolor='rgba(46,139,87,0.15)',
            line=dict(color='rgba(46,139,87,0.2)', width=0),
            showlegend=False
        ))

        fig.update_layout(
            xaxis_title='Years',
            yaxis_title='Portfolio Value',
            title='Monte Carlo Simulation: Portfolio Value Over Time',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state['portfolio_paths'] = portfolio_paths
        st.session_state['years'] = years
        st.session_state['num_simulations'] = num_simulations

# --- Download Button ---
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
st.markdown(
    """
    <div style='background-color: #f0f8ff; padding: 1em; border-radius: 10px;'>
    <h3>📘 Methodology & Key Findings</h3>
    <b>Methodology:</b>
    <ul>
    <li>Monte Carlo simulation of a lump sum investment with annual drawdowns per descendant.</li>
    <li>Each descendant draws until they reach life expectancy.</li>
    <li>Returns are sampled from a normal distribution with your chosen mean and volatility.</li>
    <li>Simulation ends when all drawdowns stop or the fund is depleted.</li>
    </ul>
    <b>Key Outputs:</b>
    <ul>
    <li>Average years the fund lasts</li>
    <li>Estimated generations covered</li>
    <li>Median and percentile final portfolio values</li>
    <li>Survival rate (how often the fund lasted the full period)</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)