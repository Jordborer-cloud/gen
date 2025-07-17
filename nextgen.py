import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.set_page_config(layout='wide')
st.title('Will My Money Last Three Generations?')

col1, col2 = st.columns([1, 2])

with col1:
    st.header('Investment Assumptions')
    investment_amount = st.number_input(
        'Current Investment Amount (ZAR)',
        min_value=0,
        value=30_000_000,
        step=1_000_000,
        format="%d"
    )
    real_return = st.selectbox('Expected Real Return per Annum (%)', [1, 2, 3, 4, 5]) / 100
    volatility = st.slider('Volatility (%)', 0, 30, 12) / 100
    num_simulations = st.number_input('Number of Simulations', min_value=100, value=1000, step=100)

    st.header('Family Assumptions')
    num_children = st.number_input('Number of 2nd Gen Children', min_value=1, value=2, step=1)

    child_grandchildren = []
    for i in range(num_children):
        grandchildren = st.number_input(f'Number of Children for 2nd Gen Child {i+1}', min_value=0, value=2, step=1)
        child_grandchildren.append(grandchildren)

    annual_draw_per_child = st.number_input('Annual Drawdown per Child (ZAR)', min_value=0, value=300_000, step=50_000, format="%d")
    life_expectancy = st.number_input('Life Expectancy (years)', min_value=50, max_value=120, value=85)

with col2:
    st.header("Simulation Results")

    total_years = int(life_expectancy - 30)
    num_total_descendants = num_children + sum(child_grandchildren)
    yearly_draw = annual_draw_per_child * num_total_descendants

    results = []
    for _ in range(num_simulations):
        value = investment_amount
        for year in range(total_years):
            rand_return = np.random.normal(real_return, volatility)
            value = value * (1 + rand_return) - yearly_draw
            if value <= 0:
                results.append(year)
                break
        else:
            results.append(total_years)

    avg_years = np.mean(results)
    gen_duration = int(life_expectancy - 30)
    gen_count = avg_years // gen_duration

    st.metric(label="Average Years Fund Lasts", value=f"{avg_years:.1f} years")
    st.metric(label="Estimated Generations Covered", value=f"{int(gen_count)}")

    sim_years = total_years
    portfolio_paths = []

    for _ in range(num_simulations):
        path = []
        value = investment_amount
        for year in range(sim_years):
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

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in np.random.choice(num_simulations, size=min(10, num_simulations), replace=False):
        ax.plot(portfolio_paths[i], color='lightgray', linewidth=0.8, alpha=0.7)

    ax.plot(median_path, color='blue', label='Median Portfolio Value', linewidth=2)
    ax.fill_between(range(sim_years), p10, p90, color='blue', alpha=0.2, label='10th–90th Percentile Range')

    ax.set_title('Monte Carlo Simulation: Portfolio Value Over Time')
    ax.set_xlabel('Years')
    ax.set_ylabel('Portfolio Value (ZAR)')
    ax.legend()
    ax.ticklabel_format(style='plain', axis='y')

    st.pyplot(fig)

st.markdown("---")
st.subheader("📘 Methodology & Key Findings")
st.markdown("""
### Methodology

This simulation uses a Monte Carlo approach to estimate how long a lump sum investment can sustain annual drawdowns across multiple generations.

- The model starts with a defined **initial investment amount**, which is subject to **annual real returns** drawn from a normal distribution based on the specified mean return and volatility.
- Each year, a **fixed drawdown per descendant** is subtracted from the portfolio.
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
