from collections import defaultdict
from pathlib import Path
import sqlite3
import streamlit as st
import altair as alt
import pulp
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Procurement Scenario Planning")

st.write("The app is designed to solve supply chain scenarios and optimization problems by guiding the business user through in the decision required for an optimal sourcing strategy. "
        "To get started, you will find below in the tables few supplier and demand dummy data that you can either use or modify.")


c1, spacer, c2 = st.columns([1,0.1,1])


with c1:

    st.subheader("Supplier data table")

    with st.container():
        c1.write("The following table is fully editable and it can be either updated or be used as-is. If you want to use your own data, just copy paste from a .csv or .xlsx file with the same column structure and the table will update once you click on commit changes.")

    def connect_db():
        """Connects to the sqlite database."""
        DB_FILENAME = Path(__file__).parent / 'data/supplier_data.db'
        db_already_exists = DB_FILENAME.exists()
        conn = sqlite3.connect(DB_FILENAME)
        db_was_just_created = not db_already_exists
        return conn, db_was_just_created

    # Function to initialize supplier data
    def init_supplier_data(conn):
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS supplier_data (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Supplier TEXT,
                Capacity INTEGER,
                Cost REAL,
                Emissions REAL
            )
            """
        )

        # Load data from the internal CSV file
        supplier_df = pd.read_csv('data/supplier_data.csv')

        # Prepare the data for insertion
        rows_to_insert = supplier_df[['Supplier', 'Capacity', 'Cost', 'Emissions']].values.tolist()

        # Insert the data from CSV into the database
        cursor.executemany(
            """
            INSERT INTO supplier_data (Supplier, Capacity, Cost, Emissions)
            VALUES (?, ?, ?, ?)
            """,
            rows_to_insert
        )

        conn.commit()
        st.toast("Data loaded from CSV and inserted into the database.")

    def load_supplier_data(conn):
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM supplier_data")
            df = cursor.fetchall()
        except:
            return None
        
        supplier_df = pd.DataFrame(df, columns=['ID','Supplier','Capacity','Cost','Emissions'])
        return supplier_df

    def update_supplier_data(conn, supplier_df, changes):
        cursor = conn.cursor()
        if changes["edited_rows"]:
            deltas = st.session_state.supplier_table["edited_rows"]
            rows = []
            for i, delta in deltas.items():
                row_dict = supplier_df.iloc[i].to_dict()
                row_dict.update(delta)
                rows.append(row_dict)

            cursor.executemany(
                """
                UPDATE supplier_data
                SET
                    Supplier = :Supplier,
                    Capacity = :Capacity,
                    Cost = :Cost,
                    Emissions = :Emissions
                WHERE ID = :ID
                """,
                rows,
            )

        if changes["added_rows"]:
            cursor.executemany(
                """
                INSERT INTO supplier_data
                    (Supplier, Capacity, Cost, Emissions)
                VALUES
                    (:Supplier, :Capacity, :Cost, :Emissions)
                """,
                (defaultdict(lambda: None, row) for row in changes["added_rows"]),
            )

        if changes["deleted_rows"]:
            cursor.executemany(
                "DELETE FROM supplier_data WHERE ID = :ID",
                ({"ID": int(supplier_df.loc[i, "ID"])} for i in changes["deleted_rows"]),
            )

        conn.commit()

    def validate_supplier_data(df):
        """Validates the structure of the uploaded supplier data."""
        required_columns = ['Supplier', 'Capacity', 'Cost', 'Emissions']
        return all(col in df.columns for col in required_columns)

    # Connect to database and create table if needed
    conn, db_was_just_created = connect_db()

    # Initialize data.
    if db_was_just_created:
        init_supplier_data(conn)
        st.toast("Database initialized with some sample data.")

    # Load data from database
    supplier_df = load_supplier_data(conn)


    # Display data with editable table
    edited_supplier_df = st.data_editor(
        supplier_df,
        disabled=["ID"],  # Don't allow editing the 'ID' column.
        num_rows="dynamic",  # Allow appending/deleting rows.
        key="supplier_table",
    )

    if 'supplier_table' not in st.session_state:
        st.session_state['supplier_table'] = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}

    has_uncommitted_changes = any(len(v) for v in st.session_state.supplier_table.values())

    st.button(
        "Commit changes",
        type="primary",
        disabled=not has_uncommitted_changes,
        # Update data in database
        on_click=update_supplier_data,
        args=(conn, supplier_df, st.session_state.supplier_table),
        key="supplier_commit_button" 
    )

with c2:
    st.subheader("Supplier comparison")
    st.write("The normalized radar chart overlays and compares the different suppliers present in the table for a first qualitative.")

    # Function to create a combined radar chart with adjusted scaling and transparency
    def create_adjusted_radar_chart(supplier_df):
        # Prepare the radar chart
        suppliers = supplier_df['Supplier'].tolist()  # List of suppliers
        emissions = supplier_df['Emissions'].tolist()  # Emissions values
        costs = supplier_df['Cost'].tolist()  # Cost values
        capacities = supplier_df['Capacity'].tolist()  # Capacity values
        
        # Normalize emissions and costs to a scale of 0 to 1
        max_emissions = max(emissions)
        max_costs = max(costs)
        max_capacity = max(capacities)

        normalized_emissions = [e / max_emissions for e in emissions]
        normalized_costs = [c / max_costs for c in costs]
        adjusted_capacities = [c / max_capacity for c in capacities]  # Normalize for comparison

        # Create a radar chart
        fig = go.Figure()

        # Add traces for each category with increased transparency
        fig.add_trace(go.Scatterpolar(
            r=normalized_emissions + [normalized_emissions[0]],  # Close the loop for emissions
            theta=suppliers + [suppliers[0]],  # Close the loop
            fill='toself',
            name='Emissions',
            line=dict(color='#660066'), 
            fillcolor='rgba(102, 0, 102, 0.1)'  # Set fill color with transparency
        ))

        fig.add_trace(go.Scatterpolar(
            r=normalized_costs + [normalized_costs[0]],  # Close the loop for costs
            theta=suppliers + [suppliers[0]],  # Close the loop
            fill='toself',
            name='Cost',
            line=dict(color='#0A817D'),
            fillcolor='rgba(10, 129, 125, 0.1)'  # Set fill color with transparency
        ))

        fig.add_trace(go.Scatterpolar(
            r=adjusted_capacities + [adjusted_capacities[0]],  # Close the loop for capacities
            theta=suppliers + [suppliers[0]],  # Close the loop
            fill='toself',
            name='Capacity',
            line=dict(color='#87cefa'),  # Lighter blue tone for capacities
            fillcolor='rgba(135, 206, 250, 0.1)'  # Set fill color with transparency
        ))

        # Update layout
        fig.update_layout(
            #title='Emissions, Cost, and Capacity',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]  # All categories will be scaled between 0 and 1
                )
            ),
            showlegend=True
        )

        return fig

    # Assuming supplier_df is available and contains the necessary data
    if supplier_df is not None and not supplier_df.empty:
        # Create and display the adjusted radar chart with transparency
        adjusted_radar_fig = create_adjusted_radar_chart(supplier_df)
        st.plotly_chart(adjusted_radar_fig)



c3, spacer, c4 = st.columns([1,0.1,1])

with c3:
    st.subheader("Demand data table")
    with st.container():
        c3.write("The following table is fully editable and it can be either updated or be used as-is. If you want to use your own data, just copy paste from a .csv or .xlsx file with the same column structure and the table will update once you click on commit changes.")

    def connect_db():
        """Connects to the sqlite database."""
        DB_FILENAME = Path(__file__).parent / 'data/demand_data.db'
        db_already_exists = DB_FILENAME.exists()

        conn = sqlite3.connect(DB_FILENAME)
        db_was_just_created = not db_already_exists

        return conn, db_was_just_created

    # Function to initialize demand data
    def init_demand_data(conn):
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS demand_data (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Period INTEGER,
                Demand REAL
            )
            """
        )

        # Load data from the internal CSV file
        demand_df = pd.read_csv('data/demand_data.csv')

        rows_to_insert = demand_df[['Period','Demand']].values.tolist()

        # Insert the data from CSV into the database
        cursor.executemany(
            """
            INSERT INTO demand_data (Period, Demand)
            VALUES (?, ?)
            """,
            rows_to_insert
        )

        conn.commit()
        st.toast("Data loaded from CSV and inserted into the database.")


    def load_demand_data(conn):
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM demand_data")
            df = cursor.fetchall()
        except:
            return None

        demand_df = pd.DataFrame(df, columns=['ID', 'Period', 'Demand'])

        return demand_df


    def update_demand_data(conn, demand_df, changes):
        """Updates the demand data in the database."""
        cursor = conn.cursor()

        if changes["edited_rows"]:
            deltas = st.session_state.demand_table["edited_rows"]
            rows = []

            for i, delta in deltas.items():
                row_dict = demand_df.iloc[i].to_dict()
                row_dict.update(delta)
                rows.append(row_dict)

            cursor.executemany(
                """
                UPDATE demand_data
                SET
                    Period = :Period,
                    Demand = :Demand
                WHERE ID = :ID
                """,
                rows,
            )

        if changes["added_rows"]:
            cursor.executemany(
                """
                INSERT INTO demand_data (Period, Demand)
                VALUES (:Period, :Demand)
                """,
                (defaultdict(lambda: None, row) for row in changes["added_rows"]),
            )

        if changes["deleted_rows"]:
            cursor.executemany(
                "DELETE FROM demand_data WHERE ID = :ID",
                ({"ID": int(demand_df.loc[i, "ID"])} for i in changes["deleted_rows"]),
            )

        conn.commit()


    # Connect to database and create table if needed
    conn, db_was_just_created = connect_db()

    # Initialize data.
    if db_was_just_created:
        init_demand_data(conn)
        st.toast("Database initialized with some sample data.")

    # Load data from database
    demand_df = load_demand_data(conn)

    # Display data with editable table
    edited_demand_df = st.data_editor(
        demand_df,
        disabled=["ID"],  # Don't allow editing the 'id' column.
        num_rows="dynamic",  # Allow appending/deleting rows.
        key="demand_table",
    )

    if 'demand_table' not in st.session_state:
        st.session_state['demand_table'] = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}


    has_uncommitted_changes = any(len(v) for v in st.session_state.demand_table.values())

    st.button(
        "Commit changes",
        type="primary",
        disabled=not has_uncommitted_changes,
        # Update data in database
        on_click=update_demand_data,
        args=(conn, demand_df, st.session_state.demand_table),
        key="demand_commit_button" 
    )

with c4:
    st.subheader("Demand over time")
    st.write("The demand over time is structured in periods which are editable by the user. While pre-loaded data reflects 12 months period, it can be adjusted to match for exammple 4 weeks or 6 months")

    # Plotting the demand data
    if demand_df is not None and not demand_df.empty:

 
         # Create a line plot using Plotly
        fig = px.line(demand_df, x='Period', y='Demand', markers=True)
        
        # Update layout for better visibility
        fig.update_layout(
            xaxis_title='Period',
            yaxis_title='Demand',
            xaxis=dict(tickmode='linear'),  # Ensure all periods are shown
            template='plotly_white'  # Clean layout
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)


#if st.button("Go to Scenario Planning"):

# Title and Description
st.title("Simulation of Baseline vs Planned Scenario")
st.write(
    "The app leverages data on supplier capacity, cost, emissions, and monthly demand. The app computes the optimal solution "
    "that minimizes total costs while meeting demand and adhering to emissions constraints. Interactive sliders allow sensitivity analysis by adjusting the emissions cap and carbon cost in real-time, providing valuable insights into the trade-offs between costs and emissions."
)

# Function to load supplier data from the database
def load_supplier_data():
    DB_FILENAME = Path(__file__).parent / 'data/supplier_data.db'
    db_already_exists_s = DB_FILENAME.exists()
    conn = sqlite3.connect(DB_FILENAME)
    db_was_just_created_s = not db_already_exists_s

    # Fetch data from the suppliers table
    supplier_df = pd.read_sql_query("SELECT * FROM supplier_data", conn)
    conn.close()  # Close the connection after loading data
    return supplier_df, db_was_just_created_s

# Function to load demand data for a specific month from the database
def load_demand_data():
    DB_FILENAME = Path(__file__).parent / 'data/demand_data.db'
    db_already_exists_d = DB_FILENAME.exists()
    conn = sqlite3.connect(DB_FILENAME)
    db_was_just_created_d = not db_already_exists_d

    # Fetch data from the suppliers table
    supplier_df = pd.read_sql_query("SELECT * FROM demand_data", conn)
    conn.close()  # Close the connection after loading data
    return supplier_df, db_was_just_created_d

# Load supplier data
supplier_df, db_was_just_created_s = load_supplier_data()

#supplier_df = pd.read_csv('data/supplier_data.csv')

capacities = supplier_df['Capacity'].values
costs = supplier_df['Cost'].values
emissions = supplier_df['Emissions'].values
suppliers = supplier_df['Supplier'].values

# Load demand data
demand_df, db_was_just_created_d = load_demand_data()

#demand_df = pd.read_csv('data/demand_data.csv')


# input sliders for sensitivity analysis
st.subheader("Configuration Settings")

col1, spacer1, col2, spacer2, col3 = st.columns([1,0.1,1,0.1,1])

ec_max = int(round(sum([emissions[i] * capacities[i] for i in range(len(suppliers))]),-2))

with col1:
    st.write("Set the emissions cap to limit the total emissions allowed during the sourcing process.")
    emissions_cap = st.slider('Emissions Cap', 0, ec_max , ec_max)
with col2:
    st.write("Define the carbon cost per unit of emissions. This will influence the sourcing strategy.")
    CarbonCost = st.slider('Carbon Cost', 0.0, 5.0, 0.0, step=0.1)
with col3:
    st.write("Select the rolling period for which you want to analyze the sourcing data. ")
    period = st.slider('Period', 1, max(demand_df['Period'].values), 1)

demand = demand_df['Demand'].values[period-1]

def baseline_sourcing(demand, capacities, costs):
    prob = pulp.LpProblem("Supplier_Sourcing_Problem", pulp.LpMinimize)

    # Decision variables for each supplier
    suppliers = len(capacities)
    x = [pulp.LpVariable(f'x{i+1}', lowBound=0, upBound=capacities[i], cat='Integer') for i in range(suppliers)]

    # Objective function: Minimize cost
    total_cost = pulp.lpSum([costs[i] * x[i] for i in range(suppliers)])
    prob += total_cost, "Total_Cost"

    # Constraints
    prob += pulp.lpSum(x) == demand, "Demand_Constraint"

    # Solve the problem
    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        x_vals_bl = [pulp.value(var) for var in x]
        total_cost = sum(costs[i] * x_vals_bl[i] for i in range(suppliers))
    else:
        x_vals_bl = [0] * suppliers  # Return a list of zeros for all suppliers
        total_cost = 0

    return x_vals_bl, total_cost


# Function to optimize sourcing based on emissions cap and CarbonCost
def optimize_sourcing(demand, capacities, costs, emissions, emissions_cap, CarbonCost):
    prob = pulp.LpProblem("Supplier_Sourcing_Problem", pulp.LpMinimize)

    # Decision variables for each supplier
    suppliers = len(capacities)
    x = [pulp.LpVariable(f'x{i+1}', lowBound=0, upBound=capacities[i], cat='Integer') for i in range(suppliers)]

    # Objective function: Minimize cost + emissions penalty
    total_cost = pulp.lpSum([costs[i] * x[i] for i in range(suppliers)]) + CarbonCost * pulp.lpSum([emissions[i] * x[i] for i in range(suppliers)])
    prob += total_cost, "Total_Cost_and_Emissions_Penalty"

    # Constraints
    prob += pulp.lpSum(x) == demand, "Demand_Constraint"
    prob += pulp.lpSum([emissions[i] * x[i] for i in range(suppliers)]) <= emissions_cap, "Emissions_Constraint"

    # Solve the problem
    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        x_vals = [pulp.value(var) for var in x]
        total_emissions = sum(emissions[i] * x_vals[i] for i in range(suppliers))
        total_cost = sum(costs[i] * x_vals[i] for i in range(suppliers)) + CarbonCost * total_emissions
    else:
        x_vals, total_emissions, total_cost = [0]*suppliers, 0, 0

    return x_vals, total_emissions, total_cost


# Calculate baseline values
x_vals_bl, total_cost_bl = baseline_sourcing(demand, capacities, costs)
emissions_vals_bl = [emissions[i] * x_vals_bl[i] for i in range(len(x_vals_bl))]
costs_vals_bl = [(costs[i] + CarbonCost) * x_vals_bl[i] for i in range(len(x_vals_bl))]

# Calculate current scenario values
x_vals_opt, total_emissions_opt, total_cost_opt = optimize_sourcing(demand, capacities, costs, emissions, emissions_cap, CarbonCost)
emissions_vals_opt = [emissions[i] * x_vals_opt[i] for i in range(len(x_vals_opt))]
costs_vals_opt = [(costs[i] + CarbonCost) * x_vals_opt[i] for i in range(len(x_vals_opt))]


table_data = [[suppliers[i], f"{x_vals_bl[i]:.0f}", f"{x_vals_opt[i]:.0f}", f"{emissions_vals_bl[i]:.2f}", f"{emissions_vals_opt[i]:.2f}", f"${costs_vals_bl[i]:.2f}", f"${costs_vals_opt[i]:.2f}"] for i in range(len(x_vals_opt))]
table_data.append(['Total', f"{sum(x_vals_bl):.0f}", f"{sum(x_vals_opt):.0f}", f"{sum(emissions_vals_bl):.2f}", f"{sum(emissions_vals_opt):.2f}", f"${sum(costs_vals_bl):.2f}", f"${sum(costs_vals_opt):.2f}"])
column_labels = ['Supplier', 'Baseline Units Sourced', 'Current Scenario Units Sourced', 'Baseline Emissions','Current Scenario Emissions','Baseline Cost','Current Scenario Cost']


tab1, tab2 = st.tabs(["Table view", "Graphical view"])
tab1.write("""The baseline is a function of the slider values chosen by the user, reflecting the cost-minimizing scenario without considering emissions constraints. 
           The simulated current scenario outcomes including emission cap and cost of carbon are displayed in the table below""")
tab2.write("""The baseline is a function of the slider values chosen by the user, reflecting the cost-minimizing scenario without considering emissions constraints (orange diamonds).
           The simulated current scenario outcomes including emission cap and cost of carbon are plotted respectively in the bar chart and in the pie chart.""")


with tab1:
    # Create DataFrame without displaying the index
    st.dataframe(pd.DataFrame(table_data, columns=column_labels), hide_index=True)

with tab2:

    c5, c6= st.columns(2)
    c7, c8= st.columns(2)


    with st.container():
        c5.write("Emissions, Current Scenario vs Baseline")
        c6.write("Costs, Current Scenario vs Baseline")

    with c5:
            # Create DataFrame for Altair chart
        chart_data = pd.DataFrame({
            'Supplier': supplier_df['Supplier'],
            'Current Scenario Emissions': emissions_vals_opt,
            'Baseline Emissions': emissions_vals_bl
        })

        # Create a "Total" row with summed values
        total_row = pd.DataFrame({
            'Supplier': ['Total'],
            'Current Scenario Emissions': [sum(emissions_vals_opt)],
            'Baseline Emissions': [sum(emissions_vals_bl)]
        })

        # Use pd.concat() to append the "Total" row to the chart_data DataFrame
        chart_data = pd.concat([chart_data, total_row], ignore_index=True)

        # aqua color scheme
        custom_colors = ['#7cf9d6', '#6feeb7', '#7fccb6', '#009f8b','#007667']
        
        # Plot using Altair for Emissions with dynamic color based on values
        st.altair_chart(
            # Layer 1: Bar chart for Current Scenario emissions
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X('Current Scenario Emissions:Q', axis=alt.Axis(title='Emissions CO2e kg')),
                y=alt.Y('Supplier:N', sort='-x', axis=alt.Axis(title='Supplier')),
                color=alt.Color('Current Scenario Emissions:Q', 
                                scale=alt.Scale(domain=[min(emissions_vals_opt), sum(emissions_vals_opt)], 
                                                range=custom_colors),  # Custom color scheme
                                legend=None)  # No legend needed for colors
            )
            # Layer 2: Diamond markers for baseline emissions
            + alt.Chart(chart_data)
            .mark_point(
                shape="diamond",
                filled=True,
                color="orange",
                size=120,
            )
            .encode(
                x=alt.X('Baseline Emissions:Q'),
                y='Supplier:N'
            ),
            use_container_width=True,
        )
    with c6:
        # Create DataFrame for Altair chart
        cost_chart_data = pd.DataFrame({
            'Supplier': supplier_df['Supplier'],
            'Scenario Costs': [costs[i] * x_vals_opt[i] for i in range(len(x_vals_opt))],
            'Baseline Costs': [costs[i] * x_vals_bl[i] for i in range(len(x_vals_bl))]
        })

        # Create a "Total" row with summed values
        total_cost_row = pd.DataFrame({
            'Supplier': ['Total'],
            'Scenario Costs': [sum(costs[i] * x_vals_opt[i] for i in range(len(x_vals_opt)))],
            'Baseline Costs': [sum(costs[i] * x_vals_bl[i] for i in range(len(x_vals_bl)))]
        })

        # Use pd.concat() to append the "Total" row to the cost_chart_data DataFrame
        cost_chart_data = pd.concat([cost_chart_data, total_cost_row], ignore_index=True)

        # Plot using Altair for Costs with dynamic color based on values
        st.altair_chart(
            alt.Chart(cost_chart_data)
            .mark_bar()
            .encode(
                x=alt.X('Scenario Costs:Q', axis=alt.Axis(title='Costs $')),
                y=alt.Y('Supplier:N', sort='-x', axis=alt.Axis(title='Supplier')),
                color=alt.Color('Scenario Costs:Q', 
                                scale=alt.Scale(domain=[min(costs_vals_opt), sum(costs_vals_opt)], 
                                                range=custom_colors),
                                legend=None)
            )
            + alt.Chart(cost_chart_data)
            .mark_point(
                shape="diamond",
                filled=True,
                color="orange",
                size=120,
            )
            .encode(
                x=alt.X('Baseline Costs:Q', axis=alt.Axis(title='')),
                y='Supplier:N'
            ),
            use_container_width=True,
        )



    with st.container():
        c7.write("Emissions percentage-wise by supplier")
        c8.write("Costs percentage-wise by supplier")

    with c7:

        total_emissions_opt = sum(emissions_vals_opt)

        # Calculate percentages for the pie chart
        percentages = [round(val / total_emissions_opt * 100, 2) if total_emissions_opt > 0 else 0 for val in emissions_vals_opt]

        # Create a DataFrame with the data for the pie chart
        pie_data = pd.DataFrame({
            'Supplier': suppliers,
            'Emissions': emissions_vals_opt,
            'Percentage': percentages
        })

        f_pie_data = pie_data[pie_data['Emissions'] > 0]

        # Create the pie chart using Plotly
        fig = px.pie(f_pie_data, names='Supplier', values='Emissions', color_discrete_sequence=custom_colors)

        # Set hover template to display the rounded percentage
        fig.update_traces(hovertemplate='%{label}: %{value:.2f} emissions (%{percent:.2%})')

        fig.update_layout(
            width=300,  # Adjust the width
            height=300,  # Adjust the height
            legend=dict(
                x=-0.4,  # Move the legend to the left
                y=0.5,   # Center it vertically
        ))

        # Display the pie chart in Streamlit
        st.plotly_chart(fig)

    with c8:

        total_costs_opt = sum(costs_vals_opt)

        # Calculate percentages for the pie chart
        percentages = [round(val / total_costs_opt * 100, 2) if total_cost_opt > 0 else 0 for val in costs_vals_opt]

        # Create a DataFrame with the data for the pie chart
        pie_data = pd.DataFrame({
            'Supplier': suppliers,
            'Costs': costs_vals_opt,
            'Percentage': percentages
        })

        f_pie_data = pie_data[pie_data['Costs'] > 0]

        # Create the pie chart using Plotly
        fig = px.pie(f_pie_data, names='Supplier', values='Costs', color_discrete_sequence=custom_colors)

        # Set hover template to display the rounded percentage
        fig.update_traces(hovertemplate='%{label}: %{value:.2f} costs (%{percent:.2%})')

        fig.update_layout(
            width=300,  # Adjust the width
            height=300,  # Adjust the height
            legend=dict(
                x=-0.4,  # Move the legend to the left
                y=0.5,   # Center it vertically
        ))

        # Display the pie chart in Streamlit
        st.plotly_chart(fig)
            
    #with st.container():
    #    c5.write("Emission vs Cost")
    #    c6.write("Units sourced percentage-wise by supplier")


    st.markdown("""
        *For a guided tour and more information, please contact Alessandro Silvestro*
    """)
