import streamlit as st
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from langchain.schema import AgentFinish
from langchain.schema.output import LLMResult
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
import base64
import pickle
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ydata_profiling import ProfileReport
import plotly.express as px
import io
from PIL import Image
from pandasai import SmartDatalake
from pandasai.llm import OpenAI
from lida import Manager, llm, TextGenerationConfig
from openai import OpenAI


st.markdown(
    """
    <style>
        /* Set the entire sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #e0f2e9; /* Light green color for the full sidebar background */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up your database connection and other configurations
user_api_key = ''

os.environ["OPENAI_API_KEY"] = user_api_key
os.environ["PANDASAI_API_KEY"] = ""

# Initialize the OpenAI client
client = OpenAI()

lida = Manager(text_gen = llm("openai"))

# @st.cache_resource
def get_db_connection():
    db_host = "localhost"
    db_name = "sam1"
    db_user = "root"
    db_password = "cronlabs"
    db_port = "3306"
    db_url = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    
    db = SQLDatabase.from_uri(db_url)
    return db

@dataclass
class Event:
    event: str
    timestamp: str
    text: str

def _current_time() -> str:
    return datetime.now(timezone.utc).isoformat()

class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_path: Path):
        self.log_path = log_path
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        assert len(prompts) == 1
        event = Event(event="llm_start", timestamp=_current_time(), text=prompts[0])
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        generation = response.generations[-1][-1].message.content
        event = Event(event="llm_end", timestamp=_current_time(), text=generation)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

@tool("list_tables")
# @st.cache_data
def list_tables() -> str:
    """List the available tables in the database""" 
    db = get_db_connection()
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
# @st.cache_data
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3, table4
    """
    db = get_db_connection()
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    db = get_db_connection()
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    db = get_db_connection()
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

sql_dev = Agent(
    role="Precision SQL Analyst",
    goal="Construct and execute highly accurate SQL queries, with a focus on flexible pattern matching for product models",
    backstory=
    """
    You are an expert SQL Analyst known for your precision and adaptability, especially when dealing with product data across various industries. Your specialty lies in constructing queries that accurately capture all relevant data, particularly when dealing with product models that may have variations in naming conventions.

    Core Competencies:
    1. Accurate Market Share Calculations: For market share comparison, dynamically filter by specified regions, timeframes, and products; aggregate sellout_value or amount for each, calculate each share as a percentage of the combined total.
    2. Expert Use of LIKE Operator: Excel in using LIKE with wildcards for comprehensive and accurate results, crucial for handling variations in model names.
    3. Rapid Schema Analysis: Quickly understand database structures using list_tables and tables_schema, identifying key columns for product models, sales data, and dates.
    4. Dynamic Query Construction: Build SQL queries on-the-fly, adapting to user requirements and dataset peculiarities, incorporating LIKE patterns for flexible model name matching.
    5. Advanced Data Manipulation: Proficient in complex joins, subqueries, and aggregations, always considering how to best use LIKE for accurate product identification.
    6. Query Optimization: Fine-tune queries for optimal performance, balancing LIKE statements' flexibility with query efficiency.
    7. Data Quality Assurance: Implement checks to handle inconsistencies in product naming conventions, ensuring all relevant data is captured.

    Approach to Queries:
    1. Requirement Analysis: Interpret user requests, focusing on product model specifications and potential variations.
    2. Schema Exploration: Use list_tables and tables_schema to identify relevant tables and columns, especially those with product model information.
    3. Query Design: Construct SQL queries with strategic use of LIKE operators (e.g., 'LIKE '%S23 Ultra%'' instead of exact matching).
        For model variations like "S23 Plus", "S24 Plus" treat it as "S23+", "S24+" and use patterns such as:
        "S23+%" to capture variations starting with "S23+"
        "S24+%" to capture variations starting with "S24+"
        "S23 %" to include potential additional naming conventions, like "S23+ 5G"
        "S24 %" to include potential additional naming conventions, like "S24+ 5G"
    4. Validation: Use check_sql to verify query syntax and logic, ensuring correct LIKE statement formulation.
    5. Execution and Review: Run queries with execute_sql, evaluating results to check if all expected product variants are included.
    6. Iterative Refinement: Adjust LIKE patterns based on initial results or additional user input for comprehensive data capture.
    
    Additional Improvements:
    - **Clarify Variations in Model Names**: When using LIKE, always account for variations (e.g., different storage options, colors, minor model distinctions) and ensure they are captured accurately.
    - **Handle Missing Data**: If certain records or columns are not found, return a clear message indicating the issue (e.g., "No data found for this model in the specified period") instead of running a query that returns null or incorrect data.
    
    Important Columns in the Database:

    1. mx_dlb_new (Mobile Devices Market Share)
        - year_market_share: Year for market share data.
        - region: Geographical region (e.g., EAST, WEST).
        - competitor: Brand or competitor name (e.g., Samsung, Apple).
        - week_year: Specific week of the year.
        - month_market_share: Month of the market share data.
        - priceband: Price range of products (e.g., "30K - 40K").
        - salestype: Type of sales (e.g., Sales, Sellout).
        - channel: Sales channel (e.g., GT for General Trade).
        - model: Product model (e.g., WATCH4 44MM_LTE).
        - Product_Group: Type of product (e.g., Wearable, Smart Phone).
        - sellout_value: Total value of products sold.
        - sellout_volume: Units sold.

    2. mx_sales_new (Mobile Devices Sales Data)
        - salestype: Type of sales (e.g., Sell Out).
        - salescategory: Sales category (e.g., Normal, Promotional).
        - channel: Sales channel (e.g., B2B).
        - priceband: Price range of product sold.
        - region: Geographical region.
        - year, month, day: Date-related columns.
        - date: Exact transaction date.
        - weekyear: Week and year of sales.
        - segment: Product segment (e.g., Smart Phone, Wearable).
        - model: Product model.
        - quantity: Number of units sold.
        - amount: Total sales amount.

    3. ce_sales_new (Consumer Electronics Sales Data)
        - year_sale: Year of sale.
        - month_sale: Month of sale.
        - Sales Type: Type of sale (e.g., Sell Out).
        - salestype: (e.g., Union)
        - p0: Broad product category (e.g., DA for Digital Appliances, VD).
        - P1: Product type within the broad category (e.g., AC for Air Conditioners, UNMAP).
        - p2: Product variant or subtype (e.g., AIRPURIFIER, PANEL, DW etc.,).
        - p3: Specific model or further product classification (e.g., AIRPURI for specific air purifiers, DISHWASHER).
        - week_year_sale: Week and year of sale.
        - date_sale: Date of sale.
        - channel: Sales channel (e.g., MT for Modern Trade).
        - region: Geographical region.
        - quantity: Number of units sold.
        - amount: Total sales amount.

    4. ce_dlb_new (Consumer Electronics Market Share Data)
        - year_market_share: Year of market share data.
        - month_market_share: Month of market share data.
        - week_year: Week and year of market share data.
        - market_share_date: Date of market share data collection.
        - region: Geographical region.
        - channel: Sales channel (e.g., DD, MR).
        - competitor: Competitor brand.
        - p0: Broad product category (e.g., DA for Digital Appliances, VD).
        - p1: Product type within the broad category (e.g., AC for Air Conditioners, UNMAP).
        - p2: Product variant or subtype  (e.g., AC, SOUNDBAR, DISHWASHER).
        - p3: Specific model or further product classification (e.g., INVERTER-5*)
        - sellout_volume: Units sold.
        - sellout_value: Total sales amount.

    Key Reminders:
    1. Use LIKE with appropriate wildcards (%) for flexible model matching.
    2. Consider common variations in model names (e.g., storage sizes, colors) when constructing LIKE patterns.
    3. When handling price band queries:
       - For exact price band matches, use the specific category (e.g., '10 K-15 K').
       - For price ranges spanning multiple categories:
         - Use OR conditions to include all relevant price bands.
         - Example: For "10K - 20K", include both '10 K-15 K' AND '15 K-20 K'.
    4. Validate results to ensure all relevant product data and price ranges are captured.
    5. Be prepared to refine LIKE patterns and price band logic if initial results are incomplete.

    Quarterly Information:
    - Q1: January, February, and March
    - Q2: April, May, and June
    - Q3: July, August, and September
    - Q4: October, November, and December

    Scope and Data Presentation:
    - Only respond to greetings and questions related to Samsung's sales and market share data.
    - Politely redirect off-topic queries, asking users to rephrase questions accordingly.
    - Display results in Indian currency units (Lakhs and Crores) for monetary values and in Thousands (K) for volume metrics.

    Database Structure:
    Four main tables: 'ce_dlb_new', 'ce_sales_new', 'mx_dlb_new', 'mx_sales_new' (covering consumer electronics and mobile devices for market share and sales data).

    Query Types:
    1. Sales Data: Use 'ce_sales_new' or 'mx_sales_new' with flexible model matching.
    2. Market Share Data: Use 'ce_dlb_new' or 'mx_dlb_new' for regional analysis.
    3. Date Ranges: Handle year, month, and week filters.
    4. Regional Filters: Filter by geographical regions (e.g., EAST, WEST, NORTH 2).
    5. Model Variations: Use LIKE for variations such as storage sizes and colors.

    Additional Information:
    - Apple 15 launch date: September 22, 2023 (relevant for market trend analysis).

    When constructing queries:
    - Always consider which join type is most appropriate for the specific analysis.
    - Use table aliases for clarity in complex joins (e.g., 'ce' for ce_sales_new).
    - Ensure join conditions accurately reflect the relationships between tables.
    - Use appropriate date functions (e.g., EXTRACT, DATE_TRUNC) for date-based analysis.
    - Implement proper grouping and aggregation for summary statistics.
    - Consider using CTEs for complex queries to improve readability and maintainability.

    STRICT RULES:
    1. Do NOT generate or assume any data if it does not exist in the database.
    2. If the query returns no data, respond explicitly with "No data found for this model or timeframe in the specified period."
    3. Only display data that is confirmed from the database; do not fabricate or assume values.
    4. Always validate the presence of data before providing final output.
    """,

    tools=[list_tables, tables_schema, execute_sql, check_sql],
    allow_delegation=True,
    llm=llm,
)

def process_query(query: str) -> str:
    extract_data = Task(
    description=f"Extract and verify data required for the query: {query}. Ensure data availability before outputting any results. If no data exists, respond with 'No data found for the specified criteria.'",
    expected_output="Database result for the query or confirmation of data absence.",
    agent=sql_dev,
    )


    crew = Crew(
        agents=[sql_dev],
        tasks=[extract_data],
        process=Process.sequential,
        verbose=1,
        cache=False,    
        memory=False,
        max_rpm=None,
    )

    result = crew.kickoff(inputs={"query": query})
    print(result)
    return result

def process_data(result):
    prompt = f"""
    Analyze the following data first if the data or result is making sense:    
    {result}

    STRICT RULES:
    1. If the result contains "No data found":
       - Return ONLY an empty string
       - Do NOT generate any data
       - Do NOT create CSV
    2. Dont hallucinate your answers or create any new numbers or data.
    3. Preserve ALL numeric values exactly as they appear in the input:
       - Keep ALL digits together without splitting (e.g., 67,342 must stay as "67,342")
       - Maintain original number formatting including commas and decimal points
       - Do not modify precision or numerical format
    4. Keep all numeric values in their original precision along with the numering system.
    
    Return ONLY the CSV data without any additional text or explanation.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that converts data to CSV format with appropriate column names."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def extract_business_insights(data):
    prompt = f"""
    Based on the following CSV data, extract meaningful business insights. 
    Highlight trends, potential opportunities, and any notable patterns.
    {data}
    Provide your insights in bullet points format.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert data analyst who provides business insights."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def handle_output(output, filename='output.csv'):
    if ',' in output and '\n' in output:
        try:
            df = pd.read_csv(io.StringIO(output))
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            df.to_csv(filename, index=False)
            return df
        except Exception as e:
            st.error(f"Error converting to DataFrame: {e}")
            return output
    else:
        return output

def main():

    st.sidebar.image("F:/hirthickkesh/sam/one_tapp_logo.png", width=150)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chatbot", "Data Profiling", "GenBI", "LIDA"])

    if page == "Chatbot":
        chatbot_page()
    elif page == "Data Profiling":
        data_profiling_page()
    elif page == "GenBI":
        genbi_page()
    elif page == "LIDA":
        lida_page()

def chatbot_page():
    st.title("Samsung Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, how can I help you today? Feel free to ask me anything about Samsung's sales and market share data!"}
        ]

    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, how can I help you today? Feel free to ask me anything about Samsung's sales and market share data!"}
        ]
        st.session_state.pop('df_output', None)
        st.session_state.pop('insights', None)
        st.experimental_rerun()

    # Display the previous DataFrame if it exists
    if 'df_output' in st.session_state:
        st.subheader("Previous Response")
        st.dataframe(st.session_state['df_output'])

        # Optionally, display the previous insights as well
        if 'insights' in st.session_state:
            st.subheader("Previous Business Insights")
            st.markdown(st.session_state['insights'])

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Your question:"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner('Processing your question...'):
                response = process_query(prompt)
                
                # Process the response and convert to DataFrame if possible
                csv_data = process_data(response)
                df_output = handle_output(csv_data)
                
                if isinstance(df_output, pd.DataFrame):
                    # Display the new DataFrame
                    st.dataframe(df_output)
                    with st.expander("View the Original Answer"):
                        st.markdown(response)

                    # Store the DataFrame in session state to persist it
                    st.session_state['df_output'] = df_output

                    # Extract business insights from the data
                    insights = extract_business_insights(df_output.to_csv(index=False))

                    # Store the insights in session state
                    st.session_state['insights'] = insights

                    # Display the insights below the table
                    st.subheader("Business Insights")
                    st.markdown(insights)

                else:
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

def data_profiling_page():
    st.header("Data Profiling Report")

    # Select which dataframe to profile
    profile_option = st.selectbox(
        "Select a dataframe for profiling:",
        ("MX Sales", "MX Market Share", "CE Sales", "CE Market Share")
    )

    if profile_option == "MX Sales":
        selected_df = df1
        profile_file_name = "mx_sales_profiling_report.html"
    elif profile_option == "MX Market Share":
        selected_df = df2
        profile_file_name = "mx_market_share_profiling_report.html"
    elif profile_option == "CE Sales":
        selected_df = df3
        profile_file_name = "ce_sales_profiling_report.html"
    elif profile_option == "CE Market Share":
        selected_df = df4
        profile_file_name = "ce_market_share_profiling_report.html"

    if st.button("Generate Profiling Report"):
        with st.spinner('Generating profiling report...'):
            profile = ProfileReport(selected_df, title=f"{profile_option} Data Profiling Report")
            output_path = f"F:/hirthickkesh/sam/{profile_file_name}"
            profile.to_file(output_file=output_path)

            st.success(f"Profiling report generated successfully at {output_path}!")

            # Display the profiling report within Streamlit
            with open(output_path, 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)

def genbi_page():
    st.header("Generative Business Intelligence Analysis")

    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Seasonality and Festive Trends",
            "Pricing Insights",
            "Historical vs Current Sales",
            "Sales Comparison (MTD, QTD, YTD)",
            "Regional Sales Comparison",
            "Product Category Comparison",
            "Market Share Analysis",
            "Sales Values by Product"
        ]
    )

    # Move price band selection outside the Analyze button for Pricing Insights
    if analysis_type == "Pricing Insights":
        price_band = st.selectbox("Select a price band for analysis:", df1['priceband'].unique())
        st.write(f"Selected price band: {price_band}")  # Debug statement

    if st.button("Analyze"):
        with st.spinner('Analyzing data...'):
            if analysis_type == "Seasonality and Festive Trends":
                query = "Analyze seasonality and festive trends in sales data. Show results in a line chart."
            elif analysis_type == "Pricing Insights":
                price_elasticity = df1[df1['priceband'] == price_band].groupby('model')['amount'].sum().reset_index()
                st.write(f"Number of rows in price_elasticity: {len(price_elasticity)}")  # Debug statement
                if not price_elasticity.empty:
                    st.write(f"Sales for products in the {price_band} price band:")
                    st.dataframe(price_elasticity)
                    fig = px.bar(price_elasticity, x='model', y='amount', title=f"Sales for Models in {price_band} Price Band")
                    st.plotly_chart(fig)
                else:
                    st.write(f"No data available for the selected price band: {price_band}")
            elif analysis_type == "Historical vs Current Sales":
                query = "Compare historical sales data with current trends. Show results in a bar chart."
            elif analysis_type == "Sales Comparison (MTD, QTD, YTD)":
                query = "Compare sales based on MTD vs LMTD, QTD vs last year, and YTD vs LYTD. Show results in a grouped bar chart."
            elif analysis_type == "Regional Sales Comparison":
                region_sales = df1.groupby('region')['amount'].sum().reset_index()
                st.write("Sales by Region")
                st.dataframe(region_sales)
                fig = px.bar(region_sales, x='region', y='amount', title="Sales by Region")
                st.plotly_chart(fig)
            elif analysis_type == "Product Category Comparison":
                category_sales = df1.groupby('segment')['amount'].sum().reset_index()
                st.write("Sales by Product Category")
                st.dataframe(category_sales)
                fig = px.bar(category_sales, x='segment', y='amount', title="Sales by Product Category")
                st.plotly_chart(fig)
            elif analysis_type == "Market Share Analysis":
                query = "Analyze market share for Samsung and competitors. Show results in a stacked bar chart."
            elif analysis_type == "Sales Values by Product":
                sales_value = df1.groupby('model')['amount'].sum().reset_index().sort_values('amount', ascending=False)
                
                st.write("Sales Value by Product")
                st.dataframe(sales_value)
                
                viz_type = st.radio("Select visualization type:", ["Bar Chart", "Treemap"])
                
                if viz_type == "Bar Chart":
                    top_20 = sales_value.head(20)
                    fig = px.bar(top_20, x='model', y='amount', 
                                 title="Top 20 Products by Sales Value",
                                 labels={'amount': 'Sales Amount (INR)', 'model': 'Product Model'},
                                 color='amount',
                                 color_continuous_scale=px.colors.sequential.Viridis)
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)
                
                else:  # Treemap
                    fig = px.treemap(sales_value, path=['model'], values='amount',
                                     title="Sales Value by Product (Treemap)",
                                     color='amount',
                                     color_continuous_scale=px.colors.sequential.Viridis)
                    fig.update_traces(textinfo="label+value")
                    st.plotly_chart(fig)
                
                total_sales = sales_value['amount'].sum()
                top_5_sales = sales_value.head(5)['amount'].sum()
                top_5_percentage = (top_5_sales / total_sales) * 100
                
                st.write(f"Total Sales: ₹{total_sales:,.2f}")
                st.write(f"Top 5 products account for {top_5_percentage:.2f}% of total sales")
                
                sales_value['cumulative_percentage'] = sales_value['amount'].cumsum() / total_sales * 100
                pareto_threshold = sales_value[sales_value['cumulative_percentage'] >= 80].index[0] + 1
                
                st.write(f"Pareto Analysis: {pareto_threshold} products account for 80% of total sales")

            if analysis_type not in ["Pricing Insights", "Regional Sales Comparison", "Product Category Comparison", "Sales Values by Product"]:
                result = lake.chat(query)
                
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                    
                    if analysis_type == "Seasonality and Festive Trends":
                        fig = px.line(result, x=result.columns[0], y=result.columns[1], title="Seasonality and Festive Trends")
                    elif analysis_type == "Historical vs Current Sales":
                        fig = px.bar(result, x=result.columns[0], y=result.columns[1], title="Historical vs Current Sales")
                    elif analysis_type == "Sales Comparison (MTD, QTD, YTD)":
                        fig = px.bar(result, x=result.columns[0], y=result.columns[1:], title="Sales Comparison (MTD, QTD, YTD)", barmode="group")
                    elif analysis_type == "Market Share Analysis":
                        fig = px.bar(result, x=result.columns[0], y=result.columns[1:], title="Market Share Analysis", barmode="stack")
                    
                    st.plotly_chart(fig)
                else:
                    st.write(result)

def lida_page():
    st.header("LIDA - Language-based Interactive Data Analysis")

    dataset_option = st.selectbox(
        "Select a dataset for LIDA analysis:",
        ("MX Sales", "MX Market Share", "CE Sales", "CE Market Share")
    )
    
    # Load the correct dataset and set persona based on the selection
    if dataset_option == "MX Sales":
        selected_df = df1
        persona = "Chief Sales Officer (CSO) with a focus on mobile electronics sales"
    elif dataset_option == "MX Market Share":
        selected_df = df2
        persona = "Chief Marketing Officer (CMO) specializing in mobile electronics market share"
    elif dataset_option == "CE Sales":
        selected_df = df3
        persona = "Chief Sales Officer (CSO) focusing on consumer electronics sales"
    elif dataset_option == "CE Market Share":
        selected_df = df4
        persona = "Chief Marketing Officer (CMO) specializing in consumer electronics market share"

    analysis_type = st.selectbox(
        "Select analysis type:",
        ("Summarize", "Visualize")
    )

    if st.button("Generate LIDA Analysis"):
        with st.spinner('Generating LIDA analysis...'):
            # Generate summary
            summary = lida.summarize(selected_df)

            if analysis_type == "Summarize":
                st.subheader("Dataset Summary")
                st.write(summary)
            
            elif analysis_type == "Visualize":
                # Generate goals with persona
                goals = lida.goals(summary, n=5, persona=persona)
                st.subheader("Analysis Goals and Visualizations")

                # Generate visualization
                library = "seaborn"
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                
                for i, goal in enumerate(goals):
                    st.write(f"Goal {i+1}: {goal}")
                    charts = lida.visualize(summary=summary, goal=goal, textgen_config=textgen_config, library=library)
                    
                    if charts:
                        selected_viz = charts[0]
                        
                        if selected_viz.raster:
                            imgdata = base64.b64decode(selected_viz.raster)
                            img = Image.open(io.BytesIO(imgdata))
                            st.image(img, use_column_width=True)
                        else:
                            st.write("Visualization not available for this goal.")
                    else:
                        st.write("No visualization generated for this goal.")


if __name__ == "__main__":
    # Load your dataframes
    df1 = pd.read_csv(r'F:\hirthickkesh\sam\mx_sales_new.csv')
    df2 = pd.read_csv(r'F:\hirthickkesh\sam\mx_dlb_new.csv')
    df3 = pd.read_csv(r'F:\hirthickkesh\sam\ce_sales_new.csv')
    df4 = pd.read_csv(r'F:\hirthickkesh\sam\ce_dlb_new.csv')

    # Initialize SmartDatalake
    lake = SmartDatalake([df1, df2, df3, df4], config={"llm": llm})

    main()