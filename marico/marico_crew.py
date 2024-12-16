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
import pickle
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
import os
import io
import base64
from PIL import Image
from openai import OpenAI
from lida import Manager, llm, TextGenerationConfig
import matplotlib

data = pd.read_csv(r'F:\hirthickkesh\marico\final_poc_data.csv')

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
# user_api_key = ''

user_api_key = ''
os.environ["OPENAI_API_KEY"] = user_api_key

client = OpenAI()

lida = Manager(text_gen = llm("openai"))


def get_db_connection():
    db_host = "localhost"
    db_name = "marico1"
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
def list_tables() -> str:
    """List the available tables in the database""" 
    db = get_db_connection()
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
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
    goal="Construct and execute highly accurate SQL queries for comprehensive analysis of product ordering, billing, and regional distribution data.",
    backstory="""
You are an expert SQL Analyst specializing in order, billing, and distribution data across multiple regions, brands, and product lines. Your expertise is in constructing flexible and precise SQL queries that handle variations in product naming conventions, discrepancies in order fulfillment, and dynamic regional and retailer-level filtering.

Core Competencies:
1. Advanced Retailer Analysis: Utilize the RTRCODE and RTRNAME columns to track orders and billing data at a retailer level, enhancing insights into distribution and performance by individual retailers.
2. Expert Use of LIKE Operator: Skilled in using the LIKE operator with wildcards for SKU_NAME to accurately capture variations in SKU details and flexible model naming across packaging or regional conventions.
3. Dynamic Query Construction: Craft flexible SQL queries tailored to user needs, utilizing the new columns like RTRCODE and RTRNAME for granular tracking.
4. Enhanced Data Validation: Implement checks to manage any inconsistencies or missing data in SKU_NAME, RTRNAME, and ORDER_STATUS, ensuring complete and accurate reporting.
5. Query Optimization: Balance query complexity with LIKE flexibility for efficient SKU matching, optimizing performance across potentially large datasets.
6. Sales Volume Forecasting: Utilize past six months of order and billing data to project next month's sales volume accurately, leveraging trends, seasonal variations, and historical patterns in the data.

Approach to Queries:
1. Requirement Analysis: Interpret requests with a focus on retailer (RTRCODE, RTRNAME), SKU, and regional details.
2. Schema Exploration: Use list_tables and tables_schema to identify columns such as SKU_CODE, SKU_NAME, RTRCODE, RTRNAME, and ORDER_STATUS.
3. Query Design: Construct SQL queries with LIKE statements for SKU variations, now with the ability to add retailer-level filters using RTRCODE and RTRNAME.
4. Validation: Confirm query accuracy with check_sql, ensuring all specified columns, including RTRCODE and RTRNAME, are accurately represented.
5. Execution and Review: Execute queries with execute_sql and validate results, particularly for completeness in capturing SKU variations, order statuses, and accurate regional or retailer-based data.
6. Sales Forecasting: Analyze temporal data (month_year, month, and Year), SKU-level trends, and regional factors to predict sales volume for upcoming periods, particularly focusing on the next month.
7. Growth-Focused Queries: Always calculate and present growth metrics in percentages when related to sales or revenue trends.

Important Columns in the Dataset:
- month_year: Combined month and year for temporal analysis. Months should always be represented as Jan, Feb, Mar (and not alphabetically).
- month and Year: Separate month and year fields for granular time-based analysis.
- DISTCODE: Distributor code for tracking orders by distributor.
- RTRCODE and RTRNAME: Retailer identifiers and names for retailer-level analysis.
- SKU_CODE and SKU_NAME: Product identifiers, with SKU_NAME needing LIKE patterns for flexible matching.
- BRAND_NAME: Brand associated with each SKU.
- ORDER_STATUS: Indicates order fulfillment status (e.g., "Fully Serviced").
- ORDER_QTY, BILLED_QTY, BALANCE_QTY: Metrics on ordered, billed, and remaining quantities.
- ORDER_VALUE, BILLED_VALUE, LOSS_VALUE: Financial metrics, with LOSS_VALUE indicating potential unbilled amounts.
- CUSTOMER_CODE: Identifier for customers linked to orders.
- REGION_NAME and STATE_NAME: Regional and state-level data for geolocation-based analysis.
- Area_Type: Classification of the customer’s location as either "Urban" or "Rural," providing insight into urban-rural trends in demand and order fulfillment patterns.
- channel: Classifying transactions as either GT (General Trade) or MT (Modern Trade), which is essential for trade-type analysis and segmentation.

Key Reminders:
1. Flexible SKU and Retailer Matching: Use LIKE with wildcards for flexible SKU and retailer name filtering.
2. Order Status Variants: Include ORDER_STATUS for analyzing fulfillment status, such as "Fully Serviced."
3. Price and Quantity Metrics: Display financial data in Lakhs and Crores and quantities in thousands (K).
4. Retailer and Regional Filters: Use RTRCODE, RTRNAME, REGION_NAME, and STATE_NAME for targeted retailer and geographical analysis.
5. Data Validation: Include checks for missing or null data, returning a clear message if relevant records are absent (e.g., "No data found for specified SKU").
6. Forecast Accuracy: Ensure that sales projections are data-driven, leveraging recent order and billing trends to provide actionable forecasts.

SKU_NAME Prefix Indicators:
- 'P' at the start of SKU_NAME indicates a Parachute product.
- 'S' at the start of SKU_NAME indicates a Saffola product.
- 'NHR' or 'NSA' at the start of SKU_NAME indicates a Nihar product.
- 'Livon' at the start of SKU_NAME indicates a Livon product.
- 'SW' at the start of SKU_NAME indicates a Set Wet product.
- 'SF' at the start of SKU_NAME indicates a Saffola product, commonly used for Saffola foods like soya chunks and oats.
- 'PCNO' at the start of SKU_NAME indicates a Parachute Coconut Oil product.
- 'REV' at the start of SKU_NAME indicates a Revive product.
- 'H&C' at the start of SKU_NAME indicates a Hair & Care product.
- 'MALT' at the start of SKU_NAME suggests a Maltova product.

Scope and Data Presentation:
- Response Handling: Answer queries specifically related to order, billing, SKU, and retailer data.
- Currency and Quantity Format: Display financials in Lakhs and Crores, with quantities in thousands (K).

Database Structure:
This dataset contains a single primary table with order, billing, SKU, and retailer details, ideal for granular SKU-level, retailer-level, and regional analysis.

Query Types:
1. Order and Billing Data: Focus on ORDER_QTY, BILLED_QTY, BALANCE_QTY, and related financial metrics.
2. Temporal Analysis: Use month_year, month, and Year for date-specific insights.
3. Retailer and Regional Analysis: Filter by RTRCODE, RTRNAME, REGION_NAME, and STATE_NAME.
4. SKU Variations: Utilize LIKE for SKU_NAME variations across products and packaging.
5. Sales Volume Forecasting: Analyze trends in the past six months of data to predict next month's sales volume, identifying patterns across SKUs, regions, and order channels.

    """,
    
    tools=[list_tables, tables_schema, execute_sql, check_sql],
    allow_delegation=True,
    llm=llm,
)



def process_query(query: str) -> str:
    extract_data = Task(
    description=f"Extract and verify data required for the query: {query}. Ensure data availability before outputting any results. If no data exists, respond with 'No data found for the specified criteria.'",
    expected_output="Database result for the query in structured and meaningfull way or confirmation of data absence.",
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
    return result
 
def extract_business_insights(result):
    prompt = f"""
    Extract meaningful business insights. 
    Highlight trends, potential opportunities, and any notable patterns.
    {result}
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

def main():
    st.sidebar.image("F:/hirthickkesh/sam/one_tapp_logo.png", width=150)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chatbot", "LIDA"])

    if page == "Chatbot":
        chatbot_page()
    elif page == "LIDA":
        lida_page()

def chatbot_page():
    st.title("Marico Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, how can I help you today? Feel free to ask me anything about Marico's sales and market share data!"}
        ]

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, how can I help you today? Feel free to ask me anything about Marico's sales and market share data!"}
        ]
        st.session_state.pop('response', None)
        st.session_state.pop('insights', None)
        st.experimental_rerun()

    # Display the previous response if it exists
    if 'response' in st.session_state:
        st.subheader("Previous Response")
        st.markdown(st.session_state['response'])

        # Display insights in a dropdown if they exist
        if 'insights' in st.session_state:
            st.subheader("Previous Business Insights")
            with st.expander("View Insights"):
                st.markdown(st.session_state['insights'])

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input
    if prompt := st.chat_input("Your question:"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner('Processing your question...'):
                # Process the query and get the response
                response = process_query(prompt)

                # Display response directly
                st.markdown(response)
                st.session_state['response'] = response

                # Extract and display insights in an expander
                insights = extract_business_insights(response)
                st.session_state['insights'] = insights

                st.subheader("Business Insights")
                with st.expander("View Insights"):
                    st.markdown(insights)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})


def lida_page():
    st.header("LIDA - Language-based Interactive Data Analysis")
    persona_options = {
        "Chief Supply Chain Officer (CSCO)": "Focuses on optimizing fulfillment, inventory, and distribution performance.",
        "Chief Financial Officer (CFO)": "Monitors revenue, losses, and financial metrics related to order fulfillment.",
        "Regional Sales Manager": "Enhances sales performance and manages distributor relationships across regions."
    }
    persona_choice = st.selectbox("Select a persona for analysis:", list(persona_options.keys()))
    persona_description = persona_options[persona_choice]
    st.write(f"**Persona Focus**: {persona_description}")

    analysis_type = st.selectbox(
        "Select analysis type:",
        ("Summarize", "Visualize")
    )

    if st.button("Generate LIDA Analysis"):
        with st.spinner('Generating LIDA analysis...'):
            summary = lida.summarize(data)

            if analysis_type == "Summarize":
                st.subheader("Dataset Summary")
                st.write(summary)
            
            elif analysis_type == "Visualize":
                goals = lida.goals(summary, n=5, persona=persona_choice)
                st.subheader("Analysis Goals and Visualizations")

                # library = "seaborn"
                library = 'matplotlib'
                textgen_config = TextGenerationConfig(n=1, temperature=0.7, use_cache=True)
                
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
    main()