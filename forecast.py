import pandas as pd
import snowflake.connector
import streamlit as st
from prophet import Prophet
from st_aggrid import AgGrid
import toml

login_data = toml.load("secrets.toml")
sfAccount = login_data['sfAccount']
sfUser = login_data['sfUser']
sfPassword = login_data['sfPassword']
sfWarehouse = login_data['sfWarehouse']
sfDatabase = login_data['sfDatabase']
sfSchema = login_data['sfSchema']

conn = snowflake.connector.connect(user=sfUser,
                                   password=sfPassword,
                                   account=sfAccount,
                                   warehouse=sfWarehouse,
                                   database=sfDatabase,
                                   schema=sfSchema)

def add_filler_to_dataframe(df, filler):
    """This assumes the dataframe only has one column"""
    df.loc[-1] = [filler]  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()
    return df

def add_filler_to_list(ls, filler):
    ls.insert(0, filler)
    return ls

def ask_for_database():
    database_query = run_show_query("show databases in account;")
    database_query = add_filler_to_list(database_query, None)
    database = st.selectbox("Select Database: ", database_query)
    return database

def ask_for_schema(database):
    schema_query = run_show_query("show schemas in {};".format(database))
    schema_query = add_filler_to_list(schema_query, None)
    schema = st.selectbox("Select Schema: ", schema_query)
    return schema

def ask_for_table(schema):
    try:
        table_query = run_show_query("show tables in {};".format(schema))
        table_query = add_filler_to_list(table_query, None)
        table = st.selectbox("Select Table: ", table_query)
        return table
    except:
        st.write("No tables exist in schema: {}".format(schema))
        
def select_columns(table, schema):
    column_query = run_query("Select COLUMN_NAME From INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME = '{}'".format(schema, table))
    columns = st.multiselect("Select fields in dataframe: ", column_query)

    filter_dim = []
    for col in columns:
        filter_query = run_query("Select Distinct {} From {}".format(col, table))
        filter_query = add_filler_to_dataframe(filter_query, "ALL")
        user_filter = st.selectbox("Filter {}".format(col), filter_query)
        filter_dim.append((col, user_filter))
        
    return filter_dim

def create_input_dataframe(columns, table):
    select_statement, where_statement = "Select ", "Where "
    for item in columns:
        select_statement += "{}, ".format(item[0])
        if item[1] != "ALL":
            where_statement += "{} = {} AND ".format(item[0], item[1])
            
    select_statement = select_statement[:len(select_statement) - 2] + ' '
    where_statement = where_statement[:len(where_statement) - 4]

    final_query = select_statement + "From {} ".format(table) + where_statement

    print('final query: ', final_query)
    df = run_query(final_query)
    return df

def create_lookahead_slider(max_periods):
    look_ahead = st.slider("Select Look-Ahead Period", 0, max_periods, max_periods//2)
    st.write("You selected look-ahead period: {}".format(look_ahead))
    return look_ahead


@st.experimental_memo(ttl=6000) #caches query for ttl seconds
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetch_pandas_all() #can try fetch_pandas_batches
    
@st.experimental_memo(ttl=6000)
def run_show_query(query):
    """snowflake.connector does not work with SHOW commands"""
    with conn.cursor() as cur:
        cur.execute(query)
        res = cur.fetchall() #can try fetch_pandas_batches
        return parse_show_query(res)
    
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')
    
def parse_show_query(res):
    return [res[i][1] for i in range(len(res))]

def clean_dataframe_for_prophet(df):
    """assumes user has selected datestamp and target as first two columns"""
    df = df[[df.columns[0], df.columns[1]]]
    df.columns = ["ds", "y"]
    print('length: ', len(df))
    df = df.groupby("ds", as_index=False).sum()
    print('new length: ', len(df))
    return df
    
def model_lifecycle(df, look_ahead, seg_name):
    print('forecast entered')
    set_to_create_forecast(True)
    df = clean_dataframe_for_prophet(df)
    last_time = max(df['ds'])
    m = Prophet()
    m = train_prophet_model(m, df)
    forecast = inference_prophet_model(m, look_ahead)
    chart = make_plotly_chart(m, forecast, seg_name, last_time)
    st.pyplot(chart)
    return (forecast, m, chart)
    
def train_prophet_model(model, df):
    model.fit(df)
    return model

def inference_prophet_model(model, look_ahead):
    future = model.make_future_dataframe(periods=look_ahead)
    forecast = model.predict(future)
    print('type: ', type(forecast))
    return forecast

def make_plotly_chart(model, forecast, seg_name, last_known):
    fig = model.plot(forecast, xlabel="Time", ylabel="{} Quantity".format(seg_name))
    ax = fig.gca()
    ax.set_title("Demand Forecast", size=34)
    ax.set_xlabel("Time", size=26)
    ax.set_ylabel("Demand Quantity", size=26)
    return fig

def set_dataframe_and_forecast(val):
    set_to_create_dataframe(val)
    set_to_create_forecast(val)
    return

def set_to_create_dataframe(val):
    st.session_state.to_create_dataframe = val
    return
    
def set_to_create_forecast(val):
    st.session_state.to_create_forecast = val
    return

def initialize_stateful_variables(ls):
    for var in ls:
        if var[0] not in st.session_state:
            st.session_state[var[0]] = var[1]
    return

def main():
    initialize_stateful_variables([("to_create_dataframe", False), ("to_create_forecast", False)])
    database = ask_for_database()
    if not database: st.stop()
    
    schema = ask_for_schema(database)
    if not schema: st.stop()
    
    table = ask_for_table(schema)
    if not table: st.stop()
    
    columns = select_columns(table, schema)
    if not columns: st.stop()
    
    with st.form("dataframe_form"):
        to_create_dataframe = st.form_submit_button("View Input Data")
        if to_create_dataframe or st.session_state.to_create_dataframe:
            st.session_state.to_create_dataframe = True
            input_dataframe = create_input_dataframe(columns, table)
            st.dataframe(input_dataframe)
            #AgGrid(input_dataframe)
            st.write("Your dataframe has {} rows".format(len(input_dataframe)))
        else:
            st.stop()
    
    with st.form(key="slider_form"):
        look_ahead = create_lookahead_slider(730)
        to_create_forecast = st.form_submit_button("Create Forecast")
        if to_create_forecast or st.session_state.to_create_forecast:
            st.session_state.to_create_forecast = True
            prediction, model, chart = model_lifecycle(input_dataframe, look_ahead, "Demand")
        else:
            st.stop()
            
    if type(prediction) != pd.DataFrame: st.stop()
    st.dataframe(prediction)
    #AgGrid(prediction)
    st.download_button("Download Prediction Dataset", data=convert_df(prediction), file_name='prediction.csv', mime='text/csv')            

main()