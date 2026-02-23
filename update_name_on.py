import pandas as pd
from function_connection import DataFrameToSQL,DatabaseConnection,FileConverter,print_table
from datetime import date, timedelta
from dotenv import load_dotenv
import numpy as np
import os
load_dotenv()

def get_data_name_on():
    query_data_date = """
        SELECT 
            *
        FROM
            user_name_bg_1000
        WHERE
            status = TRUE
        order BY
            last_update DESC

            """
    params = {} 
    data_code_name = DatabaseConnection("BOT").get_data(query_data_date, **params)
    return data_code_name
def get_shift_today():
    today = date.today()
    query_data_date = """
        SELECT 
            *
        FROM
            shift_schedule
        WHERE
            date = %(today)s
        order BY
            date DESC

            """
    params = {"today": today} 
    data_code_name = DatabaseConnection("BOT").get_data(query_data_date, **params)
    df_vertical = data_code_name.melt()
    df_vertical = df_vertical.rename(columns={"variable": "shift", "value": "code_name"})
    df_vertical = df_vertical[(df_vertical['shift'] != 'cafe_schedule') & (df_vertical['code_name'] != '-') & (df_vertical['shift'].str[0] != 'd')]
    df_vertical[['shift_type', 'shift_no']] = df_vertical['shift'].str.split('_', expand=True)
    mapping = {'day': 8, 'receive': 7}
    df_vertical['last_shift'] = (
        df_vertical['shift_no']
            .map(mapping)
            .fillna(pd.to_numeric(df_vertical['shift_no'], errors='coerce')))
    df_vertical = df_vertical.sort_values(by='last_shift', ascending=True).reset_index(drop=True)
    df_vertical = df_vertical[['code_name', 'shift', 'last_shift']]
    return df_vertical

    
def main():
    dataframe_shift = get_shift_today()
    # print_table(dataframe_shift)
    dataframe_name = get_data_name_on()
    # print_table(dataframe_name)
    dataframe_update = pd.merge(dataframe_name, dataframe_shift, on="code_name", how="left")
    dataframe_update = dataframe_update[['id', 'code_name','shift', 'last_shift_y','last_update']]
    dataframe_update = dataframe_update.rename(columns={"last_shift_y": "last_shift","shift": "remark"})
    dataframe_update['last_shift'] = dataframe_update['last_shift'].fillna(0).astype(int)
    # print_table(dataframe_update)
    DatabaseConnection("BOT").upsert_dataframe(dataframe_update,"user_name_bg_1000",["id"],"temp_user_name_bg_1000",1000)
    
    # print_table(dataframe_update)
    
if __name__ == "__main__":
    main()