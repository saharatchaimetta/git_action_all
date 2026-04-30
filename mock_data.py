import pandas as pd
from function_connection import DataFrameToSQL,DatabaseConnection,FileConverter,print_table
import os
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import numpy as np
load_dotenv(dotenv_path=r"C:\Users\User\Desktop\Project\Python Projects\line_bot\shift-scheduler\clane_data\.env.local")
today = date.today()

def change_code_to_name(dataframe_change:pd.DataFrame):
    query_data_date = """
        SELECT 
            name,
            code_name
        FROM
            line_bot.user_name_bg_1000
        order BY
            last_update DESC

            """
    params = {} 
    data_code_name = DatabaseConnection("BOT").get_data(query_data_date, **params)
    
    code_to_name = (
        data_code_name
        .set_index("code_name")["name"]
        .to_dict()
    )

    SHIFT_COLS = [
        "cafe_schedule",
        "shift_1",
        "shift_2",
        "shift_3",
        "shift_4",
        "shift_5",
        "shift_6",
        "shift_7",
        "shift_8",
        "shift_receive",
        "free_day",
    ]

    df_named = dataframe_change.copy()

    for col in SHIFT_COLS:
        df_named[col] = df_named[col].apply(
            lambda x: (
                "-" if x == "-"                       # ข้อ 1
                else code_to_name[x]                  # ข้อ 2
                if x in code_to_name
                else "ALL"                             # ข้อ 3
            )
        )

    return df_named
# ===== CONFIG =====



def get_data_from_db():
    query_data_date = """
        SELECT 
            name,
            last_update,
            code_name,
            last_shift,
            status

        FROM
            line_bot.user_name_bg_1000
        order BY
            last_update DESC

            """
    params = {} 
    data_code_name = DatabaseConnection("BOT").get_data(query_data_date, **params)
    print_table(data_code_name.head(10))
    data_code_name = data_code_name[(data_code_name['status'] == True) & (data_code_name['last_shift'] <= 8)]
    data_code_name['last_update'] = pd.to_datetime(data_code_name['last_update']).dt.date
    print_table(data_code_name)
    print(today)
    if data_code_name["last_update"].iloc[0] != today:
        print("Data is outdated. Please update the data.")
    today_check = date.today()
    print("Data fetched on:", today_check)
    print_table(data_code_name.head(10))
    if data_code_name.empty:
        raise ValueError("❌ ไม่มีข้อมูลพนักงาน")
    return data_code_name


def shift_schedule(data_name_on):
    BASE_SHIFTS = ["shift_1","shift_2","shift_3","shift_4","shift_5","shift_6"]
    OPTIONAL_SHIFTS = ["shift_receive","free_day"]
    ALL_SHIFTS = BASE_SHIFTS + OPTIONAL_SHIFTS

    WEEKDAY_TO_CAFE = {
        0: "C", 1: "F", 2: "G", 3: "H", 4: "E", 5: "B", 6: "All"
    }
    # ================== SORT BY LAST SHIFT ==================
    data_code_name = data_name_on
    # data_code_name = data_code_name[(data_code_name['status'] == True) & (data_code_name['last_shift'] <= 8)]
    print_table(data_code_name.head(10))
    df_sorted = data_code_name.sort_values("last_shift")
    codes = df_sorted["code_name"].tolist()
    n_people = len(codes)

    # ================== DEFINE SHIFT USED ==================
    shift_cols = BASE_SHIFTS.copy()

    if n_people >= 7:
        shift_cols.append("shift_receive")

    if n_people >= 8:
        shift_cols.append("free_day")

    # ================== DAY 1 ==================
    base_row = {
        "date": today,
        "day_off": today.weekday() >= 5,
    }

    for col, code in zip(shift_cols, codes):
        base_row[col] = code

    # ช่องที่ไม่ใช้
    for col in ALL_SHIFTS:
        base_row.setdefault(col, None)

    df_day_1 = pd.DataFrame([base_row])

    # ================== GENERATE 30 DAYS ==================
    rows = []
    current = df_day_1.iloc[0].copy()

    for _ in range(30):
        next_date = current["date"] + timedelta(days=1)
        day_off = next_date.weekday() >= 5

        new = current.copy()
        new["date"] = next_date
        new["day_off"] = day_off

        # คอลัมน์ที่มีค่า
        active_cols = [c for c in ALL_SHIFTS if current[c] is not None]

        # ค่าในผลัด
        active_vals = current[active_cols].tolist()

        # rotate ค่า
        rotated_vals = active_vals[-1:] + active_vals[:-1]

        # ใส่ค่ากลับ
        for col, val in zip(active_cols, rotated_vals):
            new[col] = val


        rows.append(new)
        current = new

    # ================== FINAL DF ==================
    df_schedule = pd.concat([df_day_1, pd.DataFrame(rows)], ignore_index=True)
    df_schedule['shift_7'] = np.where(df_schedule['day_off'] == True, df_schedule['shift_receive'], "-")
    df_schedule['shift_8'] = np.where(df_schedule['day_off'] == True, df_schedule['free_day'], "-")
    df_schedule['shift_receive'] = np.where(df_schedule['day_off'] == True, "-", df_schedule['shift_receive'])
    df_schedule['free_day'] = np.where(df_schedule['day_off'] == True, "-", df_schedule['free_day'])
    df_schedule["cafe_schedule"] = pd.to_datetime(df_schedule["date"]).dt.weekday.map(WEEKDAY_TO_CAFE)
    df_schedule["date"] = pd.to_datetime(df_schedule["date"])  # สำคัญสำหรับ DB
    df_schedule = df_schedule.fillna('-')

    # ================== RESULT ==================
    print_table(df_schedule.head(40))
    df_schedule_named = change_code_to_name(df_schedule)
    print_table(df_schedule_named)
    # df_schedule_named.to_csv('output.csv', encoding='utf-8-sig' )
    



    DatabaseConnection("BOT").upsert_dataframe(df_schedule,"line_bot.shift_schedule",["date"],"temp_shift_schedule",1000)



def main():
    data_name_on = get_data_from_db()
    shift_schedule(data_name_on)
    
    
if __name__ == "__main__":
    main()