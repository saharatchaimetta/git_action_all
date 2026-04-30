import pandas as pd
from function_connection import DataFrameToSQL,DatabaseConnection,FileConverter,print_table
import os
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import numpy as np
load_dotenv(dotenv_path=r"C:\Users\User\Desktop\Python Projects\line_bot\shift-scheduler\clane_data\.env.local")
today = date.today()

def change_code_to_name(dataframe_change):
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
query_data_date = """
    SELECT 
        name,
        code_name,
        last_shift

    FROM
        user_name_bg_1000
    WHERE
        status = True

        """
params = {} 
data_code_name = DatabaseConnection("BOT").get_data(query_data_date, **params)
print_table(data_code_name.head(10))
if data_code_name.empty:
    raise ValueError("❌ ไม่มีข้อมูลพนักงาน")

# ================== CONFIG ==================
BASE_SHIFTS = ["shift_1","shift_2","shift_3","shift_4","shift_5","shift_6"]
OPTIONAL_SHIFTS = ["shift_receive","free_day"]
ALL_SHIFTS = BASE_SHIFTS + OPTIONAL_SHIFTS

WEEKDAY_TO_CAFE = {
    0: "C", 1: "F", 2: "G", 3: "H", 4: "E", 5: "B", 6: "All"
}

# ================== SORT BY LAST SHIFT ==================
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

    # ---- rotate เฉพาะผลัดที่มีค่า ----
    # คอลัมน์ที่มีค่า
    active_cols = [c for c in ALL_SHIFTS if current[c] is not None]

    # ค่าในผลัด
    active_vals = current[active_cols].tolist()

    # rotate ค่า
    rotated_vals = active_vals[-1:] + active_vals[:-1]

    # ใส่ค่ากลับ
    for col, val in zip(active_cols, rotated_vals):
        new[col] = val


    for col in ALL_SHIFTS:
        if col not in active_cols:
            new[col] = None

    # ---- day off special shift ----
    if day_off:
        new["shift_7"] = new["shift_1"]
        new["shift_8"] = new["shift_2"]
    else:
        new["shift_7"] = None
        new["shift_8"] = None

    rows.append(new)
    current = new

# ================== FINAL DF ==================
df_schedule = pd.concat([df_day_1, pd.DataFrame(rows)], ignore_index=True)

df_schedule["cafe_schedule"] = pd.to_datetime(df_schedule["date"]).dt.weekday.map(WEEKDAY_TO_CAFE)
df_schedule["date"] = pd.to_datetime(df_schedule["date"])  # สำคัญสำหรับ DB
df_schedule = df_schedule.fillna('-')

# ================== RESULT ==================
print_table(df_schedule.head(40))
df_schedule_named = change_code_to_name(df_schedule)
print_table(df_schedule_named)



DatabaseConnection("BOT").upsert_dataframe(df_schedule,"shift_schedule",["date"],"temp_shift_schedule",1000)


