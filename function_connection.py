import pandas as pd
from typing import List, Optional
import numpy as np
import psycopg2
from io import BytesIO
import sys
from dotenv import load_dotenv
import io
import os
from datetime import date, datetime
from dataclasses import dataclass
load_dotenv()
arguments = sys.argv
from tabulate import tabulate

@dataclass
class SQLQueries:
    create_table: str
    insert: str
    update: str

class DataFrameToSQL:
    def __init__(self, df: pd.DataFrame, table_name: str, primary_keys: List[str] | str):
        """
        Initialize with DataFrame and table configuration
        
        Args:
            df: pandas DataFrame to convert
            table_name: target table name
            primary_keys: list of primary key columns or single primary key column name
        """
        self.df = df
        self.table_name = table_name
        self.primary_keys = [primary_keys] if isinstance(primary_keys, str) else primary_keys
        self._validate_inputs()
        
    def _validate_inputs(self):
        """ตรวจสอบความถูกต้องของ input"""
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not all(pk in self.df.columns for pk in self.primary_keys):
            raise ValueError("Primary keys must exist in DataFrame columns")

    # def _get_sql_type(self, dtype, series=None) -> str:
    #     """แปลง pandas dtype เป็น SQL type (รองรับ date จริง)"""

    #     if np.issubdtype(dtype, np.integer):
    #         return 'INTEGER'
    #     if pd.api.types.is_datetime64_any_dtype(dtype):
    #         return "TIMESTAMP"
    #     if np.issubdtype(dtype, np.floating):
    #         return 'DOUBLE PRECISION'

    #     if np.issubdtype(dtype, np.datetime64):
    #         return 'DATE'

    #     if np.issubdtype(dtype, np.bool_):
    #         return 'BOOLEAN'

    #     # ⭐ สำคัญ: object ที่เป็น datetime.date
    #     if dtype == object and series is not None:
    #         if series.dropna().apply(lambda x: isinstance(x, (date, datetime))).all():
    #             return 'DATE'

    #     return 'TEXT'
    def _get_sql_type(self, dtype, series=None) -> str:
        """แปลง pandas dtype เป็น SQL type (รองรับ timezone + python date)"""

        # 1️⃣ datetime64 (รวม timezone)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "DATE"  # หรือ "TIMESTAMP" ถ้าเก็บเวลา

        # 2️⃣ integer (รองรับ Int64 ด้วย)
        if pd.api.types.is_integer_dtype(dtype):
            return "INTEGER"

        # 3️⃣ float
        if pd.api.types.is_float_dtype(dtype):
            return "DOUBLE PRECISION"

        # 4️⃣ boolean
        if pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"

        # 5️⃣ object ที่เป็น python date จริง
        if pd.api.types.is_object_dtype(dtype) and series is not None:
            if series.dropna().apply(lambda x: isinstance(x, (date, datetime))).all():
                return "DATE"

        return "TEXT"

    def generate_queries(self, temp_table: Optional[str] = None) -> SQLQueries:
        """
        สร้าง SQL queries สำหรับ CREATE, INSERT, และ UPDATE
        
        Args:
            temp_table: ชื่อตาราง temporary (ถ้าไม่ระบุจะใช้ชื่อเดียวกับ table_name)
        Returns:
            SQLQueries object containing all necessary queries
        """
        temp_table = temp_table or f"temp_{self.table_name}"
        
        # สร้าง column definitions
        # column_defs = [
        #     f"{col} {self._get_sql_type(dtype)}"
        #     for col, dtype in self.df.dtypes.items()
        # ]
        column_defs = [
            f"{col} {self._get_sql_type(dtype, self.df[col])}"
            for col, dtype in self.df.dtypes.items()
        ]
        # print(f"Column definitions: {column_defs}")  # Debugging output
        # CREATE TABLE query
        create_table = f"""
        CREATE UNLOGGED TABLE IF NOT EXISTS {temp_table} (
            {', '.join(column_defs)}
        )
        """
        
        # Column names for INSERT/UPDATE
        columns = ', '.join(self.df.columns)
        
        # INSERT query
        insert = f"""
        INSERT INTO {self.table_name} ({columns})
        SELECT {columns} FROM {temp_table}
        ON CONFLICT ({', '.join(self.primary_keys)}) DO NOTHING
        """
        
        # UPDATE query
        update_columns = [col for col in self.df.columns if col not in self.primary_keys]
        update_sets = [f"{col} = EXCLUDED.{col}" for col in update_columns]
        
        update = f"""
        INSERT INTO {self.table_name} ({columns})
        SELECT {columns} FROM {temp_table}
        ON CONFLICT ({', '.join(self.primary_keys)})
        DO UPDATE SET {', '.join(update_sets)}
        """
        
        return SQLQueries(create_table, insert, update)

class DatabaseConnection:
    CONFIG = {
        'BOT': {
            'dbname': os.getenv('dbname_BOT'),
            'user': os.getenv('user_BOT'),
            'password': os.getenv('password_BOT'),
            'host': os.getenv('host_BOT'),
            'port': os.getenv('port_BOT'),
            'sslmode': 'require'
        }
    }
    
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        if project_name not in self.CONFIG:
            raise ValueError(f"Unknown project: {project_name}")
    
    def connect(self):
        """สร้าง database connection"""
        return psycopg2.connect(**self.CONFIG[self.project_name])
    
    def fetch_dataframe(self, query: str) -> pd.DataFrame:
        """ดึงข้อมูลเป็น DataFrame"""
        with self.connect() as conn:
            return pd.read_sql(query, conn)
    
    def upsert_dataframe(self, df: pd.DataFrame, table_name: str, primary_keys: List[str] | str,
                        temp_table: Optional[str] = None, chunk_size: int = 10000):
        """
        Upsert DataFrame เข้า database
        
        Args:
            df: DataFrame ที่ต้องการ upsert
            table_name: ชื่อตารางปลายทาง
            primary_keys: primary key columns
            temp_table: ชื่อตาราง temporary (optional)
            chunk_size: ขนาด chunk สำหรับการ copy ข้อมูล
        """
        
        sql_generator = DataFrameToSQL(df, table_name, primary_keys)
        temp_table = temp_table or f"temp_{table_name}"
        queries = sql_generator.generate_queries(temp_table)
        # print(queries)
        with self.connect() as conn:
            conn.autocommit = False
            with conn.cursor() as cursor:
                try:
                    # สร้างและเคลียร์ temporary table
                    cursor.execute(queries.create_table)
                    cursor.execute(f"TRUNCATE TABLE {temp_table}")
                    
                    # Copy ข้อมูลแบบ chunk
                    for start in range(0, len(df), chunk_size):
                        chunk = df.iloc[start:start + chunk_size]
                        buffer = BytesIO()
                        chunk.to_csv(buffer, index=False, header=False, sep="\t", encoding='utf-8')
                        buffer.seek(0)
                        cursor.copy_from(buffer, temp_table, sep="\t")
                    
                    # ทำการ upsert
                    cursor.execute(queries.update)
                    cursor.execute(f"DROP TABLE IF EXISTS {temp_table};")
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise Exception(f"Error during upsert: {str(e)}")

    def get_data(self, query: str, batch_size: int = 500, **params) -> pd.DataFrame:
        """
        ดึงข้อมูลจากฐานข้อมูลโดยใช้ SQL query ที่เขียนเอง และสามารถฟิลเตอร์ข้อมูลด้วยตัวแปรบางตัว
        Args:
            query: คำสั่ง SQL ที่ต้องการรัน
            params: ตัวแปรที่ใช้สำหรับฟิลเตอร์ใน SQL (เช่น price=1000, item_number='A12345')
            batch_size: จำนวนแถวที่ดึงในแต่ละครั้ง
        Returns:
            DataFrame ที่ได้จากการดึงข้อมูล
        """
        try:
            with self.connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    columns = [desc[0] for desc in cursor.description]  # ดึงชื่อคอลัมน์

                    data = []  # รายการที่เก็บข้อมูล
                    while True:
                        batch = cursor.fetchmany(batch_size)  # ดึงข้อมูลเป็นชุด
                        if not batch:  # ถ้าไม่มีข้อมูลให้ออกจากลูป
                            # print("No more data to fetch.")
                            break
                        # print(f"Fetched {len(batch)} rows.")  # ดูจำนวนแถวที่ดึงมา
                        data.extend(batch)  # เพิ่มข้อมูลที่ดึงมาเข้าใน list

                    if not data:
                        "No data fetched."
                        return pd.DataFrame(columns=columns)
    

                    # แปลงข้อมูลเป็น DataFrame
                    df = pd.DataFrame(data, columns=columns)
                    
                    # คืนค่า DataFrame ที่ได้
                    return df
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def update_data(self, schema_name: str, table_name: str, temp_table_name: str, df: pd.DataFrame, key_columns: list, batch_size: int = 1000):
        """
        อัปเดตข้อมูลจาก DataFrame ไปยังฐานข้อมูลแบบ Batch Update

        Args:
            schema_name: ชื่อ schema ที่ตารางอยู่
            table_name: ชื่อตารางที่ต้องการอัปเดต
            df: DataFrame ที่มีข้อมูลที่ต้องการอัปเดต
            key_columns: รายชื่อคอลัมน์ที่ใช้เป็นเงื่อนไข WHERE
            batch_size: จำนวนข้อมูลที่อัปเดตในแต่ละรอบ (ค่าเริ่มต้นคือ 1000)
        """
        if df.empty:
            raise ValueError("DataFrame ว่างเปล่า!")

        update_columns = [col for col in df.columns if col not in key_columns]  # คอลัมน์ที่ต้องอัปเดต
        if not update_columns:
            raise ValueError("ไม่มีคอลัมน์ให้อัปเดต!")

        # สร้าง TEMP TABLE สำหรับอัปเดตข้อมูล
        temp_table = temp_table_name

        try:
            with self.connect() as conn:
                with conn.cursor() as cursor:
                    # สร้าง Temporary Table เพื่อโหลดข้อมูลเข้าไปก่อน
                    create_temp_table_sql = f"""
                    CREATE TEMP TABLE {temp_table} AS 
                    SELECT * 
                    FROM 
                        {schema_name}.{table_name} WHERE 1=0; 
                    """
                    cursor.execute(create_temp_table_sql)

                    # แบ่งข้อมูลออกเป็นหลายๆ batch เพื่อเพิ่มประสิทธิภาพ
                    num_batches = (len(df) // batch_size) + 1
                    for batch_num in range(num_batches):
                        batch_df = df.iloc[batch_num * batch_size: (batch_num + 1) * batch_size]

                        # ใช้ COPY เพื่อเพิ่มข้อมูลใน TEMP TABLE
                        cols = ", ".join(batch_df.columns)
                        
                        # ใช้ StringIO แทนการใช้ line_terminator ใน to_csv
                        output = io.StringIO()
                        batch_df.to_csv(output, index=False, header=False, sep=',')  # ไม่มี line_terminator
                        output.seek(0)  # ไปที่จุดเริ่มต้นของ StringIO
                        copy_sql = f"COPY {temp_table} ({cols}) FROM stdin WITH CSV DELIMITER ','"
                        cursor.copy_expert(copy_sql, output)

                        # ใช้ UPDATE ... FROM ... เพื่ออัปเดตตารางหลัก
                        # set_clause = ", ".join([f"{table_name}.{col} = {temp_table}.{col}" for col in update_columns])  # ระบุ schema ใน SET
                        set_clause = ", ".join([f"{col} = {temp_table}.{col}" for col in update_columns])  # ระบุ schema ใน SET
                        # print(set_clause)
                        # where_clause = " AND ".join([f"{table_name}.{col} = {temp_table}.{col}" for col in key_columns])  # ระบุ schema ใน WHERE
                        where_clause = " AND ".join([f"{schema_name}.{table_name}.{col} = {temp_table}.{col}" for col in key_columns])  # ระบุ schema ใน WHERE
                        # print(where_clause)

                        update_sql = f"""
                        UPDATE {schema_name}.{table_name} 
                        SET 
                            {set_clause}
                        FROM {temp_table}
                        WHERE {where_clause};
                        """
                        # print(f"Executing update SQL: {update_sql}")  # Print for debugging
                        cursor.execute(update_sql)

                    conn.commit()

                    # print(f"✅ อัปเดตข้อมูลจาก DataFrame ไปยัง {schema_name}.{table_name} สำเร็จ!")
        except Exception as e:
            raise Exception(f"เกิดข้อผิดพลาดในการอัปเดตข้อมูล: {str(e)}")


        
class FileConverter:
    def __init__(self, input_file: str, detect_digit: int):
        self.input_file = input_file
        self.detect_digit = detect_digit

    def _detect_columns(self, header_lines):
        dash_counts = [line.count('-') for line in header_lines]
        pattern_idx = dash_counts.index(max(dash_counts))
        
        # column header line is right above the dash line
        column_header_line = header_lines[pattern_idx - 1]
        dash_line          = header_lines[pattern_idx]
        
        col_positions = []
        in_dash = False
        start_index = 0
        
        for i, ch in enumerate(dash_line):
            if ch == '-':
                if not in_dash:
                    in_dash = True
                    start_index = i
            else:
                if in_dash:
                    in_dash = False
                    end_index = i
                    col_positions.append((start_index, end_index))
        
        columns = []
        for (start, end) in col_positions:
            raw_name = column_header_line[start:end].strip()
            col_name = (raw_name.lower()
                                .replace(' ', '_')
                                .replace('.', '_')
                                .replace('/', '_')
                                .replace('-', '_'))
            columns.append(col_name)
        
        return columns, col_positions, dash_line

    def _parse_line(self, line, col_positions):
        row_values = []
        for (start, end) in col_positions:
            chunk = line[start:end].strip()
            row_values.append(chunk)
        return row_values

    def prn_to_dataframe(self, columns_wanted=None):
        with open(self.input_file, encoding='iso-8859-1') as f:
            lines = f.readlines()
        
        header_lines = lines[:20]
        columns_all, col_positions, dash_line = self._detect_columns(header_lines)
        
        # Possibly find where the dash_line is in the file to skip it
        dash_index = lines.index(dash_line + '\n') if (dash_line + '\n') in lines else None
        
        data_lines = []
        for i, ln in enumerate(lines):
            # skip dash line & header line 
            if dash_index and (i == dash_index or i == dash_index - 1):
                continue
            # keep lines that pass detect_digit
            if len(ln) > self.detect_digit and ln[self.detect_digit].isdigit():
                data_lines.append(ln.rstrip('\n'))
        
        parsed_rows = [
            self._parse_line(ln, col_positions) 
            for ln in data_lines
        ]
        
        df = pd.DataFrame(parsed_rows, columns=columns_all)
        
        if columns_wanted:
            df = df[columns_wanted]
        return df

def print_table(dtf):
    print(tabulate(dtf,headers='keys', tablefmt='simple_outline')) 