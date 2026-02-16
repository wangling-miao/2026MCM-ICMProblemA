
import pandas as pd
import os
import sys

# Copy necessary functions
def standardize_columns(df):
    column_mapping = {
        'Test_Time(s)': 'Time', 'Time(s)': 'Time',
        'Current(A)': 'Current', 'Current (A)': 'Current',
        'Voltage(V)': 'Voltage', 'Voltage (V)': 'Voltage',
        'Charge_Capacity(Ah)': 'Charge_Capacity',
        'Discharge_Capacity(Ah)': 'Discharge_Capacity',
        'Internal_Resistance(Ohm)': 'Internal_Resistance',
        'Step_Index': 'Step',
        'Temperature(C)': 'Temperature', 'Temperature (C)': 'Temperature',
        'Aux_Temperature(C)': 'Temperature'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    required = ['Time', 'Voltage', 'Current']
    for col in required:
        if col not in df.columns:
            for c in df.columns:
                if col.lower() in str(c).lower() and c not in df.columns.values:
                     df.rename(columns={c: col}, inplace=True)
                     break
    return df

def load_arbin_data(file_path, sheet_name):
    try:
        print(f"Reading {sheet_name}...")
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=6)
        print("Columns before std:", df.columns.tolist())
        df = standardize_columns(df)
        print("Columns after std:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error reading Arbin sheet {sheet_name}: {e}")
        return None

file_path = r"c:\Users\chenp\Documents\比赛\2026美赛\code\data\10_22_2014_PLN_7to12.xlsx"
sheet_name = 'Channel_1-007'

if os.path.exists(file_path):
    df = load_arbin_data(file_path, sheet_name)
    if df is not None:
        print("Head:")
        print(df.head())
else:
    print("File not found.")
