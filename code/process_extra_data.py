
import pandas as pd
import numpy as np
import os
import sys
import warnings

# Add current directory to path to import the existing processor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import process_battery_data as base_processor

warnings.filterwarnings('ignore')

# Configuration
EXTRA_DATA_DIR = r"c:\Users\chenp\Documents\比赛\2026美赛\code\extra-data"
OUTPUT_DIR = r"c:\Users\chenp\Documents\比赛\2026美赛\code\processed_extra_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def standardize_and_convert_units(df):
    """
    Standardize columns and convert units (mV -> V, mA -> A) if detected.
    Also handles DateTime -> Seconds conversion.
    """
    # 1. Basic Cleaning of column names
    df.columns = df.columns.astype(str).str.strip()
    
    # 2. Map Column Names (English and Chinese support)
    rename_map = {}
    
    for col in df.columns:
        c_lower = col.lower()
        # Voltage
        if 'voltage' in c_lower or '电压' in c_lower:
            if '(mv)' in c_lower:
                rename_map[col] = 'Voltage_mV'
            elif '(v)' in c_lower:
                rename_map[col] = 'Voltage'
            elif c_lower == 'voltage' or c_lower == '电压':
                rename_map[col] = 'Voltage'
            else:
                 rename_map[col] = 'Voltage' # Fallback loose match
                
        # Current
        elif 'current' in c_lower or '电流' in c_lower:
            if '(ma)' in c_lower:
                rename_map[col] = 'Current_mA'
            elif '(a)' in c_lower:
                rename_map[col] = 'Current'
            elif c_lower == 'current' or c_lower == '电流':
                rename_map[col] = 'Current'
            else:
                 rename_map[col] = 'Current'
                
        # Temperature
        elif 'temp' in c_lower or '温度' in c_lower:
            rename_map[col] = 'Temperature'
            
        # Time
        elif 'time' in c_lower or 'date' in c_lower or '时间' in c_lower:
            rename_map[col] = 'Time'
            
        # Capacity
        elif ('cap' in c_lower or '容量' in c_lower) and ('ah' in c_lower or '安时' in c_lower or 'mah' in c_lower):
            if 'discharge' in c_lower or '放电' in c_lower:
                rename_map[col] = 'Discharge_Capacity'
            elif 'charge' in c_lower or '充电' in c_lower:
                rename_map[col] = 'Charge_Capacity'
            else:
                 rename_map[col] = 'Capacity'
                 
        # SOC
        elif 'soc' in c_lower or 'level' in c_lower or '电量' in c_lower:
             rename_map[col] = 'SOC'

    print(f"    Renaming columns: {rename_map}")
    df.rename(columns=rename_map, inplace=True)
    
    # Remove duplicates (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"    Columns after rename: {df.columns.tolist()}")
    
    # 3. Unit Conversion
    if 'Voltage_mV' in df.columns:
        df['Voltage'] = df['Voltage_mV'] / 1000.0
        df.drop(columns=['Voltage_mV'], inplace=True)
        print("    Converted Voltage mV -> V")
        
    if 'Current_mA' in df.columns:
        df['Current'] = df['Current_mA'] / 1000.0
        df.drop(columns=['Current_mA'], inplace=True)
        print("    Converted Current mA -> A")
        
    # 4. Time Conversion
    if 'Time' in df.columns:
        # Check if Time is datetime-like
        try:
            if not pd.api.types.is_numeric_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'])
                # Convert to seconds relative to start
                start_time = df['Time'].min()
                df['Time'] = (df['Time'] - start_time).dt.total_seconds()
                print("    Converted DateTime to Relative Seconds")
        except Exception as e:
            print(f"    Warning: Time conversion failed: {e}")

    return df

def process_waveform_data(df, file_name, sheet_name):
    """
    Process time-series waveform data (Voltage/Current vs Time).
    """
    print(f"  -> Processing as Waveform Data: {file_name} [{sheet_name}]")
    
    # Pre-check data validity
    if 'Voltage' not in df.columns or df['Voltage'].dropna().empty:
        print("    Error: Voltage column missing or empty.")
        return None, None
        
    # 1. Clean & Outliers
    print(f"    Raw Rows: {len(df)}")
    df_clean = base_processor.detect_and_remove_outliers(df)
    if df_clean.empty:
        print("    Warning: All data removed during outlier detection. Check Voltage/Current ranges.")
        print(f"      Voltage Range: {df['Voltage'].min()} - {df['Voltage'].max()}")
        return None, None
    print(f"    Clean Rows: {len(df_clean)}")

    # 2. Denoise
    df_denoised = base_processor.denoise_battery_data(df_clean)
    
    # 3. Resample
    try:
        df_resampled = base_processor.resample_data(df_denoised)
    except Exception as e:
        print(f"    Error in Resampling: {e}")
        return None, None
    
    # 4. Feature Extraction
    features = base_processor.extract_features(df_resampled)
    
    # Add Identity
    features['Source_File'] = file_name
    features['Source_Sheet'] = sheet_name
    
    return df_resampled, features

def main():
    print(f"Processing Extra Data from: {EXTRA_DATA_DIR}")
    
    files = [f for f in os.listdir(EXTRA_DATA_DIR) if f.endswith(('.xlsx', '.xls', '.csv'))]
    all_features = []
    
    for filename in files:
        file_path = os.path.join(EXTRA_DATA_DIR, filename)
        print(f"\nScanning: {filename}")
        
        try:
            # Determine file type and strategy
            is_waveform = False
            
            if filename.lower() == 'nasa_data1.xlsx':
                # NASA Data: Likely mixed (Summary + Waveform)
                xl = pd.ExcelFile(file_path)
                for sheet in xl.sheet_names:
                    print(f"  Inspecting Sheet: {sheet}")
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    df = standardize_and_convert_units(df)
                    
                    # Check if this sheet looks like waveform data
                    # Waveform needs Time, Voltage, Current
                    required_cols = {'Time', 'Voltage', 'Current'}
                    if required_cols.issubset(df.columns):
                        # Process as waveform
                        df_proc, feats = process_waveform_data(df, filename, sheet)
                        if df_proc is not None:
                            out_name = f"proc_{os.path.splitext(filename)[0]}_{sheet}.csv"
                            df_proc.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                            all_features.append(feats)
                    else:
                        print(f"  -> Skipping Waveform Flow (Missing {required_cols - set(df.columns)}). Saving as clean CSV.")
                        # Just save it as 'clean' data (standardized cols)
                        out_name = f"clean_{os.path.splitext(filename)[0]}_{sheet}.csv"
                        df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                        
            elif 'battery_data.xlsx' in filename.lower() and '2' not in filename: 
                # Realtime usage log (battery_data.xlsx) - strictly checks it's not the (2) one
                # Note: filename might be "battery_data.xlsx" or "battery_data (1).xlsx" etc
                # Readme says 'battery_data.xlsx' is the realtime one.
                # 'battery_data (2).xlsx' is RUL.
                
                df = pd.read_excel(file_path)
                df = standardize_and_convert_units(df)
                
                # It has Time, Voltage, Current (after conversion)
                if 'Voltage' in df.columns and 'Current' in df.columns:
                     df_proc, feats = process_waveform_data(df, filename, 'Main')
                     if df_proc is not None:
                        out_name = f"proc_{os.path.splitext(filename)[0]}.csv"
                        df_proc.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                        all_features.append(feats)
                     else:
                         # Fallback: Save clean data even if waveform processing failed
                         out_name = f"clean_{os.path.splitext(filename)[0]}.csv"
                         df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                         print(f"  -> Saved clean data fallback: {out_name}")

            else:
                # Other files (RUL, Users, etc.)
                # Just converting to CSV for consistency, maybe minimal cleaning (dropna)
                print("  -> Processing as Tabular/Summary Data (No Waveform Processing).")
                try:
                    xl = pd.ExcelFile(file_path)
                    for sheet in xl.sheet_names:
                        df = pd.read_excel(file_path, sheet_name=sheet)
                        # Maybe standardize columns just in case
                        df = standardize_and_convert_units(df)
                        
                        out_name = f"clean_{os.path.splitext(filename)[0]}_{sheet}.csv"
                        df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                except:
                    # Try CSV
                    try:
                        df = pd.read_csv(file_path)
                        df = standardize_and_convert_units(df)
                        out_name = f"clean_{os.path.splitext(filename)[0]}.csv"
                        df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                    except Exception as e:
                        print(f"  Error reading {filename}: {e}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            import traceback
            traceback.print_exc()

    # Save Summary Features
    if all_features:
        features_df = pd.DataFrame(all_features)
        summary_path = os.path.join(OUTPUT_DIR, "extra_data_features_summary.csv")
        features_df.to_csv(summary_path, index=False)
        print(f"\nFeature Extraction Complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
