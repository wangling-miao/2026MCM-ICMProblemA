
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import signal
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = r"c:\Users\chenp\Documents\比赛\2026美赛\code\data"
OUTPUT_DIR = r"c:\Users\chenp\Documents\比赛\2026美赛\code\processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Step 1: Data Loading & Structuring
# -------------------------------------------------------------------------

def standardize_columns(df):
    """
    Standardize column names to a common format.
    """
    # Ensure all column names are strings
    df.columns = df.columns.astype(str)
    
    # Remove duplicate columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()]

    # Common variations map to standard names
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
    
    # Ensure required columns exist
    required = ['Time', 'Voltage', 'Current']
    for col in required:
        if col not in df.columns:
            # Try to find a column that looks like it (case insensitive)
            for c in df.columns:
                if col.lower() in c.lower() and c not in df.columns.values:
                     # Check if renaming would create a duplicate
                     if col in df.columns:
                         continue
                     df.rename(columns={c: col}, inplace=True)
                     break
    
    # Final deduplication just in case
    df = df.loc[:, ~df.columns.duplicated()]

    return df

def load_arbin_data(file_path, sheet_name):
    """
    Load Arbin test data (Excel), auto-detecting header row.
    """
    try:
        # First read a few rows to find the header
        preview = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=20)
        
        header_idx = 0
        found_header = False
        
        # Search for a row containing key terms
        for idx, row in preview.iterrows():
            row_str = row.astype(str).str.lower().tolist()
            # Check for presence of 'voltage' and 'current' in the row
            # Handle cases where values might not be strings
            try:
                if any('voltage' in str(x) for x in row_str) and any('current' in str(x) for x in row_str):
                    header_idx = idx
                    found_header = True
                    break
            except:
                continue
        
        if not found_header and header_idx == 0:
             # Fallback: maybe the user was right about metadata being exactly 6 lines for some files?
             # But if row 0 was data, checking row 6 directly is risky.
             # If we didn't find keywords, we assume 0 or try 6 if 0 looks like pure numbers?
             # Let's trust logic: if keywords found, use that. If not, use 0 to capture basic CSV-like excel.
             pass

        print(f"  -> Header detected at row {header_idx}")
        
        # Load full data with detected header
        # Note: skiprows is inclusive of the rows BEFORE header. 
        # If header_idx is 0, skiprows=0 (default).
        # If header_idx is 6, we want to skip 0-5, so header=header_idx usually works if we use header argument,
        # but pd.read_excel 'header' param expects 0-indexed row number of header.
        
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
        df = standardize_columns(df)
        return df
    except Exception as e:
        print(f"Error reading Arbin sheet {sheet_name}: {e}")
        return None

def load_csv_data(file_path):
    """
    Load CSV data.
    """
    try:
        # Attempt to detect if metadata lines exist.
        # Simple heuristic: read first few lines, find header.
        # But user didn't specify for CSV. Let's try standard read first.
        df = pd.read_csv(file_path)
        
        # If the first column is not numeric immediately, it might have metadata.
        # For now, assume standard CSV headers.
        df = standardize_columns(df)
        return df
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        return None

# -------------------------------------------------------------------------
# Step 2: Outlier Detection & Removal
# -------------------------------------------------------------------------

def detect_and_remove_outliers(df):
    """
    Multi-stage outlier detection.
    """
    original_len = len(df)
    
    # 1. Physical Constraints
    # Note: Handling missing columns gracefully
    conditions = []
    
    if 'Voltage' in df.columns:
        conditions.append((df['Voltage'] >= 1.5) & (df['Voltage'] <= 5.0))
    
    if 'Current' in df.columns:
        conditions.append((df['Current'] > -20) & (df['Current'] < 20))
    
    if 'Temperature' in df.columns:
        conditions.append((df['Temperature'] >= -20) & (df['Temperature'] <= 60)) # Per paper specs
    
    # Combine conditions
    mask = np.ones(len(df), dtype=bool)
    for cond in conditions:
        mask = mask & cond
    
    df_clean = df[mask].copy()
    
    # 2. Statistical (IQR) for Voltage and Current
    # Only applying if enough data points
    if len(df_clean) > 100:
        for col in ['Voltage', 'Current']:
            if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
                Q1 = df_clean[col].quantile(0.01) # 1st percentile to avoid over-clipping
                Q3 = df_clean[col].quantile(0.99) # 99th percentile
                IQR = Q3 - Q1
                # Using a very loose bound to remove only extreme outliers
                lower_bound = Q1 - 5 * IQR 
                upper_bound = Q3 + 5 * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
    
    # 3. Time Continuity (Simple deduplication and basic jump check)
    if 'Time' in df_clean.columns:
        df_clean = df_clean.sort_values('Time')
        df_clean.drop_duplicates(subset=['Time'], inplace=True)
        
        # Basic time jump filter (User-specified)
        df_clean['Time_diff'] = df_clean['Time'].diff()
        df_clean = df_clean[(df_clean['Time_diff'] < 100) | (df_clean['Time_diff'].isna())]
        df_clean.drop('Time_diff', axis=1, inplace=True)

    # 4. Critical Strategy for Resting Data (Selective Preservation)
    # "Removed: Redundant resting data lasting hours... Preserved: Short-term relaxation periods"
    if 'Current' in df_clean.columns and 'Time' in df_clean.columns:
        # Define 'Rest' as very low current
        rest_mask = df_clean['Current'].abs() < 0.05 # < 50mA
        
        # If no rest data, skip
        if rest_mask.any():
            # Identify discontinuous segments of rest
            # We assign a unique ID to each consecutive block of rest
            # Change in mask value indicates start/end of block
            block_id = (rest_mask != rest_mask.shift()).cumsum()
            
            # Filter only rest blocks
            rest_blocks = df_clean[rest_mask].groupby(block_id)
            
            indices_to_drop = []
            
            for _, block in rest_blocks:
                # If block is long
                start_time = block['Time'].iloc[0]
                end_time = block['Time'].iloc[-1]
                duration = end_time - start_time
                
                # If rest lasts longer than 30 minutes (1800s), trim it
                # Keep first 15 mins (900s) for relaxation analysis
                if duration > 1800:
                    cutoff_time = start_time + 900
                    # Find indices in this block where time > cutoff_time
                    drop_rows = block[block['Time'] > cutoff_time].index
                    indices_to_drop.extend(drop_rows)
            
            if indices_to_drop:
                original_count = len(df_clean)
                df_clean.drop(indices_to_drop, inplace=True)
                print(f"  - Selective Preservation: Removed {len(indices_to_drop)} redundant resting points.")

    removed = original_len - len(df_clean)
    if original_len > 0:
        percent = 100 * removed / original_len
        print(f"  - Outliers/Rest removed: {removed} ({percent:.2f}%)")
    
    return df_clean

# -------------------------------------------------------------------------
# Step 3: Noise Filtering (Savitzky-Golay)
# -------------------------------------------------------------------------

def denoise_battery_data(df, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter.
    """
    df_denoised = df.copy()
    
    # Adjust window_length if data is too short
    if len(df) < window_length:
        window_length = len(df) // 2 * 2 + 1 # Make odd and smaller
        if window_length < polyorder + 2:
             # Too short to filter
             return df_denoised

    for col in ['Voltage', 'Current', 'Temperature']:
        if col in df.columns:
             try:
                df_denoised[f'{col}_smooth'] = savgol_filter(
                    df[col], 
                    window_length=window_length, 
                    polyorder=polyorder
                )
             except Exception as e:
                 print(f"  - Warning: SG filter failed for {col}: {e}")
                 df_denoised[f'{col}_smooth'] = df[col] # Fallback
    
    return df_denoised

# -------------------------------------------------------------------------
# Step 4: Resampling & Alignment
# -------------------------------------------------------------------------

def resample_data(df, target_frequency='1s'):
    """
    Resample to fixed frequency.
    """
    if 'Time' not in df.columns or df.empty:
        return df

    # Create Datetime index
    # Note: 'Time' is usually seconds from start.
    # We use a dummy start date to leverage pandas time-series tools.
    df['Datetime'] = pd.to_datetime(df['Time'], unit='s', origin='unix')
    df.set_index('Datetime', inplace=True)
    
    # Resample
    # Only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_resampled = df[numeric_cols].resample(target_frequency).mean().interpolate(method='linear')
    
    # Reset Time
    # Calculate seconds from the first index
    if not df_resampled.empty:
        start_time = df_resampled.index[0]
        df_resampled['Time'] = (df_resampled.index - start_time).total_seconds()
    
    return df_resampled.reset_index(drop=True)

# -------------------------------------------------------------------------
# Step 5: Feature Extraction
# -------------------------------------------------------------------------

def calculate_discharge_capacity_by_integration(df):
    """
    Calculate total discharge capacity using trapezoidal integration (Equation 3).
    Q_k = ∫ I(t) dt
    """
    if 'Current' not in df.columns or 'Time' not in df.columns:
        return np.nan
    
    try:
        # Filter discharge periods (positive current convention: discharge > 0)
        discharge_mask = df['Current'] > 0
        if not discharge_mask.any():
            # Try negative convention (discharge < 0)
            discharge_mask = df['Current'] < 0
            current_sign = -1
        else:
            current_sign = 1
        
        if not discharge_mask.any():
            return np.nan
        
        # Get discharge segments
        time = df.loc[discharge_mask, 'Time'].values
        current = df.loc[discharge_mask, 'Current'].values * current_sign
        
        if len(time) < 2:
            return np.nan
        
        # Trapezoidal integration: Q = ∫I(t)dt
        # Convert to Ah (time in seconds, current in A)
        capacity_ah = np.trapz(current, time) / 3600.0
        
        return abs(capacity_ah)
    except Exception as e:
        print(f"    Warning: Capacity integration failed: {e}")
        return np.nan


def calculate_soc_profile(df, total_capacity):
    """
    Calculate State of Charge (SOC) profile.
    SOC = remaining_capacity / total_capacity
    """
    if total_capacity is None or np.isnan(total_capacity) or total_capacity <= 0:
        return None
    
    if 'Discharge_Capacity' in df.columns:
        # Use existing discharge capacity column
        soc = 1.0 - (df['Discharge_Capacity'] / total_capacity)
        return soc.clip(0, 1)
    elif 'Current' in df.columns and 'Time' in df.columns:
        # Calculate from current integration
        try:
            current = df['Current'].values
            time = df['Time'].values
            
            # Cumulative capacity discharged
            capacity_discharged = np.zeros(len(df))
            for i in range(1, len(df)):
                dt = time[i] - time[i-1]
                if dt > 0 and dt < 100:  # Avoid large time jumps
                    # Trapezoidal integration
                    dq = (current[i] + current[i-1]) / 2 * dt / 3600.0
                    if dq > 0:  # Only count discharge
                        capacity_discharged[i] = capacity_discharged[i-1] + dq
                    else:
                        capacity_discharged[i] = capacity_discharged[i-1]
                else:
                    capacity_discharged[i] = capacity_discharged[i-1]
            
            soc = 1.0 - (capacity_discharged / total_capacity)
            return pd.Series(soc).clip(0, 1)
        except Exception as e:
            print(f"    Warning: SOC calculation failed: {e}")
            return None
    
    return None


def extract_resistance_temperature_profile(df, soc_series):
    """
    Extract temperature-dependent internal resistance profile R(SOC, T).
    Returns features characterizing the resistance behavior.
    """
    features = {}
    
    try:
        if 'Internal_Resistance' not in df.columns or soc_series is None:
            return {}
        
        # Create a temporary dataframe with SOC, Temperature, and Resistance
        temp_df = pd.DataFrame({
            'SOC': soc_series,
            'Resistance': df['Internal_Resistance']
        })
        
        if 'Temperature' in df.columns:
            temp_df['Temperature'] = df['Temperature']
        else:
            temp_df['Temperature'] = 25.0  # Assume room temperature
        
        # Remove NaN values
        temp_df = temp_df.dropna()
        
        if len(temp_df) < 10:
            return {}
        
        # Extract key features
        # 1. Resistance at different SOC levels
        soc_bins = [0.0, 0.2, 0.5, 0.8, 1.0]
        for i in range(len(soc_bins) - 1):
            soc_low, soc_high = soc_bins[i], soc_bins[i+1]
            mask = (temp_df['SOC'] >= soc_low) & (temp_df['SOC'] < soc_high)
            if mask.sum() > 0:
                features[f'R_SOC_{int(soc_low*100)}_{int(soc_high*100)}'] = temp_df.loc[mask, 'Resistance'].mean()
        
        # 2. Resistance at different temperatures
        temp_bins = [-20, 0, 25, 60]
        for i in range(len(temp_bins) - 1):
            temp_low, temp_high = temp_bins[i], temp_bins[i+1]
            mask = (temp_df['Temperature'] >= temp_low) & (temp_df['Temperature'] < temp_high)
            if mask.sum() > 0:
                features[f'R_Temp_{temp_low}_{temp_high}'] = temp_df.loc[mask, 'Resistance'].mean()
        
        # 3. Low SOC resistance rise (critical for "voltage collapse")
        low_soc_mask = temp_df['SOC'] < 0.2
        mid_soc_mask = (temp_df['SOC'] >= 0.4) & (temp_df['SOC'] <= 0.6)
        
        if low_soc_mask.sum() > 0 and mid_soc_mask.sum() > 0:
            r_low = temp_df.loc[low_soc_mask, 'Resistance'].mean()
            r_mid = temp_df.loc[mid_soc_mask, 'Resistance'].mean()
            features['R_increase_ratio_low_SOC'] = r_low / r_mid if r_mid > 0 else np.nan
        
        # 4. Temperature sensitivity (simplified Arrhenius check)
        if 'Temperature' in df.columns:
            cold_mask = temp_df['Temperature'] < 10
            warm_mask = temp_df['Temperature'] > 20
            
            if cold_mask.sum() > 0 and warm_mask.sum() > 0:
                r_cold = temp_df.loc[cold_mask, 'Resistance'].mean()
                r_warm = temp_df.loc[warm_mask, 'Resistance'].mean()
                features['R_increase_ratio_cold'] = r_cold / r_warm if r_warm > 0 else np.nan
        
    except Exception as e:
        print(f"    Warning: Resistance profile extraction failed: {e}")
    
    return features


def extract_features(df):
    """
    Extract battery features (Enhanced with integration and R(SOC,T) profile).
    Implements Equations 3 and 4 from the paper.
    """
    features = {}
    if df.empty:
        return features

    # ====================================================================
    # 1. Capacity Features (Equation 3: Q_k = ∫I(t)dt)
    # ====================================================================
    try:
        # Try to get discharge capacity from existing column first
        if 'Discharge_Capacity' in df.columns and pd.api.types.is_numeric_dtype(df['Discharge_Capacity']):
            cap = df['Discharge_Capacity'].max()
        else:
            # Calculate by integration if column doesn't exist
            cap = calculate_discharge_capacity_by_integration(df)
            
        # SANITY CHECK: Single cell capacity shouldn't exceed ~10 Ah for this dataset
        # If it does, it's likely a cumulative multi-cycle file or wrong unit.
        if cap is not None and cap > 10.0:
            print(f"    Warning: Detected capacity {cap:.2f} Ah > 10Ah. Marking as NaN (Multi-cycle/Cumulative).")
            features['discharge_capacity'] = np.nan
        else:
            features['discharge_capacity'] = cap
            
    except Exception as e:
        print(f"    Warning: Discharge capacity extraction failed: {e}")
        features['discharge_capacity'] = np.nan
        
    try:
        if 'Charge_Capacity' in df.columns and pd.api.types.is_numeric_dtype(df['Charge_Capacity']):
            features['charge_capacity'] = df['Charge_Capacity'].max()
        else:
            features['charge_capacity'] = np.nan
    except:
        features['charge_capacity'] = np.nan

    try:
        if features.get('charge_capacity', 0) > 0 and not np.isnan(features.get('charge_capacity', np.nan)):
            features['coulombic_efficiency'] = features['discharge_capacity'] / features['charge_capacity']
        else:
            features['coulombic_efficiency'] = np.nan
    except:
        features['coulombic_efficiency'] = np.nan
    
    # ====================================================================
    # 2. Voltage Features
    # ====================================================================
    try:
        if 'Voltage' in df.columns and pd.api.types.is_numeric_dtype(df['Voltage']):
            features['avg_voltage'] = df['Voltage'].mean()
            features['voltage_std'] = df['Voltage'].std()
            features['voltage_range'] = df['Voltage'].max() - df['Voltage'].min()
            features['min_voltage'] = df['Voltage'].min()
            features['max_voltage'] = df['Voltage'].max()
        else:
            features['avg_voltage'] = np.nan
            features['voltage_std'] = np.nan
            features['voltage_range'] = np.nan
            features['min_voltage'] = np.nan
            features['max_voltage'] = np.nan
    except:
        features['avg_voltage'] = np.nan
        features['voltage_std'] = np.nan
        features['voltage_range'] = np.nan
        features['min_voltage'] = np.nan
        features['max_voltage'] = np.nan
    
    # ====================================================================
    # 3. Calculate SOC Profile
    # ====================================================================
    total_capacity = features.get('discharge_capacity', None)
    soc_series = calculate_soc_profile(df, total_capacity)
    
    # ====================================================================
    # 4. Internal Resistance Features (Equation 4: R(SOC, T))
    # ====================================================================
    try:
        if 'Internal_Resistance' in df.columns and pd.api.types.is_numeric_dtype(df['Internal_Resistance']):
            features['avg_resistance'] = df['Internal_Resistance'].mean()
            features['min_resistance'] = df['Internal_Resistance'].min()
            features['max_resistance'] = df['Internal_Resistance'].max()
            features['resistance_std'] = df['Internal_Resistance'].std()
            
            # Extract temperature-dependent resistance profile
            resistance_profile_features = extract_resistance_temperature_profile(df, soc_series)
            features.update(resistance_profile_features)
        else:
            features['avg_resistance'] = np.nan
            features['min_resistance'] = np.nan
            features['max_resistance'] = np.nan
            features['resistance_std'] = np.nan
    except Exception as e:
        print(f"    Warning: Resistance feature extraction failed: {e}")
        features['avg_resistance'] = np.nan
        features['min_resistance'] = np.nan
        features['max_resistance'] = np.nan
        features['resistance_std'] = np.nan
    
    # ====================================================================
    # 5. Temperature Features
    # ====================================================================
    try:
        if 'Temperature' in df.columns and pd.api.types.is_numeric_dtype(df['Temperature']):
            features['max_temperature'] = df['Temperature'].max()
            features['min_temperature'] = df['Temperature'].min()
            features['avg_temperature'] = df['Temperature'].mean()
            features['temperature_std'] = df['Temperature'].std()
        else:
            features['max_temperature'] = np.nan
            features['min_temperature'] = np.nan
            features['avg_temperature'] = np.nan
            features['temperature_std'] = np.nan
    except:
        features['max_temperature'] = np.nan
        features['min_temperature'] = np.nan
        features['avg_temperature'] = np.nan
        features['temperature_std'] = np.nan
        
    return features

# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

import traceback

def validate_preprocessing(df_original, df_processed, save_path):
    """
    Visualize preprocessing results.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Check if smooth columns exist
        v_smooth = 'Voltage_smooth' if 'Voltage_smooth' in df_processed.columns else 'Voltage'
        c_smooth = 'Current_smooth' if 'Current_smooth' in df_processed.columns else 'Current'

        # Voltage
        if 'Voltage' in df_original.columns and v_smooth in df_processed.columns:
            axes[0, 0].plot(df_original['Time'].values, df_original['Voltage'].values, alpha=0.5, label='Original')
            axes[0, 0].plot(df_processed['Time'].values, df_processed[v_smooth].values, label='Processed', linewidth=2)
            axes[0, 0].set_title('Voltage Signal')
            axes[0, 0].legend()
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Voltage (V)')

        # Current
        if 'Current' in df_original.columns and c_smooth in df_processed.columns:
            axes[0, 1].plot(df_original['Time'].values, df_original['Current'].values, alpha=0.5, label='Original')
            axes[0, 1].plot(df_processed['Time'].values, df_processed[c_smooth].values, label='Processed', linewidth=2)
            axes[0, 1].set_title('Current Signal')
            axes[0, 1].legend()
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Current (A)')
        
        # PSD (Voltage)
        if 'Voltage' in df_original.columns:
            f_orig, Pxx_orig = signal.welch(df_original['Voltage'].dropna().values, fs=1.0)
            f_proc, Pxx_proc = signal.welch(df_processed[v_smooth].dropna().values, fs=1.0)
            axes[1, 0].semilogy(f_orig, Pxx_orig, label='Original')
            axes[1, 0].semilogy(f_proc, Pxx_proc, label='Processed')
            axes[1, 0].set_title('Power Spectral Density (Voltage)')
            axes[1, 0].legend()
        
        # Histogram
        if 'Voltage' in df_original.columns:
            axes[1, 1].hist(df_original['Voltage'].dropna().values, bins=50, alpha=0.5, label='Original')
            axes[1, 1].hist(df_processed[v_smooth].dropna().values, bins=50, alpha=0.5, label='Processed')
            axes[1, 1].set_title('Voltage Distribution')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  - Validation plot saved to {save_path}")
    except Exception as e:
        print(f"Error checking validation: {e}")
        print(traceback.format_exc())
        # Debug info
        try:
             print(f"    df_original columns: {df_original.columns.tolist()}")
             print(f"    df_processed columns: {df_processed.columns.tolist()}")
        except:
            pass

# -------------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------------

def process_file(file_path):
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    filename = os.path.basename(file_path)
    base_name, ext = os.path.splitext(filename)
    
    processed_dfs = []
    
    if ext.lower() == '.xlsx' or ext.lower() == '.xls':
        try:
            xl = pd.ExcelFile(file_path)
            # Filter for sheets that look like data (Channel or Sheet)
            # Some files might just have "Channel_...", others might be different.
            # If no sheets match "Channel", process all.
            sheets = [s for s in xl.sheet_names if "Channel" in s]
            if not sheets:
                sheets = xl.sheet_names 
                # If too many sheets (e.g. > 20), maybe only take first few? 
                # User said "process all".
            
            print(f"  Found {len(sheets)} sheets.")
            
            for sheet in sheets:
                print(f"  -> Sheet: {sheet}")
                df = load_arbin_data(file_path, sheet)
                if df is not None and not df.empty:
                    # Run Pipeline
                    df_clean = detect_and_remove_outliers(df)
                    df_denoised = denoise_battery_data(df_clean)
                    df_resampled = resample_data(df_denoised)
                    features = extract_features(df_resampled)
                    
                    # Save Validation Plot (for the first sheet of each file only, to save space/time)
                    if not processed_dfs: 
                        val_path = os.path.join(OUTPUT_DIR, f"val_{base_name}_{sheet}.png")
                        validate_preprocessing(df, df_denoised, val_path)
                    
                    # Save Processed Data
                    out_name = f"proc_{base_name}_{sheet}.csv"
                    out_path = os.path.join(OUTPUT_DIR, out_name)
                    df_resampled.to_csv(out_path, index=False)
                    
                    processed_dfs.append({
                        'file': filename,
                        'sheet': sheet,
                        'output_file': out_name,
                        **features
                    })
                    
        except Exception as e:
            print(f"Error processing Excel file {filename}: {e}")
            
    elif ext.lower() == '.csv':
        try:
            df = load_csv_data(file_path)
            if df is not None and not df.empty:
                # Run Pipeline
                df_clean = detect_and_remove_outliers(df)
                df_denoised = denoise_battery_data(df_clean)
                df_resampled = resample_data(df_denoised)
                features = extract_features(df_resampled)
                
                # Validation Plot
                val_path = os.path.join(OUTPUT_DIR, f"val_{base_name}.png")
                validate_preprocessing(df, df_denoised, val_path)
                
                # Save Processed Data
                out_name = f"proc_{base_name}.csv"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                df_resampled.to_csv(out_path, index=False)
                
                processed_dfs.append({
                    'file': filename,
                    'sheet': 'N/A',
                    'output_file': out_name,
                    **features
                })
        except Exception as e:
            print(f"Error processing CSV file {filename}: {e}")

    return processed_dfs

def main():
    print("Starting Data Preprocessing Pipeline...")
    all_features = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.xlsx', '.xls', '.csv'))]
    
    for f in files:
        file_path = os.path.join(DATA_DIR, f)
        results = process_file(file_path)
        all_features.extend(results)
        
    # Save Summary Features
    if all_features:
        features_df = pd.DataFrame(all_features)
        summary_path = os.path.join(OUTPUT_DIR, "all_files_features_summary.csv")
        features_df.to_csv(summary_path, index=False)
        print(f"\nProcessing Complete. Summary saved to {summary_path}")
    else:
        print("\nNo data processed.")

if __name__ == "__main__":
    main()
