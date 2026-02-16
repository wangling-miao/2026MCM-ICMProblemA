
import pandas as pd
import numpy as np
import os
import glob
import scipy.io

INPUT_DIR = r"c:\Users\chenp\Documents\比赛\2026美赛\code\find-data"
OUTPUT_DIR = r"c:\Users\chenp\Documents\比赛\2026美赛\code\processed_find_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_component_power():
    """Process 1.csv: Display and CPU power data"""
    print("\n--- Processing 1.csv (Component Power) ---")
    fpath = os.path.join(INPUT_DIR, "1.csv")
    if not os.path.exists(fpath):
        print("  1.csv not found.")
        return

    try:
        df = pd.read_csv(fpath)
        # Rename French columns (likely from an OLED power study)
        rename_map = {
            'RougeMesuré': 'Red_Pixel_Avg',
            'VertMesuré': 'Green_Pixel_Avg',
            'BleuMesuré': 'Blue_Pixel_Avg',
            'Rouge': 'Red_Pixel',
            'Vert': 'Green_Pixel',
            'Bleu': 'Blue_Pixel',
            'M_PL': 'Measured_Power_uW'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # CPU Columns check
        # Assuming 'CPU_ENERGY_UW' is instantaneous power or energy chunk
        # If it's power, convert to W
        if 'Measured_Power_uW' in df.columns:
            df['Total_Power_W'] = df['Measured_Power_uW'] / 1e6
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, "clean_component_power.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path} with {len(df)} rows.")
        
    except Exception as e:
        print(f"  Error processing 1.csv: {e}")

def process_user_behavior():
    """Process 2.tsv: User App Switching"""
    print("\n--- Processing 2.tsv (User Behavior) ---")
    fpath = os.path.join(INPUT_DIR, "2.tsv")
    if not os.path.exists(fpath):
        print("  2.tsv not found.")
        return

    try:
        df = pd.read_csv(fpath, sep='\t')
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, "clean_user_behavior.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path} with {len(df)} rows.")
    except Exception as e:
        print(f"  Error processing 2.tsv: {e}")

def process_hppc_data():
    """Process Folder 3: HPPC/Pulse Data from .mat files"""
    print("\n--- Processing Folder 3 (HPPC/Pulse Data) ---")
    base_path = os.path.join(INPUT_DIR, "3")
    
    # Recursively find .mat files
    mat_files = glob.glob(os.path.join(base_path, "**", "*.mat"), recursive=True)
    
    # Priority: Look for "Pulse" or "HPPC" in filename
    target_files = [f for f in mat_files if "Pulse" in f or "HPPC" in f or "Mixed" in f]
    
    if not target_files:
        print("  No explicit HPPC/Pulse .mat files found. Using first available .mat file as sample.")
        target_files = mat_files[:1]
        
    for mfile in target_files[:3]: # limit to 3 files
        try:
            print(f"  Extracting: {os.path.basename(mfile)}")
            data = scipy.io.loadmat(mfile)
            
            # Navigate struct (McMaster data usually has 'meas' struct)
            # Structure: data['meas'][0][0]['Voltage']...
            
            dfs = []
            
            # Heuristic to find data fields
            keys = [k for k in data.keys() if not k.startswith('__')]
            for k in keys:
                val = data[k]
                # Check for nested struct
                if val.dtype.names:
                    # It's a structure
                    names = val.dtype.names
                    if 'Voltage' in names and 'Current' in names and 'Time' in names:
                        # Extract
                        V = val['Voltage'][0][0].flatten()
                        I = val['Current'][0][0].flatten()
                        T = val['Time'][0][0].flatten()
                        
                        temp = np.nan
                        if 'Temperature' in names:
                             temp = val['Temperature'][0][0].flatten()
                        
                        df = pd.DataFrame({'Time': T, 'Voltage': V, 'Current': I})
                        if isinstance(temp, np.ndarray) and len(temp) == len(T):
                            df['Temperature'] = temp
                        
                        dfs.append(df)
            
            if dfs:
                full_df = pd.concat(dfs)
                out_name = "proc_hppc_" + os.path.basename(mfile).replace('.mat', '.csv')
                full_df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
                print(f"    Saved {out_name}")
            else:
                print("    Could not identify V/I/T structure in .mat file.")

        except Exception as e:
            print(f"  Error processing {os.path.basename(mfile)}: {e}")

def process_calendar_aging():
    """Process Folder 4: Calendar Aging"""
    print("\n--- Processing Folder 4 (Calendar Aging) ---")
    base_path = os.path.join(OUTPUT_DIR, "..", "find-data", "4") # normalize path
    
    # Structure: 4/Capacity_25C/3M/File...
    # We want to aggregate: Temp, Duration, Capacity
    
    records = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.csv', '.xls', '.xlsx')):
                # Infer metadata from path
                path_parts = root.split(os.sep)
                
                temp_str = "25C" # default
                duration_str = "0M"
                
                # Try to guess Temp
                for p in path_parts:
                    if "Capacity_" in p:
                        temp_str = p.replace("Capacity_", "")
                
                # Try to guess duration (folder name)
                parent = os.path.basename(root)
                if parent.endswith("M") or parent.endswith("W"):
                    duration_str = parent
                
                # Read file to get capacity
                fpath = os.path.join(root, file)
                try:
                    # Assumption: File contains a capacity value or discharge curve
                    if file.endswith('.csv'):
                         df = pd.read_csv(fpath)
                    else:
                         df = pd.read_excel(fpath)
                         
                    # Look for "Capacity" or "Ah"
                    cap = np.nan
                    col_map = [c for c in df.columns if "cap" in c.lower() or "ah" in c.lower()]
                    
                    if col_map:
                         cap = df[col_map[0]].max()
                    
                    records.append({
                        'Temp': temp_str,
                        'Duration': duration_str,
                        'Capacity': cap,
                        'File': file
                    })
                    
                except:
                    pass
    
    if records:
        df_cal = pd.DataFrame(records)
        out_path = os.path.join(OUTPUT_DIR, "calendar_aging_summary.csv")
        df_cal.to_csv(out_path, index=False)
        print(f"  Saved {out_path} with {len(df_cal)} records.")
    else:
        print("  No calendar aging data extracted.")

def main():
    process_component_power()
    process_user_behavior()
    process_hppc_data()
    process_calendar_aging()

if __name__ == "__main__":
    main()
