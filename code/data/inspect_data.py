
import os
import pandas as pd
import glob
import sys

def inspect_file(filepath, f_out):
    f_out.write(f"\n{'='*40}\n")
    f_out.write(f"File: {os.path.basename(filepath)}\n")
    f_out.write(f"{'='*40}\n")
    
    try:
        if filepath.endswith('.csv'):
            # Read first few lines as text to detect header
            with open(filepath, 'r') as f:
                head = [next(f) for _ in range(5)]
            f_out.write("First 5 lines (raw):\n")
            for line in head:
                f_out.write(line)
            f_out.write("\n")
            
            # Try reading with pandas
            try:
                df = pd.read_csv(filepath, nrows=5)
                f_out.write(f"\nPandas parse (header=0):\n")
                f_out.write(f"Columns: {list(df.columns)}\n")
                f_out.write(f"Dtypes:\n{df.dtypes}\n")
            except Exception as e:
                f_out.write(f"Pandas read_csv error: {e}\n")

        elif filepath.endswith('.xlsx'):
            try:
                xls = pd.ExcelFile(filepath, engine='openpyxl')
                for sheet_name in xls.sheet_names:
                    f_out.write(f"\nSheet: {sheet_name}\n")
                    df = pd.read_excel(xls, sheet_name=sheet_name, nrows=5)
                    f_out.write(f"Columns: {list(df.columns)}\n")
                    f_out.write(f"First row values: {df.iloc[0].tolist() if not df.empty else 'Empty'}\n")
            except Exception as e:
                f_out.write(f"Pandas read_excel error: {e}\n")

        elif filepath.endswith('.xls'):
            try:
                # Install xlrd if needed, but assuming env has it or we can try openpyxl (but openpyxl doesn't support xls)
                # If xlrd is missing, we might fail.
                df = pd.read_excel(filepath, nrows=5) 
                f_out.write(f"Columns: {list(df.columns)}\n")
            except Exception as e:
                f_out.write(f"Pandas read_excel (.xls) error: {e}\n")

    except Exception as e:
        f_out.write(f"General error inspecting file: {e}\n")

def main():
    directory = r'c:\Users\chenp\Documents\比赛\2026美赛\code'
    output_file = os.path.join(directory, 'data_summary_log.txt')
    
    extensions = ['*.csv', '*.xlsx', '*.xls']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    files.sort()
    
    pln_files = [f for f in files if "PLN" in os.path.basename(f) and "Number_SOC" not in os.path.basename(f)]
    other_files = [f for f in files if f not in pln_files]
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Process other files
        for f in other_files:
            inspect_file(f, f_out)
            
        # Process a sample of PLN files
        if pln_files:
            f_out.write(f"\n\n--- PLN Series Files ({len(pln_files)} files found) ---\n")
            f_out.write("Inspecting a sample to verify consistency.\n")
            samples = [pln_files[0]]
            if len(pln_files) > 1:
                samples.append(pln_files[len(pln_files)//2])
            if len(pln_files) > 2:
                samples.append(pln_files[-1])
            
            for f in samples:
                inspect_file(f, f_out)

if __name__ == "__main__":
    main()
