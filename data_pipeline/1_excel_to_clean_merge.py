import pandas as pd
import os
import glob

def excel_to_csv(input_dir="raw_excel", output_dir="raw_csv"):
    """
    Step 1: Converts all Excel files in the input directory to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))
    
    if not excel_files:
        print(f"No Excel files found in {input_dir}")
        return

    for i, file_path in enumerate(excel_files, start=1):
        print(f"Converting {file_path} to CSV...")
        try:
            try:
                # Try standard excel engine
                df = pd.read_excel(file_path)
            except ValueError:
                # PRGI files are often malformed HTML tables saved as .xls
                tables = pd.read_html(file_path, flavor="lxml")
                df = tables[0]
                
            # If the parser failed to map column headers due to broken HTML syntax, forcibly map them
            if len(df.columns) == 9:
                # Based on standard PRGI 9-column export specs
                df.columns = [
                    "Title Code", "Title Name (English)", "Hindi Title", 
                    "Register Serial No", "Regn. No", "Owner Name", 
                    "State", "Publication City/District", "Periodicity"
                ]

            output_csv = os.path.join(output_dir, f"file{i}.csv")
            df.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"Saved: {output_csv}")
        except Exception as e:
            print(f"Error converting {file_path}: {e}")

def merge_csvs(input_dir="raw_csv", output_file="combined_raw.csv"):
    """
    Step 2: Merges multiple CSV files, case-folds title columns, and removes duplicates.
    """
    csv_files = glob.glob(os.path.join(input_dir, "file*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape before deduplication: {combined_df.shape}")
    
    # Ensure necessary columns exist before cleaning
    if "Title Name (English)" not in combined_df.columns or "Hindi Title" not in combined_df.columns:
        print("Warning: Expected columns 'Title Name (English)' or 'Hindi Title' not found. Check CSV headers.")
    
    # Fill NAs to avoid errors during string operations
    combined_df["Title Name (English)"] = combined_df["Title Name (English)"].fillna("")
    combined_df["Hindi Title"] = combined_df["Hindi Title"].fillna("")
    
    # Create Case-Folded key for English Title
    combined_df["_title_name_casefolded"] = combined_df["Title Name (English)"].astype(str).str.lower().str.strip()
    
    # Drop duplicates
    # Duplicates defined by Exact Title Name (English), Hindi Title, or the Case-Folded English Name
    combined_df = combined_df.drop_duplicates(subset=["Title Name (English)", "Hindi Title", "_title_name_casefolded"])
    
    # Drop the temporary case-fold column before saving
    combined_df = combined_df.drop(columns=["_title_name_casefolded"])
    
    combined_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Combined shape after deduplication: {combined_df.shape}")
    print(f"Saved merged dataset to {output_file}")

if __name__ == "__main__":
    # Example execution
    # Place your 5 excel files into a 'raw_excel' folder in the same directory before running.
    excel_to_csv()
    merge_csvs(output_file="combined_raw.csv")
