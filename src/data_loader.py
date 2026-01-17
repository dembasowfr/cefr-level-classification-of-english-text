import kagglehub
import pandas as pd
import os
import glob

def download_cefr_dataset():
    """
    Downloads the CEFR dataset from KaggleHub.
    Returns the path to the dataset files.
    """
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("amontgomerie/cefr-levelled-english-texts")
    print(f"Dataset downloaded to: {path}")
    return path

def load_cefr_data(path):
    """
    Loads the CEFR dataset from the specified path.
    Assumes the dataset is a CSV file.
    Returns a pandas DataFrame.
    """
    # Find CSV files in the directory
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    
    # Assuming there's one main CSV file, or we concat them
    print(f"Found {len(csv_files)} CSV file(s): {csv_files}")
    
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)
        
    final_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(final_df)} rows.")
    return final_df

if __name__ == "__main__":
    # Test the loader
    path = download_cefr_dataset()
    df = load_cefr_data(path)
    print(df.head())
    print(df.info())
