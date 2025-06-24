import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Contoh pembersihan data sederhana
    df = df.dropna()
    return df

def encode_labels(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

def save_preprocessed_data(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Buat folder jika belum ada
    df.to_csv(output_path, index=False)

def run_pipeline():
    input_path = 'E-Commerce_Shipping_raw/Train.csv'
    output_path = 'E-Commerce_Shipping_preprocessing/preprocessed_data.csv'  # simpan dalam folder preprocessing/

    print("Loading data...")
    df = load_data(input_path)

    print("Available columns:", df.columns.tolist())  # debug log

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding labels...")
    df = encode_labels(df, column='Reached.on.Time_Y.N')  # Kolom target binary

    print("Saving preprocessed data...")
    save_preprocessed_data(df, output_path)

    print("Preprocessing pipeline completed.")

if __name__ == '__main__':
    run_pipeline()
