import pandas as pd 
import yfinance as yf
from pathlib import Path

def clean_gold_data():
    # create path for save data 
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / 'data' / 'raw'
    proceessed_dir = root / 'data' / 'process'

    raw_dir.mkdir(parents = True, exist_ok=True)
    proceessed_dir.mkdir(parents = True, exist_ok=True)

    # download gold price data (if not already downloaded)
    raw_patch = raw_dir / 'gold_price_data.csv'
    if not raw_patch.exists():
        print('Downloading gold price data...')
        df = yf.download('GC=F', start='2015-01-01', end='2026-02-14')
        df.to_csv(raw_patch)

    # start data cleaning
    # yfinance produces multi-level headers; skip the "Ticker" and "Date" rows
    df = pd.read_csv(raw_patch, skiprows=[1, 2])

    # modified name columns to easy to use 
    df.columns = [
        col.lower().replace(' ', '_') for col in df.columns
    ]

    # rename 'price' column (which holds dates) to 'date'
    if 'price' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'price': 'date'})

    # convert Date column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # remove rows with missing values
    df = df.dropna()

    # remove duplicates
    df = df.drop_duplicates()

    # index by date (from oldest to newest)
    df = df.sort_values('date')

    # save to process folder
    output_path = proceessed_dir / 'cleaned_gold_price_data.csv'
    df.to_csv(output_path, index=False)

    print("Data cleaning completed. Cleaned data saved to:", output_path)

if __name__ == "__main__":
    clean_gold_data()