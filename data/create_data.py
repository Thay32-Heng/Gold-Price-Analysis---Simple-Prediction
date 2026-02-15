import yfinance as yf 
from pathlib import Path

# create path for save data
root = Path().resolve().parent
data_dir = root / 'data'
data_dir.mkdir(exist_ok=True)

# download gold price data
print('Downloading gold price data...')
gold_data = yf.download('GC=F', start='2015-01-01', end='2026-02-14')

# save data to csv
csv_path = data_dir / 'gold_price_data.csv'
gold_data.to_csv(csv_path) 

print("Gold price data downloaded and saved to:", csv_path)