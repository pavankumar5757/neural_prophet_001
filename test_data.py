import pandas as pd
import sys

try:
    df = pd.read_csv('data.csv', skiprows=5, parse_dates=['Date'])
    print(f'Rows: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    print(f'First date: {df["Date"].iloc[0]}')
    print(f'Date type: {type(df["Date"].iloc[0])}')
    print(f'Target column: {df["Country_Demand"].iloc[0:5].tolist()}')
    print('Data loaded successfully!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

