import pandas as pd
from pybaseball import statcast

# Small test window first
start_date = "2025-04-01"
end_date = "2025-04-07"

print("Pulling Statcast data...")
df = statcast(start_date, end_date)

print("Rows pulled:", len(df))
print("Columns:", df.columns.tolist())

# Save raw sample so we can inspect it
df.to_csv("data/statcast_test_2025_04_01_to_04_07.csv", index=False)

print("Saved test CSV to data/statcast_test_2025_04_01_to_04_07.csv")