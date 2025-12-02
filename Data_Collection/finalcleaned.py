import pandas as pd
INPUT_CSV_FILE = 'github_data_summarized1.csv'
OUTPUT_CSV_FILE = 'CLEANEDD.csv'
# LOAD THE DATASET 
try:
    df = pd.read_csv(INPUT_CSV_FILE)
    print(f"Successfully loaded '{INPUT_CSV_FILE}'.")
    print(f"Initial row count: {len(df)}")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_CSV_FILE}' was not found.")
    exit()

#  KEEP MOST POPULAR REPO PER OWNER 
if 'owner' in df.columns and 'stars' in df.columns:
    initial_rows = len(df)
# Sort by stars (and forks as tiebreaker) in descending order
    df = df.sort_values(by=['stars', 'forks'], ascending=[False, False])

 # Drop duplicates per owner → keep the repo with the highest stars/forks
    df = df.drop_duplicates(subset=['owner'], keep='first')

    final_rows = len(df)
    print(f"\nRemoved {initial_rows - final_rows} duplicate rows.")
    print(f"Row count after keeping most popular repo per owner: {final_rows}")
else:
    print("\nWarning: 'owner' or 'stars' column not found, skipping duplicate removal.")

#  REMOVE UNWANTED COLUMNS 
columns_to_drop = [
    'repository_name',
    'top_contrib_categories',
    'avg_contrib_lines_changed',
    'total_contributions_to_others'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"\nRemoved columns: {', '.join(columns_to_drop)}")

#  ROUND DAY-BASED METRICS TO WHOLE NUMBERS 
# Find any columns that contain 'days' in their name
day_columns = [col for col in df.columns if 'days' in col]

if day_columns:
    print(f"\nRounding day-based columns to whole numbers: {', '.join(day_columns)}")
    for col in day_columns:
        # Fill any missing values with 0, round to the nearest whole number, then convert to integer
        df[col] = df[col].fillna(0).round().astype(int)
else:
    print("\nNo day-based columns found to round.")    
# Convert any remaining float columns to integers
for col in df.select_dtypes(include=['float']).columns:
    df[col] = df[col].fillna(0).astype(int)
print("\n Preview of Final Cleaned Data ")
print(df.head())
# Save the cleaned dataframe to a new CSV file
df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"\n Cleaned data has been saved to '{OUTPUT_CSV_FILE}'.")

