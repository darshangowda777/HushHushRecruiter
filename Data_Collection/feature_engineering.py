import pandas as pd
import numpy as np

# CONFIGURATION 
INPUT_CSV_FILE = 'CLEANED.csv'
OUTPUT_CSV_FILE = 'FEATURE_ENGINEERED1.csv'

#LOAD THE CLEANED DATASET
try:
    df = pd.read_csv(INPUT_CSV_FILE)
    print(f" Successfully loaded '{INPUT_CSV_FILE}' for Feature Engineering.\n")
except FileNotFoundError:
    print(f" Error: The file '{INPUT_CSV_FILE}' was not found.")
    exit()

#FEATURE ENGINEERING
print("Engineering New Features")

# We add a small number (epsilon) to the denominator to avoid division by zero.
epsilon = 1e-6 

# stars_per_repo (The Consistency Score)
df['stars_per_repo'] = (df['stars'] / (df['total_public_repos'] + epsilon)).round().astype(int)
print("  - Created 'stars_per_repo' (as a whole number)")

# issue_closure_rate (The Responsibility Score)
df['issue_closure_rate'] = df['owner_issues_closed'] / (df['owner_issues_opened'] + epsilon)
df['issue_closure_rate'] = np.clip(df['issue_closure_rate'], 0, 1) * 100 # As a percentage
df['issue_closure_rate'] = df['issue_closure_rate'].round().astype(int)
print("  - Created 'issue_closure_rate' (as a whole number percentage)")

# forks_per_repo (The "Real Utility" Score)
df['forks_per_repo'] = (df['forks'] / (df['total_public_repos'] + epsilon)).round().astype(int)
print("  - Created 'forks_per_repo' (as a whole number)")

# forks_to_stars_ratio (The "Developer's Developer" Score)
df['forks_to_stars_ratio'] = (df['forks'] / (df['stars'] + epsilon)).round(2)
print("  - Created 'forks_to_stars_ratio' (rounded to 2 decimals)")

#DISPLAY RESULTS AND SAVE
print("\nPreview of Data with New Features")
# Display the original columns and the new ones to see the results
display_columns = [
    'owner', 'stars', 'forks', 'stars_per_repo', 'forks_per_repo', 
    'issue_closure_rate', 'forks_to_stars_ratio'
]
print(df[display_columns].head())

# Save the enhanced dataframe to a new CSV file
df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"\n Feature engineering complete. The new data has been saved to '{OUTPUT_CSV_FILE}'.")

