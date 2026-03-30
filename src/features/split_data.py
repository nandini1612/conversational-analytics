import pandas as pd
from sklearn.model_selection import train_test_split

# STEP 1: Load dataset
df = pd.read_csv("../data/raw/synthetic_calls_v3_final.csv")

# STEP 2: Create stratification column
df['stratify_col'] = df['csat_range'] + "_" + df['issue_type']

# STEP 3: First split (70% train, 30% temp)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df['stratify_col'],
    random_state=42
)

# STEP 4: Second split (15% val, 15% test)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df['stratify_col'],
    random_state=42
)

# STEP 5: Remove helper column
for d in [train_df, val_df, test_df]:
    d.drop(columns=['stratify_col'], inplace=True)

# STEP 6: Save files
train_df.to_csv("../data/processed/train.csv", index=False)
val_df.to_csv("../data/processed/val.csv", index=False)
test_df.to_csv("../data/processed/test.csv", index=False)

# STEP 7: Print sizes
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))




# Load the generated CSVs
train_df = pd.read_csv("../data/processed/train.csv")
val_df = pd.read_csv("../data/processed/val.csv")
test_df = pd.read_csv("../data/processed/test.csv")

# Check the number of rows
print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))