import pandas as pd

# Load dataset

df = pd.read_csv(
"data/processed/flowcast_features.csv",
low_memory=False
)

print("Dataset loaded successfully!")

print("\n========== ACCIDENT DATA ANALYSIS ==========")

# Accident count statistics

print("\nAccident count statistics:")
print(df["accident_count"].describe())

# Distribution of accident counts

print("\nAccident count distribution:")
print(df["accident_count"].value_counts().sort_index())

# Binary accident occurrence

df["accident_occurred"] = (
df["accident_count"] > 0
).astype(int)

print("\n========== ACCIDENT OCCURRENCE ==========")

print(df["accident_occurred"].value_counts())

print("\nAccident occurrence percentages:")
print(
df["accident_occurred"]
.value_counts(normalize=True)
.mul(100)
.round(2)
)

# Number of positive accident records

print(
"\nRecords with accidents:",
(df["accident_count"] > 0).sum()
)

print(
"Records without accidents:",
(df["accident_count"] == 0).sum()
)

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())
