# ============================================================
# FULL ANALYSIS SCRIPT (CLEANED)
# - Correlation Analysis (Pearson)
# - EDA for Final_Dataset_Raw.csv (2000–2019)
# ============================================================

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Config
# ============================================================
DATA_PATH = "Final_Dataset_Raw.csv"
START_YEAR = 2000
END_YEAR = 2019

NUM_COLS = [
    "Health_Expenditure",
    "GDP_per_capita",
    "Life_Expectancy",
    "Unemployment_Rate",
    "Death_Rate",
    "Gov_Effectiveness",
    "Population",
]

INCOME_ORDER = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income",
]

UNITS = {
    "Health_Expenditure": "Health Expenditure (USD)",
    "Life_Expectancy": "Life Expectancy (Years)",
    "Unemployment_Rate": "Unemployment Rate (%)",
    "Death_Rate": "Deaths per 1,000 People",
    "Gov_Effectiveness": "Governance Effectiveness Index",
    "Population": "Population (People)",
    "GDP_per_capita": "GDP per Capita (USD)",
}

# ============================================================
# Helpers
# ============================================================
def read_csv_dedup(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[:, ~df.columns.duplicated()].copy()

def categorize_strength(corr_matrix: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            r = float(corr_matrix.loc[c1, c2])

            if abs(r) >= 0.70:
                strength = "Strong"
            elif abs(r) >= 0.40:
                strength = "Moderate"
            else:
                strength = "Weak"

            rows.append([c1, c2, r, strength])

    return pd.DataFrame(rows, columns=["Variable 1", "Variable 2", "Correlation", "Strength"])

# ============================================================
# PART 1 — CORRELATION ANALYSIS
# ============================================================
print("\n================ CORRELATION ANALYSIS ================")

df_corr = read_csv_dedup(DATA_PATH)

missing_cols = [c for c in NUM_COLS if c not in df_corr.columns]
if missing_cols:
    raise ValueError(f"Missing required numeric columns in dataset: {missing_cols}")

df_numeric = df_corr[NUM_COLS].dropna()

corr_matrix = df_numeric.corr(method="pearson")
print("\n=== Pearson Correlation Matrix ===")
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.xticks(rotation=45, ha="right")
plt.title(f"Correlation Matrix of Features ({START_YEAR}–{END_YEAR})")
plt.tight_layout()
plt.show()

strength_df = categorize_strength(corr_matrix, NUM_COLS)
print("\n=== Categorized Correlation Strength ===")
print(strength_df)

print("\nSaved correlation outputs (2 files).")

# ============================================================
# PART 2 — EDA (2000–2019)
# ============================================================
print("\n================ EDA ON 2000–2019 DATASET ================")

df = read_csv_dedup(DATA_PATH)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)].copy()

missing_cols_eda = [c for c in NUM_COLS if c not in df.columns]
if missing_cols_eda:
    raise ValueError(f"Missing required EDA columns in dataset: {missing_cols_eda}")

# 1) Global average line plots
df_yearly = df.groupby("Year")[NUM_COLS].mean(numeric_only=True).reset_index()

year_ticks = list(np.arange(START_YEAR, END_YEAR + 1, 5))
if END_YEAR not in year_ticks:
    year_ticks.append(END_YEAR)
year_ticks = sorted(year_ticks)

for col in NUM_COLS:
    plt.figure(figsize=(10, 5))
    plt.plot(df_yearly["Year"], df_yearly[col], marker="o")
    plt.title(f"Average {col.replace('_', ' ')} ({START_YEAR}–{END_YEAR})")
    plt.xlabel("Year")
    plt.ylabel(UNITS.get(col, col))
    plt.xticks(year_ticks)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2) Unique countries by income group (ordered)
unique_counts = (
    df.groupby("Income group")["Country_ID"]
      .nunique()
      .reset_index(name="Unique Country")
)

unique_counts["Income group"] = pd.Categorical(
    unique_counts["Income group"],
    categories=INCOME_ORDER,
    ordered=True
)
unique_counts = unique_counts.sort_values("Income group")

plt.figure(figsize=(8, 5))
sns.barplot(
    data=unique_counts,
    x="Income group",
    y="Unique Country",
    color="#003366"
)
plt.title(f"Unique Countries by Income Group ({START_YEAR}–{END_YEAR})")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print("\n=== DataFrame Info ===")
print(df.info())

# 3) Missing values per variable (exclude IDs)
exclude_cols = ["Country_ID", "Country_Name", "Year"]
df_filtered = df.drop(columns=exclude_cols, errors="ignore")

missing_df = (
    df_filtered.isna()
    .sum()
    .reset_index()
    .rename(columns={"index": "Variable", 0: "Missing_Count"})
    .sort_values("Missing_Count", ascending=False)
)

plt.figure(figsize=(12, 6))
plt.bar(missing_df["Variable"], missing_df["Missing_Count"])
plt.title(f"Missing Values per Variable ({START_YEAR}–{END_YEAR})")
plt.xlabel("Variable")
plt.ylabel("Number of Missing Entries")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("\n=== Descriptive Statistics ===")
print(df_filtered.describe())

# 4) Top 15 countries with most missing values
df["Row_Missing"] = df.isna().sum(axis=1)

missing_by_country = (
    df.groupby("Country_ID")["Row_Missing"]
      .sum()
      .reset_index(name="Total_Missing")
)

missing_top15 = missing_by_country.sort_values("Total_Missing", ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=missing_top15,
    x="Country_ID",
    y="Total_Missing",
    color="#003366"  # fixes seaborn "palette without hue" warning
)
plt.title(f"Top 15 Countries with Highest Missingness ({START_YEAR}–{END_YEAR})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


