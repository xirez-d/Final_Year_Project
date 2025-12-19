import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# DATA PREPROCESSING (2000–2022)
# - Merge 7 indicators + income group
# - Save raw merged dataset          -> Final_Dataset_Raw_2022.csv
# - Filter countries (missingness rules on 2000–2019)
# - Impute numeric values (within-country, time-ordered)
# - Save imputed dataset             -> Final_Dataset_Imputed_2022.csv
# ============================================================

# ---------------------------
# Config
# ---------------------------
START_YEAR = 2000
END_YEAR = 2022
COVERAGE_END_YEAR = 2019  # missingness rule applies only up to this year

INCOME_PATH = "Income_Level.xlsx"
RAW_OUTPUT_PATH = "Final_Dataset_Raw_2022.csv"
IMPUTED_OUTPUT_PATH = "Final_Dataset_Imputed_2022.csv"

FILES = {
    "health": "Current Health Expenditure per capita (Current US$).csv",
    "life": "Life expectancy at birth, total (years).csv",
    "unemp": "unemployment-rate.csv",
    "death": "Death rate, crude (per 1,000 people).csv",
    "gov": "Government effectiveness.csv",
    "pop": "Population.csv",
    "gdp": "GDP per capita (current US$).csv",
}

KEYS = ["Country_ID", "Country_Name", "Year"]

NUMERIC_COLS = [
    "Health_Expenditure",
    "Life_Expectancy",
    "Unemployment_Rate",
    "Death_Rate",
    "Gov_Effectiveness",
    "Population",
    "GDP_per_capita",
]


# ---------------------------
# Helpers (shared style)
# ---------------------------
def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")

def standardize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def to_numeric_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

def filter_years(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    df = to_numeric_year(df)
    return df[(df["Year"] >= start) & (df["Year"] <= end)].copy()

def impute_numeric_series_ordered(s: pd.Series) -> pd.Series:
    """
    Deterministic within-country imputation (Year-ordered):
    - start missing -> backfill
    - end missing   -> forward fill
    - middle missing -> average neighbors if available, else nearest neighbor
    """
    s = s.copy()
    n = len(s)
    for i in range(n):
        if pd.isna(s.iat[i]):
            if i == 0:
                s.iat[i] = s.bfill().iat[i]
            elif i == n - 1:
                s.iat[i] = s.ffill().iat[i]
            else:
                prev_val = s.iat[i - 1]
                next_val = s.iat[i + 1]
                if not pd.isna(prev_val) and not pd.isna(next_val):
                    s.iat[i] = (prev_val + next_val) / 2
                elif not pd.isna(prev_val):
                    s.iat[i] = prev_val
                elif not pd.isna(next_val):
                    s.iat[i] = next_val
    return s


# ============================================================
# 1) Load Income Group table
# ============================================================
income_df = pd.read_excel(INCOME_PATH)
require_cols(income_df, ["Economy", "Income group"], "Income_Level.xlsx")
income_df = income_df[["Economy", "Income group"]].copy()
income_df["Economy"] = standardize_text(income_df["Economy"])


# ============================================================
# 2) Load source datasets
# ============================================================
df1 = pd.read_csv(FILES["health"])
df2 = pd.read_csv(FILES["life"])
df3 = pd.read_csv(FILES["unemp"])
df4 = pd.read_csv(FILES["death"])
df5 = pd.read_csv(FILES["gov"])
df6 = pd.read_csv(FILES["pop"])
df7 = pd.read_csv(FILES["gdp"])


# ============================================================
# 3) Standardize schemas (Country_ID, Country_Name, Year, Value)
# ============================================================
health = df1[["REF_AREA_ID", "REF_AREA_NAME", "TIME_PERIOD", "OBS_VALUE"]].rename(
    columns={
        "REF_AREA_ID": "Country_ID",
        "REF_AREA_NAME": "Country_Name",
        "TIME_PERIOD": "Year",
        "OBS_VALUE": "Health_Expenditure",
    }
)

life = df2[["REF_AREA_ID", "REF_AREA_NAME", "TIME_PERIOD", "OBS_VALUE"]].rename(
    columns={
        "REF_AREA_ID": "Country_ID",
        "REF_AREA_NAME": "Country_Name",
        "TIME_PERIOD": "Year",
        "OBS_VALUE": "Life_Expectancy",
    }
)

unemp_col = "Unemployment, total (% of total labor force) (modeled ILO estimate)"
require_cols(df3, ["Code", "Entity", "Year", unemp_col], "unemployment-rate.csv")
unemp = df3[["Code", "Entity", "Year", unemp_col]].rename(
    columns={
        "Code": "Country_ID",
        "Entity": "Country_Name",
        unemp_col: "Unemployment_Rate",
    }
)

death = df4[["REF_AREA_ID", "REF_AREA_NAME", "TIME_PERIOD", "OBS_VALUE"]].rename(
    columns={
        "REF_AREA_ID": "Country_ID",
        "REF_AREA_NAME": "Country_Name",
        "TIME_PERIOD": "Year",
        "OBS_VALUE": "Death_Rate",
    }
)

gov = df5[["REF_AREA", "REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]].rename(
    columns={
        "REF_AREA": "Country_ID",
        "REF_AREA_LABEL": "Country_Name",
        "TIME_PERIOD": "Year",
        "OBS_VALUE": "Gov_Effectiveness",
    }
)

pop = df6[["REF_AREA_ID", "REF_AREA_NAME", "TIME_PERIOD", "OBS_VALUE"]].rename(
    columns={
        "REF_AREA_ID": "Country_ID",
        "REF_AREA_NAME": "Country_Name",
        "TIME_PERIOD": "Year",
        "OBS_VALUE": "Population",
    }
)

gdp = df7[["REF_AREA_ID", "REF_AREA_NAME", "TIME_PERIOD", "OBS_VALUE"]].rename(
    columns={
        "REF_AREA_ID": "Country_ID",
        "REF_AREA_NAME": "Country_Name",
        "TIME_PERIOD": "Year",
        "OBS_VALUE": "GDP_per_capita",
    }
)

dfs = [health, life, unemp, death, gov, pop, gdp]


# ============================================================
# 4) Merge all on KEYS
# ============================================================
final_df = dfs[0].copy()
for d in dfs[1:]:
    final_df = pd.merge(final_df, d, on=KEYS, how="outer")

final_df["Country_Name"] = standardize_text(final_df["Country_Name"])


# ============================================================
# 5) Merge Income Group (Country_Name ↔ Economy)
# ============================================================
final_df = (
    pd.merge(final_df, income_df, left_on="Country_Name", right_on="Economy", how="left")
      .drop(columns=["Economy"], errors="ignore")
)


# ============================================================
# 6) Filter to 2000–2022 and sort
# ============================================================
final_df = filter_years(final_df, START_YEAR, END_YEAR)
final_df = final_df.sort_values(["Country_ID", "Year"]).reset_index(drop=True)


# ============================================================
# 7) Save RAW output (unchanged file name)
# ============================================================
final_df.to_csv(RAW_OUTPUT_PATH, index=False)

print(f"✅ Raw merged dataset saved | Shape: {final_df.shape}")
print(f"   → File: {RAW_OUTPUT_PATH}")
print(f"   Year range: {int(final_df['Year'].min())}–{int(final_df['Year'].max())}")
print(f"   Countries: {final_df['Country_ID'].nunique()}")


# ============================================================
# 8) Filter countries by missingness rules (2000–2019 only)
# ============================================================
indicator_cols = [c for c in final_df.columns if c not in KEYS]

def country_is_valid(g: pd.DataFrame) -> bool:
    yrs = set(g["Year"].dropna().unique())
    if not (START_YEAR in yrs and COVERAGE_END_YEAR in yrs):
        return False

    g_pre = g[(g["Year"] >= START_YEAR) & (g["Year"] <= COVERAGE_END_YEAR)]
    for col in indicator_cols:
        if g_pre[col].isna().sum() > 1:
            return False
    return True

valid_ids = [cid for cid, g in final_df.groupby("Country_ID", dropna=False) if country_is_valid(g)]

filtered_df = final_df[final_df["Country_ID"].isin(valid_ids)].copy()
filtered_df = filtered_df.sort_values(["Country_ID", "Year"]).reset_index(drop=True)

print("\nAfter country filtering:")
print("  Valid countries:", len(valid_ids))
print("  Rows:", filtered_df.shape[0])


# ============================================================
# 9) Impute on filtered countries (numeric only + income group mode)
# ============================================================
df = filtered_df.copy()
numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
has_income_group = "Income group" in df.columns

print("\nMissing values BEFORE imputation:")
print(df[numeric_cols].isna().sum())

def impute_country_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Year").copy()

    for col in numeric_cols:
        g[col] = impute_numeric_series_ordered(g[col])

    if has_income_group and g["Income group"].isna().any():
        mode_vals = g["Income group"].mode()
        if not mode_vals.empty:
            g["Income group"] = g["Income group"].fillna(mode_vals.iloc[0])

    return g

df_imputed = (
    df.groupby(["Country_ID", "Country_Name"], as_index=False, group_keys=False)
      .apply(impute_country_group)
      .reset_index(drop=True)
)

print("\nMissing values AFTER imputation:")
print(df_imputed[numeric_cols].isna().sum())

df_imputed = df_imputed.sort_values(["Country_ID", "Year"]).reset_index(drop=True)


# ============================================================
# 10) Save IMPUTED output (unchanged file name)
# ============================================================
df_imputed.to_csv(IMPUTED_OUTPUT_PATH, index=False)

print("------------------------------------------------------")
print("✅ Year filtering, country filtering, and imputation completed")
print(f"Saved raw dataset      → {RAW_OUTPUT_PATH}")
print(f"Saved imputed dataset  → {IMPUTED_OUTPUT_PATH}")
print("======================================================")
