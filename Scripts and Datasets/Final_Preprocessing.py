import pandas as pd
import numpy as np

# ============================================================
# DATA PREPROCESSING (2000–2019)
# - Merge 7 indicators + income group
# - Save raw merged dataset          -> Final_Dataset_Raw.csv
# - Filter countries (missingness rules on 2000–2019)
# - Save filtered dataset            -> Final_Dataset_Filtered.csv
# - Impute numeric values (within-country, full 2000–2019 index)
# - Save imputed dataset             -> Final_Dataset_Imputed.csv
# ============================================================

# ---------------------------
# Config
# ---------------------------
START_YEAR = 2000
END_YEAR = 2019

INCOME_PATH = "Income_Level.xlsx"

RAW_OUTPUT_PATH = "Final_Dataset_Raw.csv"
FILTERED_OUTPUT_PATH = "Final_Dataset_Filtered.csv"
IMPUTED_OUTPUT_PATH = "Final_Dataset_Imputed.csv"

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


# ---------------------------
# Helpers (match 2000–2022 style)
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

def impute_series_linear_ffill_bfill(s: pd.Series) -> pd.Series:
    """
    Same imputation logic as your original 2000–2019 script:
    linear interpolation + ffill/bfill, deterministic.
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.ffill().bfill()
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
# 3) Standardize schemas
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
# 6) Filter to 2000–2019 and save RAW output
# ============================================================
final_df = filter_years(final_df, START_YEAR, END_YEAR)
final_df = final_df.sort_values(["Country_ID", "Year"]).reset_index(drop=True)

final_df.to_csv(RAW_OUTPUT_PATH, index=False)

print(f"✅ Raw merged dataset saved | Shape: {final_df.shape}")
print(f"   → File: {RAW_OUTPUT_PATH}")
print(f"   Year range: {int(final_df['Year'].min())}–{int(final_df['Year'].max())}")
print(f"   Countries: {final_df['Country_ID'].nunique()}")


# ============================================================
# 7) Filter countries by missingness rules (2000–2019)
# ============================================================
indicator_cols = [c for c in final_df.columns if c not in KEYS]

def country_is_valid(g: pd.DataFrame) -> bool:
    yrs = set(g["Year"].dropna().unique())
    if not (START_YEAR in yrs and END_YEAR in yrs):
        return False

    for col in indicator_cols:
        if g[col].isna().sum() > 1:
            return False

    return True

valid_ids = [cid for cid, g in final_df.groupby("Country_ID", dropna=False) if country_is_valid(g)]

filtered_df = final_df[final_df["Country_ID"].isin(valid_ids)].copy()
filtered_df = filtered_df.sort_values(["Country_ID", "Year"]).reset_index(drop=True)

filtered_df.to_csv(FILTERED_OUTPUT_PATH, index=False)

print("\nAfter country filtering:")
print(f"  Valid countries: {len(valid_ids)}")
print(f"  Rows: {filtered_df.shape[0]}")


# ============================================================
# 8) Imputation (numeric indicators only; Income group untouched)
# ============================================================
print("\nMissing values BEFORE imputation:")
print(filtered_df[indicator_cols].isna().sum())

pieces = []

for (cid, cname), g in filtered_df.groupby(["Country_ID", "Country_Name"], dropna=False):
    g = g.copy()

    g = g.set_index("Year").reindex(pd.Index(range(START_YEAR, END_YEAR + 1), name="Year"))

    g["Country_ID"] = cid
    g["Country_Name"] = cname

    for col in indicator_cols:
        if col != "Income group":
            g[col] = impute_series_linear_ffill_bfill(g[col])

    g = g.reset_index()
    pieces.append(g)

df_imputed = pd.concat(pieces, ignore_index=True)
df_imputed = df_imputed.sort_values(["Country_ID", "Year"]).reset_index(drop=True)

print("\nMissing values AFTER imputation:")
print(df_imputed[indicator_cols].isna().sum())


# ============================================================
# 9) Save IMPUTED dataset
# ============================================================
df_imputed.to_csv(IMPUTED_OUTPUT_PATH, index=False)

print("------------------------------------------------------")
print("✅ Year filtering, country filtering, and imputation completed")
print(f"Saved raw dataset      → {RAW_OUTPUT_PATH}")
print(f"Saved imputed dataset  → {IMPUTED_OUTPUT_PATH}")
print("======================================================")
