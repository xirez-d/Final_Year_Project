# ============================================================
# Health Expenditure Forecasting — Base Models + Stacking + Meta
# Train: 2000–2019 | Test: 2020–2022 (per country)
# Base Models:
#   - RFR, XGB, SVR  (tabular, per-year)
#   - AdaBoost       (one-hot Country_ID + Income group)
#   - LSTM           (multivariate panel with embeddings)
#   - TCN            (multivariate panel with embeddings, raw data)
#
# Stacking:
#   - Two XGB meta models:
#       Meta 1 on [WeightedAvg_ens1, Year, Country_ID]
#       Meta 2 on [WeightedAvg_ens2, Year, Country_ID]
#
# Output:
#   - stack_results_final_2022/stack_train_2000_2019_base_preds.csv
#   - stack_results_final_2022/stack_test_2020_2022_base_preds.csv
# ============================================================

import os, random, warnings, logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, RepeatVector, Concatenate
)
from tensorflow.keras.models import Model

from tcn import TCN

# ============================================================
# Config
# ============================================================
DATA_PATH_IMPUTED = "Final_Dataset_Imputed_2022.csv"
RESULTS_DIR = "stack_results_final_2022"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_END   = 2019
TEST_START  = 2020
SEQ_LEN     = 8

TARGET_COL  = "Health_Expenditure"

BASE_REQUIRED_COLS = [
    "Year",
    "Country_ID",
    "Country_Name",
    "Health_Expenditure",
    "Life_Expectancy",
    "Unemployment_Rate",
    "Death_Rate",
    "Gov_Effectiveness",
    "Population",
    "GDP_per_capita",
    "Income group",
]

BASE_NUMERIC_FEATURES = [
    "Year",
    "Life_Expectancy",
    "Unemployment_Rate",
    "Death_Rate",
    "Gov_Effectiveness",
    "Population",
    "GDP_per_capita",
]

ADA_NUMERIC_COLS = [
    "Year",
    "Life_Expectancy",
    "Unemployment_Rate",
    "Death_Rate",
    "Gov_Effectiveness",
    "Population",
    "GDP_per_capita",
]

ADA_CATEGORICAL_COLS = [
    "Country_ID",
    "Income group",
]

LSTM_ID_COL     = "Country_ID"
LSTM_INCOME_COL = "Income group"

LSTM_FEATURE_COLS = [
    "Year",
    "Life_Expectancy",
    "Unemployment_Rate",
    "Death_Rate",
    "Gov_Effectiveness",
    "Population",
    "GDP_per_capita",
]

EPS = 1e-6

TCN_ID_COL     = "Country_ID"
TCN_INCOME_COL = "Income group"
TCN_FEATURE_COLS = [
    "Year",
    "Life_Expectancy",
    "Unemployment_Rate",
    "Death_Rate",
    "Gov_Effectiveness",
    "Population",
    "GDP_per_capita",
]
COUNTRY_EMB_DIM_TCN = 8
INCOME_EMB_DIM_TCN  = 4

# ============================================================
# Helpers
# ============================================================
def safe_mape(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def report_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    print(f"{name:<28} | MAE: {mae:9.3f} | RMSE: {rmse:9.3f} | MAPE: {mape:7.2f}%")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def clip_nonnegative(a):
    return np.maximum(a, 0.0)

# ============================================================
# 1) Load full imputed dataset once (for RFR/XGB/SVR/ADA/LSTM)
# ============================================================
df_all = pd.read_csv(DATA_PATH_IMPUTED)

missing = [c for c in BASE_REQUIRED_COLS if c not in df_all.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_all = df_all[BASE_REQUIRED_COLS].copy()

# ============================================================
# Dict of test RMSEs for weighting later
# ============================================================
rmse_test = {}   # keys: "RFR", "XGB", "SVR", "ADA", "LSTM", "TCN"

# ============================================================
# 2) BASE PIPELINE — RFR / XGB / SVR
# ============================================================
df_base = df_all.copy()

df_base["Country_ID_num"]   = df_base["Country_ID"].astype("category").cat.codes
df_base["IncomeGroup_Code"] = df_base["Income group"].astype("category").cat.codes

numeric_features = BASE_NUMERIC_FEATURES + ["Country_ID_num", "IncomeGroup_Code"]

df_base = df_base.dropna(subset=[TARGET_COL] + numeric_features)

train_mask_base = df_base["Year"] <= TRAIN_END
test_mask_base  = df_base["Year"] >= TEST_START

train_df_base = df_base.loc[train_mask_base].copy()
test_df_base  = df_base.loc[test_mask_base].copy()

X_train_base = train_df_base[numeric_features].copy()
y_train_base = train_df_base[TARGET_COL].astype(float).values

X_test_base  = test_df_base[numeric_features].copy()
y_test_base  = test_df_base[TARGET_COL].astype(float).values

sort_idx_tr = np.lexsort([X_train_base["Year"].values,
                          X_train_base["Country_ID_num"].values])
sort_idx_te = np.lexsort([X_test_base["Year"].values,
                          X_test_base["Country_ID_num"].values])

X_train_base = X_train_base.iloc[sort_idx_tr].reset_index(drop=True)
y_train_base = y_train_base[sort_idx_tr]
train_df_base = train_df_base.iloc[sort_idx_tr].reset_index(drop=True)

X_test_base  = X_test_base.iloc[sort_idx_te].reset_index(drop=True)
y_test_base  = y_test_base[sort_idx_te]
test_df_base = test_df_base.iloc[sort_idx_te].reset_index(drop=True)

med_base = X_train_base.median(numeric_only=True)
X_train_base = X_train_base.fillna(med_base)
X_test_base  = X_test_base.fillna(med_base)

svr_scaler = RobustScaler().fit(X_train_base.values)
X_tr_svr   = svr_scaler.transform(X_train_base.values)
X_te_svr   = svr_scaler.transform(X_test_base.values)

base_models_tree = {
    "RFR": RandomForestRegressor(
        n_estimators=600,
        max_features=None,
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    "XGB": XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    "SVR": SVR(
        kernel="rbf",
        C=1000,
        gamma="auto"
    ),
}

tree_train_preds = {}
tree_test_preds  = {}
tree_results     = []

for name, model in base_models_tree.items():
    if name == "SVR":
        model.fit(X_tr_svr, y_train_base)
        train_preds = model.predict(X_tr_svr)
        test_preds  = model.predict(X_te_svr)
    else:
        model.fit(X_train_base.values, y_train_base)
        train_preds = model.predict(X_train_base.values)
        test_preds  = model.predict(X_test_base.values)

    train_preds = clip_nonnegative(train_preds)
    test_preds  = clip_nonnegative(test_preds)

    tree_train_preds[name] = train_preds
    tree_test_preds[name]  = test_preds

    metrics = report_metrics(
        y_test_base,
        test_preds,
        name
    )
    tree_results.append(metrics)
    rmse_test[name] = metrics["RMSE"]

# ============================================================
# 3) ADABOOST PIPELINE (ONE-HOT)
# ============================================================
df_ada = df_all.copy()

needed_ada = [TARGET_COL] + ADA_NUMERIC_COLS + ADA_CATEGORICAL_COLS
missing_ada = [c for c in needed_ada if c not in df_ada.columns]
if missing_ada:
    raise ValueError(f"Missing columns for AdaBoost: {missing_ada}")

df_ada = df_ada[needed_ada].copy()
df_ada = df_ada.dropna(subset=[TARGET_COL] + ADA_NUMERIC_COLS)

for col in ADA_CATEGORICAL_COLS:
    df_ada[col] = df_ada[col].fillna("Unknown").astype(str)

df_ada_enc = pd.get_dummies(df_ada, columns=ADA_CATEGORICAL_COLS, drop_first=False)

train_mask_ada = df_ada_enc["Year"] <= TRAIN_END
test_mask_ada  = df_ada_enc["Year"] >= TEST_START

train_df_ada_enc = df_ada_enc.loc[train_mask_ada].copy()
test_df_ada_enc  = df_ada_enc.loc[test_mask_ada].copy()

train_df_ada_orig = df_ada.loc[train_mask_ada, ["Year", "Country_ID", "Income group", TARGET_COL]].copy()
test_df_ada_orig  = df_ada.loc[test_mask_ada,  ["Year", "Country_ID", "Income group", TARGET_COL]].copy()

X_train_ada = train_df_ada_enc.drop(columns=[TARGET_COL])
y_train_ada = train_df_ada_enc[TARGET_COL].values

X_test_ada  = test_df_ada_enc.drop(columns=[TARGET_COL])
y_test_ada  = test_df_ada_enc[TARGET_COL].values

sort_idx_tr_ada = np.lexsort([train_df_ada_orig["Year"].values,
                              train_df_ada_orig["Country_ID"].values])
sort_idx_te_ada = np.lexsort([test_df_ada_orig["Year"].values,
                              test_df_ada_orig["Country_ID"].values])

X_train_ada = X_train_ada.iloc[sort_idx_tr_ada].reset_index(drop=True)
y_train_ada = y_train_ada[sort_idx_tr_ada]
train_df_ada_orig = train_df_ada_orig.iloc[sort_idx_tr_ada].reset_index(drop=True)

X_test_ada  = X_test_ada.iloc[sort_idx_te_ada].reset_index(drop=True)
y_test_ada  = y_test_ada[sort_idx_te_ada]
test_df_ada_orig = test_df_ada_orig.iloc[sort_idx_te_ada].reset_index(drop=True)

base_estimator = DecisionTreeRegressor(
    max_depth=6,
    random_state=RANDOM_SEED
)

adaboost_regressor = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=250,
    learning_rate=0.05,
    loss="linear",
    random_state=RANDOM_SEED
)

adaboost_regressor.fit(X_train_ada, y_train_ada)

y_pred_ada_test  = adaboost_regressor.predict(X_test_ada)
y_pred_ada_train = adaboost_regressor.predict(X_train_ada)

ada_metrics = report_metrics(y_test_ada, y_pred_ada_test, "AdaBoost")
rmse_test["ADA"] = ada_metrics["RMSE"]

ada_train_out = train_df_ada_orig.copy()
ada_train_out["AdaBoost_pred"] = clip_nonnegative(y_pred_ada_train)

ada_test_out = test_df_ada_orig.copy()
ada_test_out["AdaBoost_pred"] = clip_nonnegative(y_pred_ada_test)

# ============================================================
# 4) LSTM PIPELINE (MULTIVARIATE + EMBEDDINGS)
# ============================================================
df_lstm = df_all.copy()
required_lstm_cols = [LSTM_ID_COL, TARGET_COL, LSTM_INCOME_COL] + LSTM_FEATURE_COLS
missing_lstm = [c for c in required_lstm_cols if c not in df_lstm.columns]
if missing_lstm:
    raise ValueError(f"Missing columns for LSTM: {missing_lstm}")

df_lstm = df_lstm[required_lstm_cols].copy()
df_lstm = df_lstm.dropna(subset=[TARGET_COL] + LSTM_FEATURE_COLS)
df_lstm[LSTM_INCOME_COL] = df_lstm[LSTM_INCOME_COL].fillna("Unknown").astype(str)

unique_countries = sorted(df_lstm[LSTM_ID_COL].unique())
country_to_idx = {cid: i for i, cid in enumerate(unique_countries)}
idx_to_country = {i: cid for cid, i in country_to_idx.items()}
n_countries = len(unique_countries)

unique_income = sorted(df_lstm[LSTM_INCOME_COL].unique())
income_to_idx = {inc: i for i, inc in enumerate(unique_income)}
idx_to_income = {i: inc for inc, i in income_to_idx.items()}
n_income_groups = len(unique_income)

def build_sequences_for_country_lstm(g: pd.DataFrame, n_steps: int):
    g = g.loc[:, ~g.columns.duplicated()].copy()
    g = g.sort_values("Year")

    values = g[LSTM_FEATURE_COLS + [TARGET_COL]].values
    years  = g["Year"].values

    X_list, y_list, year_list = [], [], []

    for i in range(len(values) - n_steps + 1):
        end_ix = i + n_steps
        window = values[i:end_ix, :]
        seq_x = window[:, :-1]
        seq_y = window[-1, -1]
        y_year = years[end_ix - 1]

        X_list.append(seq_x)
        y_list.append(seq_y)
        year_list.append(y_year)

    if not X_list:
        return (
            np.empty((0, n_steps, len(LSTM_FEATURE_COLS))),
            np.array([]),
            np.array([]),
        )

    X_num = np.stack(X_list, axis=0)
    y_arr = np.array(y_list)
    years_seq = np.array(year_list)
    return X_num, y_arr, years_seq

X_num_all   = []
y_all       = []
years_all   = []
country_idx_all = []
income_idx_all  = []

for cid, g in df_lstm.groupby(LSTM_ID_COL):
    X_c, y_c, years_c = build_sequences_for_country_lstm(g, SEQ_LEN)
    if X_c.shape[0] == 0:
        continue

    c_idx = country_to_idx[cid]
    inc_str = g[LSTM_INCOME_COL].mode().iloc[0]
    inc_idx = income_to_idx[inc_str]

    country_idx_seq = np.full(X_c.shape[0], c_idx, dtype="int32")
    income_idx_seq  = np.full(X_c.shape[0], inc_idx, dtype="int32")

    X_num_all.append(X_c)
    y_all.append(y_c)
    years_all.append(years_c)
    country_idx_all.append(country_idx_seq)
    income_idx_all.append(income_idx_seq)

if not X_num_all:
    raise RuntimeError("No LSTM sequences created. Check SEQ_LEN and data coverage.")

X_num       = np.concatenate(X_num_all, axis=0)
y_lstm      = np.concatenate(y_all, axis=0)
years_seq   = np.concatenate(years_all, axis=0)
country_idx = np.concatenate(country_idx_all, axis=0)
income_idx  = np.concatenate(income_idx_all, axis=0)

n_features_lstm = X_num.shape[2]

train_mask_lstm = years_seq <= TRAIN_END
test_mask_lstm  = years_seq >= TEST_START

X_num_train   = X_num[train_mask_lstm]
y_train_lstm  = y_lstm[train_mask_lstm]
country_train = country_idx[train_mask_lstm]
income_train  = income_idx[train_mask_lstm]
years_train_seq = years_seq[train_mask_lstm]

X_num_test    = X_num[test_mask_lstm]
y_test_lstm   = y_lstm[test_mask_lstm]
country_test  = country_idx[test_mask_lstm]
income_test   = income_idx[test_mask_lstm]
years_test_seq = years_seq[test_mask_lstm]

if X_num_test.shape[0] == 0:
    raise RuntimeError("No LSTM test sequences created. Adjust SEQ_LEN/TRAIN_END/TEST_START.")

scaler_lstm = MinMaxScaler(feature_range=(0, 1))
X_train_flat = X_num_train.reshape(-1, n_features_lstm)
scaler_lstm.fit(X_train_flat)

X_num_train_scaled = scaler_lstm.transform(X_train_flat).reshape(X_num_train.shape)
X_num_test_scaled  = scaler_lstm.transform(
    X_num_test.reshape(-1, n_features_lstm)
).reshape(X_num_test.shape)

country_emb_dim = 8
income_emb_dim  = 4

num_input     = Input(shape=(SEQ_LEN, n_features_lstm), name="num_input")
country_input = Input(shape=(), dtype="int32", name="country_input")
income_input  = Input(shape=(), dtype="int32", name="income_input")

country_emb = Embedding(
    input_dim=n_countries,
    output_dim=country_emb_dim,
    name="country_embedding"
)(country_input)

income_emb = Embedding(
    input_dim=n_income_groups,
    output_dim=income_emb_dim,
    name="income_embedding"
)(income_input)

country_rep = RepeatVector(SEQ_LEN)(country_emb)
income_rep  = RepeatVector(SEQ_LEN)(income_emb)

x_lstm = Concatenate(axis=-1)([num_input, country_rep, income_rep])
lstm_out = LSTM(64, activation="relu")(x_lstm)
output = Dense(1, name="output")(lstm_out)

lstm_model = Model(inputs=[num_input, country_input, income_input], outputs=output)
lstm_model.compile(optimizer="adam", loss="mse")

lstm_model.fit(
    x=[X_num_train_scaled, country_train, income_train],
    y=y_train_lstm,
    epochs=60,
    batch_size=32,
    verbose=0,
    validation_data=([X_num_test_scaled, country_test, income_test], y_test_lstm),
)

y_pred_lstm_train = lstm_model.predict(
    [X_num_train_scaled, country_train, income_train],
    verbose=0
).reshape(-1)

y_pred_lstm_test = lstm_model.predict(
    [X_num_test_scaled, country_test, income_test],
    verbose=0
).reshape(-1)

country_ids_train = np.array([idx_to_country[i] for i in country_train])
country_ids_test  = np.array([idx_to_country[i] for i in country_test])

lstm_train_out = pd.DataFrame({
    "Country_ID": country_ids_train,
    "Year": years_train_seq,
    "LSTM_pred": clip_nonnegative(y_pred_lstm_train),
})
lstm_test_out = pd.DataFrame({
    "Country_ID": country_ids_test,
    "Year": years_test_seq,
    "LSTM_pred": clip_nonnegative(y_pred_lstm_test),
})

lstm_train_out = lstm_train_out.groupby(["Country_ID", "Year"], as_index=False)["LSTM_pred"].mean()
lstm_test_out  = lstm_test_out.groupby(["Country_ID", "Year"], as_index=False)["LSTM_pred"].mean()

# ============================================================
# 4b) TCN PIPELINE (MULTIVARIATE + EMBEDDINGS, RAW DATA)
# ============================================================
df_tcn = pd.read_csv(DATA_PATH_IMPUTED)
df_tcn = df_tcn.loc[:, ~df_tcn.columns.duplicated()].copy()

df_tcn[TCN_ID_COL] = df_tcn[TCN_ID_COL].astype(str)
df_tcn[TCN_INCOME_COL] = df_tcn[TCN_INCOME_COL].fillna("Unknown").astype(str)

df_tcn = df_tcn.sort_values([TCN_ID_COL, "Year"])

needed_cols_tcn = [TCN_ID_COL, "Year", TCN_INCOME_COL, TARGET_COL] + TCN_FEATURE_COLS
missing_tcn = [c for c in needed_cols_tcn if c not in df_tcn.columns]
if missing_tcn:
    raise ValueError(f"Missing columns in TCN dataset: {missing_tcn}")

df_tcn = df_tcn[needed_cols_tcn].copy()

unique_countries_tcn = sorted(df_tcn[TCN_ID_COL].unique())
country_to_idx_tcn = {cid: i for i, cid in enumerate(unique_countries_tcn)}
n_countries_tcn = len(unique_countries_tcn)

unique_income_tcn = sorted(df_tcn[TCN_INCOME_COL].unique())
income_to_idx_tcn = {inc: i for i, inc in enumerate(unique_income_tcn)}
n_income_groups_tcn = len(unique_income_tcn)

def build_sequences_for_country_tcn(g: pd.DataFrame, n_steps: int):
    g = g.loc[:, ~g.columns.duplicated()].copy()
    g = g.sort_values("Year")
    g = g.dropna(subset=TCN_FEATURE_COLS + [TARGET_COL])

    if g.shape[0] < n_steps:
        return (
            np.empty((0, n_steps, len(TCN_FEATURE_COLS))),
            np.array([]),
            np.array([]),
        )

    values = g[TCN_FEATURE_COLS + [TARGET_COL]].values
    years  = g["Year"].values

    X_list, y_list, years_list = [], [], []

    for i in range(len(values) - n_steps + 1):
        end_ix = i + n_steps
        window = values[i:end_ix]

        seq_x = window[:, :-1]
        seq_y = window[-1, -1]
        seq_year = years[end_ix - 1]

        X_list.append(seq_x)
        y_list.append(seq_y)
        years_list.append(seq_year)

    return (
        np.stack(X_list),
        np.array(y_list),
        np.array(years_list)
    )

X_all_tcn = []
y_all_tcn = []
years_all_tcn = []
country_idx_all_tcn = []
income_idx_all_tcn = []
country_label_all_tcn = []

for cid, g in df_tcn.groupby(TCN_ID_COL):
    X_c, y_c, years_c = build_sequences_for_country_tcn(g, SEQ_LEN)
    if X_c.shape[0] == 0:
        continue

    c_idx = country_to_idx_tcn[cid]
    inc_str = g[TCN_INCOME_COL].mode().iloc[0]
    inc_idx = income_to_idx_tcn[inc_str]
    n_seq = X_c.shape[0]

    X_all_tcn.append(X_c)
    y_all_tcn.append(y_c)
    years_all_tcn.append(years_c)
    country_idx_all_tcn.append(np.full(n_seq, c_idx, dtype="int32"))
    income_idx_all_tcn.append(np.full(n_seq, inc_idx, dtype="int32"))
    country_label_all_tcn.append(np.full(n_seq, cid, dtype=object))

X_all_tcn = np.concatenate(X_all_tcn, axis=0)
y_all_tcn = np.concatenate(y_all_tcn, axis=0)
years_all_tcn = np.concatenate(years_all_tcn, axis=0)
country_idx_all_tcn = np.concatenate(country_idx_all_tcn, axis=0)
income_idx_all_tcn = np.concatenate(income_idx_all_tcn, axis=0)
country_label_all_tcn = np.concatenate(country_label_all_tcn, axis=0)

train_mask_tcn = years_all_tcn <= TRAIN_END
test_mask_tcn  = years_all_tcn >= TEST_START

X_train_tcn = X_all_tcn[train_mask_tcn]
y_train_tcn = y_all_tcn[train_mask_tcn]
X_test_tcn  = X_all_tcn[test_mask_tcn]
y_test_tcn  = y_all_tcn[test_mask_tcn]

years_train_tcn = years_all_tcn[train_mask_tcn]
years_test_tcn  = years_all_tcn[test_mask_tcn]

country_train_idx_tcn = country_idx_all_tcn[train_mask_tcn]
country_test_idx_tcn  = country_idx_all_tcn[test_mask_tcn]

income_train_idx_tcn = income_idx_all_tcn[train_mask_tcn]
income_test_idx_tcn  = income_idx_all_tcn[test_mask_tcn]

country_train_labels_tcn = country_label_all_tcn[train_mask_tcn]
country_test_labels_tcn  = country_label_all_tcn[test_mask_tcn]

if X_test_tcn.shape[0] == 0:
    raise RuntimeError("No TCN test sequences found. Check TRAIN_END and TEST_START for TCN.")

n_features_tcn = X_train_tcn.shape[2]

scaler_tcn = MinMaxScaler()
scaler_tcn.fit(X_train_tcn.reshape(-1, n_features_tcn))

X_train_scaled_tcn = scaler_tcn.transform(X_train_tcn.reshape(-1, n_features_tcn)).reshape(X_train_tcn.shape)
X_test_scaled_tcn  = scaler_tcn.transform(X_test_tcn.reshape(-1, n_features_tcn)).reshape(X_test_tcn.shape)

num_input_tcn = Input(shape=(SEQ_LEN, n_features_tcn), name="num_input_tcn")
country_input_tcn = Input(shape=(), dtype="int32", name="country_input_tcn")
income_input_tcn  = Input(shape=(), dtype="int32", name="income_input_tcn")

country_emb_tcn = Embedding(
    input_dim=n_countries_tcn,
    output_dim=COUNTRY_EMB_DIM_TCN,
    name="country_embedding_tcn"
)(country_input_tcn)

income_emb_tcn = Embedding(
    input_dim=n_income_groups_tcn,
    output_dim=INCOME_EMB_DIM_TCN,
    name="income_embedding_tcn"
)(income_input_tcn)

country_rep_tcn = RepeatVector(SEQ_LEN)(country_emb_tcn)
income_rep_tcn  = RepeatVector(SEQ_LEN)(income_emb_tcn)

x_concat_tcn = Concatenate(axis=-1)([num_input_tcn, country_rep_tcn, income_rep_tcn])

tcn_out = TCN(
    nb_filters=64,
    kernel_size=3,
    dilations=[1, 2, 4, 8],
    activation="relu",
    dropout_rate=0.0,
    return_sequences=False
)(x_concat_tcn)

output_tcn = Dense(1, name="tcn_output")(tcn_out)

tcn_model = Model(
    inputs=[num_input_tcn, country_input_tcn, income_input_tcn],
    outputs=output_tcn
)

tcn_model.compile(optimizer="adam", loss="mse")

tcn_model.fit(
    [X_train_scaled_tcn, country_train_idx_tcn, income_train_idx_tcn],
    y_train_tcn,
    epochs=60,
    batch_size=32,
    validation_data=([X_test_scaled_tcn, country_test_idx_tcn, income_test_idx_tcn], y_test_tcn),
    verbose=0
)

y_pred_test_tcn = tcn_model.predict(
    [X_test_scaled_tcn, country_test_idx_tcn, income_test_idx_tcn],
    verbose=0
).ravel()

y_pred_train_tcn = tcn_model.predict(
    [X_train_scaled_tcn, country_train_idx_tcn, income_train_idx_tcn],
    verbose=0
).ravel()

tcn_train_df = pd.DataFrame({
    "Country_ID": country_train_labels_tcn,
    "Year": years_train_tcn,
    "y_true": y_train_tcn,
    "y_pred_tcn_emb": clip_nonnegative(y_pred_train_tcn),
})

tcn_test_df = pd.DataFrame({
    "Country_ID": country_test_labels_tcn,
    "Year": years_test_tcn,
    "y_true": y_test_tcn,
    "y_pred_tcn_emb": clip_nonnegative(y_pred_test_tcn),
})

tcn_all_df = pd.concat([tcn_train_df, tcn_test_df], ignore_index=True)

tcn_df_agg = (
    tcn_all_df.groupby(["Country_ID", "Year"], as_index=False)["y_pred_tcn_emb"]
             .mean()
             .rename(columns={"y_pred_tcn_emb": "TCN_pred"})
)

# ============================================================
# 5) BUILD STACKED TRAIN / TEST DATASETS (all base models)
# ============================================================
stack_train = pd.DataFrame({
    "Country_ID": train_df_base["Country_ID"].values,
    "Year": train_df_base["Year"].values,
    "Health_Expenditure": y_train_base,
})
stack_test = pd.DataFrame({
    "Country_ID": test_df_base["Country_ID"].values,
    "Year": test_df_base["Year"].values,
    "Health_Expenditure": y_test_base,
})

# Add RFR, XGB, SVR predictions
for name in ["RFR", "XGB", "SVR"]:
    stack_train[f"{name}_pred"] = tree_train_preds[name]
    stack_test[f"{name}_pred"]  = tree_test_preds[name]

# Add AdaBoost
stack_train = stack_train.merge(
    ada_train_out[["Country_ID", "Year", "AdaBoost_pred"]],
    on=["Country_ID", "Year"],
    how="left"
)
stack_test = stack_test.merge(
    ada_test_out[["Country_ID", "Year", "AdaBoost_pred"]],
    on=["Country_ID", "Year"],
    how="left"
)
stack_train.rename(columns={"AdaBoost_pred": "ADA_pred"}, inplace=True)
stack_test.rename(columns={"AdaBoost_pred": "ADA_pred"}, inplace=True)

# Add LSTM
stack_train = stack_train.merge(
    lstm_train_out,
    on=["Country_ID", "Year"],
    how="left"
)
stack_test = stack_test.merge(
    lstm_test_out,
    on=["Country_ID", "Year"],
    how="left"
)

# Add TCN
stack_train = stack_train.merge(
    tcn_df_agg,
    on=["Country_ID", "Year"],
    how="left"
)
stack_test = stack_test.merge(
    tcn_df_agg,
    on=["Country_ID", "Year"],
    how="left"
)

# ============================================================
# 5b) Compute aligned LSTM & TCN metrics on 2020–2022 test
# ============================================================
lstm_aligned = stack_test.dropna(subset=["LSTM_pred"])
if not lstm_aligned.empty:
    lstm_metrics_aligned = report_metrics(
        lstm_aligned["Health_Expenditure"].values,
        lstm_aligned["LSTM_pred"].values,
        "LSTM+Emb"
    )
    rmse_test["LSTM"] = lstm_metrics_aligned["RMSE"]

tcn_aligned = stack_test.dropna(subset=["TCN_pred"])
if not tcn_aligned.empty:
    tcn_metrics_aligned = report_metrics(
        tcn_aligned["Health_Expenditure"].values,
        tcn_aligned["TCN_pred"].values,
        "TCN+Emb"
    )
    rmse_test["TCN"] = tcn_metrics_aligned["RMSE"]

# ============================================================
# 6) TWO RMSE-BASED WEIGHTED AVERAGES (TRAIN + TEST)
#    Ensemble 1: RFR, XGB, SVR, ADA, LSTM
#    Ensemble 2: RFR, XGB, ADA, LSTM, TCN
# ============================================================
ens1_models = ["RFR", "XGB", "SVR", "ADA", "LSTM"]
ens2_models = ["RFR", "XGB", "ADA", "LSTM", "TCN"]

for m in ens1_models + ens2_models:
    if m not in rmse_test:
        raise RuntimeError(f"Missing RMSE for model {m} in rmse_test dict: {rmse_test}")

def compute_weights(model_list):
    inv_rmse = {m: 1.0 / rmse_test[m] for m in model_list}
    inv_sum = sum(inv_rmse.values())
    return {m: inv_rmse[m] / inv_sum for m in model_list}

weights_ens1 = compute_weights(ens1_models)
weights_ens2 = compute_weights(ens2_models)

def compute_weighted_avg_generic(df: pd.DataFrame, model_list, weights_dict) -> np.ndarray:
    cols = [f"{m}_pred" for m in model_list]
    preds = df[cols].values.astype(float)
    W_vec = np.array([weights_dict[m] for m in model_list])
    mask = ~np.isnan(preds)
    w = W_vec * mask
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = np.nan
    w_norm = w / w_sum
    wa = np.nansum(w_norm * preds, axis=1)
    return wa

stack_train["WeightedAvg_ens1"] = compute_weighted_avg_generic(stack_train, ens1_models, weights_ens1)
stack_test["WeightedAvg_ens1"]  = compute_weighted_avg_generic(stack_test, ens1_models, weights_ens1)

stack_train["WeightedAvg_ens2"] = compute_weighted_avg_generic(stack_train, ens2_models, weights_ens2)
stack_test["WeightedAvg_ens2"]  = compute_weighted_avg_generic(stack_test, ens2_models, weights_ens2)

# ============================================================
# 7) XGB META MODELS ON [WeightedAvg_ens, Year, Country_ID]
# ============================================================
for df_stack in (stack_train, stack_test):
    df_stack["Country_ID_cat"] = df_stack["Country_ID"].astype("category").cat.codes

# Meta 1 (Ensemble 1)
meta_features_ens1 = ["WeightedAvg_ens1", "Year", "Country_ID_cat"]

X_meta_train_ens1 = stack_train[meta_features_ens1].values
y_meta_train = stack_train["Health_Expenditure"].values

X_meta_test_ens1  = stack_test[meta_features_ens1].values
y_meta_test = stack_test["Health_Expenditure"].values

xgb_meta_ens1 = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    random_state=RANDOM_SEED,
    tree_method="hist",
)

xgb_meta_ens1.fit(X_meta_train_ens1, y_meta_train)

stack_train["XGB_Meta_ens1"] = xgb_meta_ens1.predict(X_meta_train_ens1)
stack_test["XGB_Meta_ens1"]  = xgb_meta_ens1.predict(X_meta_test_ens1)

# Meta 2 (Ensemble 2)
meta_features_ens2 = ["WeightedAvg_ens2", "Year", "Country_ID_cat"]

X_meta_train_ens2 = stack_train[meta_features_ens2].values
X_meta_test_ens2  = stack_test[meta_features_ens2].values

xgb_meta_ens2 = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    random_state=RANDOM_SEED,
    tree_method="hist",
)

xgb_meta_ens2.fit(X_meta_train_ens2, y_meta_train)

stack_train["XGB_Meta_ens2"] = xgb_meta_ens2.predict(X_meta_train_ens2)
stack_test["XGB_Meta_ens2"]  = xgb_meta_ens2.predict(X_meta_test_ens2)

# ============================================================
# 8) PRINT ENSEMBLE & META RESULTS (GROUPED)
# ============================================================
print("\nRMSE-based model weights (Ensemble 1: RFR, XGB, SVR, ADA, LSTM):")
for m in sorted(ens1_models):
    print(f"  {m}: {weights_ens1[m]:.4f}")

meta1_metrics = report_metrics(
    y_meta_test,
    stack_test["XGB_Meta_ens1"].values,
    "XGB Meta Ens1"
)

print("\nRMSE-based model weights (Ensemble 2: RFR, XGB, ADA, LSTM, TCN):")
for m in sorted(ens2_models):
    print(f"  {m}: {weights_ens2[m]:.4f}")

meta2_metrics = report_metrics(
    y_meta_test,
    stack_test["XGB_Meta_ens2"].values,
    "XGB Meta Ens2"
)

# ============================================================
# 9) SAVE
# ============================================================
train_stack_path = os.path.join(RESULTS_DIR, "stack_train_2000_2019_base_preds.csv")
test_stack_path  = os.path.join(RESULTS_DIR, "stack_test_2020_2022_base_preds.csv")

stack_train.to_csv(train_stack_path, index=False)
stack_test.to_csv(test_stack_path, index=False)

print("\nSaved stacked base + two ensembles + two meta predictions:")
print("  Train:", train_stack_path)
print("  Test :", test_stack_path)