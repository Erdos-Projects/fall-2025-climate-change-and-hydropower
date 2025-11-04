import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

# =============================
# 1. Load data
# =============================
df = pd.read_csv("train_val_2001_2021.csv")  # 2001–2021 data
TARGET_COL = "RectifHyd_MWh"

# Ensure target is numeric and drop NaNs
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL])

# Ensure 'year' column exists and numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# =============================
# 2. Identify features
# =============================
categorical_cols = ["Division_ID", "Primary Purpose", "nerc_region", "mode"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + [TARGET_COL, 'year']]

# Convert numeric columns to float and fill NaNs with median
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Encode categorical columns (LightGBM can handle raw categories, but we keep the same preprocessing)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Scale numeric features (optional for tree models, but kept for consistency)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# =============================
# 3. Time-based train/validation split
# =============================
train_df = df[df['year'] <= 2020].copy()
val_df   = df[df['year'] == 2021].copy()

X_train = train_df[numeric_cols + categorical_cols]
y_train = train_df[TARGET_COL].values

X_val   = val_df[numeric_cols + categorical_cols]
y_val   = val_df[TARGET_COL].values

print(f"Training rows: {len(train_df)}, Validation rows: {len(val_df)}")

# =============================
# 4. LightGBM datasets
# =============================
lgb_train = lgb.Dataset(X_train, label=y_train,
                        categorical_feature=categorical_cols, free_raw_data=False)
lgb_val   = lgb.Dataset(X_val,   label=y_val,
                        reference=lgb_train, categorical_feature=categorical_cols, free_raw_data=False)

# =============================
# 5. LightGBM parameters
# =============================
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}

# =============================
# 6. Training with early stopping (FIXED)
# =============================
EVAL_EVERY = 50
best_model_path = "best_dam_lgbm.model"

# Import callbacks
early_stop = lgb.early_stopping(stopping_rounds=100, verbose=True)
log_eval = lgb.log_evaluation(period=EVAL_EVERY)

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=5000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "valid"],
    callbacks=[early_stop, log_eval],  # <-- Pass as callbacks list
)

# Save the best model
gbm.save_model(best_model_path)
print(f"\nTraining complete. Best model saved to {best_model_path}")
print(f"Best iteration: {gbm.best_iteration}")

# =============================
# 7. Validation metrics on the best iteration
# =============================
val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)

val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_mae  = mean_absolute_error(y_val, val_pred)
val_r2   = r2_score(y_val, val_pred)

print(f"Validation RMSE: {val_rmse:.2f} | MAE: {val_mae:.2f} | R²: {val_r2:.3f}")

# =============================
# 8. Save preprocessing objects (same as before)
# =============================
os.makedirs("preproc", exist_ok=True)
with open("preproc/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("preproc/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nPreprocessing objects saved in ./preproc/")
