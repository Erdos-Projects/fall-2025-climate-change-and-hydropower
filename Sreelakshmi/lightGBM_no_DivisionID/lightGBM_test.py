import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

# =============================
# 1. Load test data (2022)
# =============================
test_df = pd.read_csv("final_test_2022_no_did.csv") 
TARGET_COL = "RectifHyd_MWh"

# Ensure numeric target
test_df[TARGET_COL] = pd.to_numeric(test_df[TARGET_COL], errors='coerce')
test_df = test_df.dropna(subset=[TARGET_COL])

# =============================
# 2. Define columns (same as training)
# =============================
categorical_cols = ["Primary Purpose", "nerc_region", "mode"]
numeric_cols = [c for c in test_df.columns if c not in categorical_cols + [TARGET_COL, 'year']]

# Ensure numeric features
for col in numeric_cols:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# =============================
# 3. Load preprocessing objects
# =============================
with open("preproc/label_encoders.pkl", "rb") as f:  # Updated path
    label_encoders = pickle.load(f)
with open("preproc/scaler.pkl", "rb") as f:          # Updated path
    scaler = pickle.load(f)

# Fill missing numeric values with median from test set
for col in numeric_cols:
    test_df[col] = test_df[col].fillna(test_df[col].median())

# Encode categorical features using training label encoders
for col in categorical_cols:
    le = label_encoders[col]
    # Handle unseen categories gracefully
    test_df[col] = test_df[col].astype(str).map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

# Scale numeric features using training scaler
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

# =============================
# 4. Prepare features and target
# =============================
X_test = test_df[numeric_cols + categorical_cols]
y_test = test_df[TARGET_COL].values

print(f"Test rows: {len(test_df)}")

# =============================
# 5. Load trained LightGBM model
# =============================
model_path = "best_dam_lgbm.model"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

gbm = lgb.Booster(model_file=model_path)
print(f"Loaded LightGBM model from {model_path}")
print(f"Best iteration: {gbm.best_iteration}")

# =============================
# 6. Predict & Evaluate
# =============================
# Predict using best iteration
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# Compute metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n2022 Test Results (LightGBM)")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.3f}")

# =============================
# 7. Optional: Feature importance
# =============================
import matplotlib.pyplot as plt

lgb.plot_importance(gbm, max_num_features=20, importance_type='gain', figsize=(10, 8))
plt.title("Top 20 Feature Importance (Gain) - LightGBM")
plt.tight_layout()
plt.savefig("lightgbm_feature_importance.png")
plt.show()
