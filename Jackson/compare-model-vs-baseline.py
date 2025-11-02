import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from scipy.stats import ttest_rel

# =============================
# 1. Load test data
# =============================
df = pd.read_csv("test_2022_dropped_collinear.csv")
TARGET_COL = "RectifHyd_MWh"

# Ensure target is numeric and drop NaNs
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL])

# =============================
# 2. Identify features
# =============================
categorical_cols = ["Division_ID", "Primary Purpose", "nerc_region", "mode"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + [TARGET_COL, 'year']]

# Convert numeric columns to float and fill NaNs
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# =============================
# 3. Load preprocessing objects
# =============================
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Encode categorical columns
for col in categorical_cols:
    le = label_encoders[col]
    df[col] = le.transform(df[col].astype(str))

# Scale numeric columns
df[numeric_cols] = scaler.transform(df[numeric_cols])

# =============================
# 4. Convert to tensors
# =============================
X_numeric = torch.tensor(df[numeric_cols].values, dtype=torch.float32)
X_categorical = torch.tensor(df[categorical_cols].values, dtype=torch.long)
y_test = torch.tensor(df[TARGET_COL].values, dtype=torch.float32).unsqueeze(1)

# =============================
# 5. Dataset and DataLoader
# =============================
class DamEnergyDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

test_dataset = DamEnergyDataset(X_numeric, X_categorical, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# =============================
# 6. Define model
# =============================
class DamNN(nn.Module):
    def __init__(self, n_numeric, categorical_cardinalities, emb_dim=16, hidden_dim=128):
        super(DamNN, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, emb_dim) for cardinality in categorical_cardinalities]
        )
        n_emb = emb_dim * len(categorical_cardinalities)
        self.fc1 = nn.Linear(n_numeric + n_emb, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

categorical_cardinalities = [len(label_encoders[col].classes_) for col in categorical_cols]
model = DamNN(len(numeric_cols), categorical_cardinalities)

# Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best_dam_model.pt", map_location=device))
model = model.to(device)
model.eval()

# =============================
# 7. Predict on test data
# =============================
y_preds = []
y_true = []

with torch.no_grad():
    for X_num_batch, X_cat_batch, y_batch in test_loader:
        X_num_batch, X_cat_batch = X_num_batch.to(device), X_cat_batch.to(device)
        outputs = model(X_num_batch, X_cat_batch)
        y_preds.append(outputs.cpu())
        y_true.append(y_batch)

y_preds = torch.cat(y_preds).numpy()
y_true = torch.cat(y_true).numpy()

# =============================
# 8. Baseline: mean predictor (training mean)
# =============================
train_df = pd.read_csv("train_val_2001_2021_dropped_collinear.csv")
train_df[TARGET_COL] = pd.to_numeric(train_df[TARGET_COL], errors='coerce')
train_df = train_df.dropna(subset=[TARGET_COL])
y_mean = train_df[TARGET_COL].mean()
baseline_preds = np.full_like(y_true, y_mean)

# =============================
# 9. Flatten arrays to 1D
# =============================
y_true_flat = y_true.flatten()
y_preds_flat = y_preds.flatten()
baseline_preds_flat = baseline_preds.flatten()

# =============================
# 10. Compute metrics
# =============================
model_rmse = np.sqrt(mean_squared_error(y_true_flat, y_preds_flat))
model_mae = mean_absolute_error(y_true_flat, y_preds_flat)
model_r2 = r2_score(y_true_flat, y_preds_flat)

baseline_rmse = np.sqrt(mean_squared_error(y_true_flat, baseline_preds_flat))
baseline_mae = mean_absolute_error(y_true_flat, baseline_preds_flat)

print(f"Model RMSE: {model_rmse:.2f}, MAE: {model_mae:.2f}, R²: {model_r2:.3f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}, MAE: {baseline_mae:.2f}")

# =============================
# 11. Hypothesis test: paired t-test
# =============================
from scipy.stats import ttest_rel

model_errors = np.abs(y_true_flat - y_preds_flat)
baseline_errors = np.abs(y_true_flat - baseline_preds_flat)

t_stat, p_value = ttest_rel(baseline_errors, model_errors)
print(f"Paired t-test: t = {t_stat:.3f}, p-value = {p_value:.3e}")

if p_value < 0.05:
    print("✅ Model predictions are significantly better than baseline (p < 0.05)")
else:
    print("❌ Model predictions are NOT significantly better than baseline (p >= 0.05)")
