import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# =============================
# 1. Load data
# =============================
df = pd.read_csv("train_val_2001_2021_dropped_collinear.csv")  # 2001–2021 data
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

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# =============================
# 3. Time-based train/validation split
# =============================
train_df = df[df['year'] <= 2020]
val_df = df[df['year'] == 2021]

# Convert to tensors
X_numeric_train = torch.tensor(train_df[numeric_cols].values, dtype=torch.float32)
X_categorical_train = torch.tensor(train_df[categorical_cols].values, dtype=torch.long)
y_train = torch.tensor(train_df[TARGET_COL].values, dtype=torch.float32).unsqueeze(1)

X_numeric_val = torch.tensor(val_df[numeric_cols].values, dtype=torch.float32)
X_categorical_val = torch.tensor(val_df[categorical_cols].values, dtype=torch.long)
y_val = torch.tensor(val_df[TARGET_COL].values, dtype=torch.float32).unsqueeze(1)

# =============================
# 4. PyTorch Dataset
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

train_dataset = DamEnergyDataset(X_numeric_train, X_categorical_train, y_train)
val_dataset = DamEnergyDataset(X_numeric_val, X_categorical_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

print(f"Training rows: {len(train_dataset)}, Validation rows: {len(val_dataset)}")

# =============================
# 5. Neural network
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

categorical_cardinalities = [df[col].nunique() for col in categorical_cols]
model = DamNN(len(numeric_cols), categorical_cardinalities)

# =============================
# 6. Training setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 30
best_val_rmse = float('inf')
best_model_path = "best_dam_model.pt"

# =============================
# 7. Training loop
# =============================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for X_num_batch, X_cat_batch, y_batch in train_loader:
        X_num_batch, X_cat_batch, y_batch = X_num_batch.to(device), X_cat_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_num_batch, X_cat_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_num_batch.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_num_val, X_cat_val, y_val_batch in val_loader:
            X_num_val, X_cat_val = X_num_val.to(device), X_cat_val.to(device)
            outputs = model(X_num_val, X_cat_val)
            val_preds.append(outputs.cpu())
            val_targets.append(y_val_batch)

    val_preds = torch.cat(val_preds).numpy()
    val_targets = torch.cat(val_targets).numpy()
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
          f"Val RMSE: {val_rmse:.2f} | Val MAE: {val_mae:.2f} | R²: {val_r2:.3f}")

    # Save best model
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved best model at epoch {epoch+1}")

print(f"\nTraining complete. Best model saved to {best_model_path}")

# =============================
# 8. Save preprocessing objects
# =============================
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
