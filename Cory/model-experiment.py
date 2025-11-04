import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

#--------------------------
# Code from Jackson
#--------------------------
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

df = pd.read_csv("train_val_2001_2021_dropped_collinear.csv")  # 2001–2021 data
TARGET_COL = "RectifHyd_MWh"

# Ensure target is numeric and drop NaNs
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL])

# Ensure 'year' column exists and numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

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

categorical_cardinalities = [df[col].nunique() for col in categorical_cols]
model = DamNN(len(numeric_cols), categorical_cardinalities)
#-----------------------------
# End code from Jackson
#-----------------------------

with open('model.pt', 'rb') as f:
    model.load_state_dict(torch.load(f))


df = pd.read_csv('final_test_2022.csv')

carmen_smith_id = (df['Latitude'] == 44.005).idxmax()

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL])

# Ensure 'year' column exists and numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = label_encoders[col].transform(df[col].astype(str))

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

def pred(frame):
    num = torch.tensor(frame[numeric_cols].values, dtype=torch.float32)
    cat = torch.tensor(frame[categorical_cols].values, dtype=torch.long)
    return model.forward(num, cat)

def pred_shifted(frame, temp_mean, temp_std, pcpn_mean, pcpn_std):
    num = torch.tensor(frame[numeric_cols].values, dtype=torch.float32)
    num_cols = len(frame)
    random_pcpn = torch.normal(
        mean = torch.ones(num_cols)*pcpn_mean,
        std = torch.ones(num_cols)*pcpn_std
    )
    num[:, -1] += random_pcpn
    random_temp = torch.normal(
        mean = torch.ones(num_cols)*temp_mean,
        std = torch.ones(num_cols)*temp_std
    )
    num[:, -4:-1] += random_temp[:, None]
    cat = torch.tensor(frame[categorical_cols].values, dtype=torch.long)
    return model.forward(num, cat)

def plot_results(frame, temp_mean, temp_std, pcpn_mean, pcpn_std):
    results = torch.zeros((2000, len(frame)))
    baseline = pred(frame).reshape(-1)
    for i in range(2000):
        results[i, :] = pred_shifted(frame, temp_mean, temp_std, pcpn_mean, pcpn_std).reshape(-1)
    quantiles = torch.tensor([0.01, 0.1, 0.25, 0.75, 0.9, 0.99])
    quants = torch.quantile(results, quantiles, 0)
    alpha = [.1, .4, .8, .4, .1]
    x = np.arange(0, len(frame))
    for i in range(len(quantiles)-1):
        plt.fill_between(x, quants[i, :].cpu().detach().numpy(), quants[i+1, :].cpu().detach().numpy(), alpha = alpha[i])
    plt.plot(x, baseline.cpu().detach().numpy())
    plt.show()

if __name__ == '__main__':
    plot_results(df[0:12], 1, .1, -1, .1)
