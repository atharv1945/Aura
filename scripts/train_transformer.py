import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os
import math

#  CONFIGURATION & DATA LOADING
PROCESSED_DIR = "data/processed"
TEMPORAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, "temporal_features.csv")
OUTPUT_DIR = "models"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "transformer_model.pth")
SCORES_SAVE_PATH = os.path.join(PROCESSED_DIR, "temporal_risk_scores.csv")

# Hyperparameters
SEQUENCE_LENGTH = 90
BATCH_SIZE = 32
EPOCHS = 15 
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading temporal data for Transformer...")
try:
    temporal_df = pd.read_csv(TEMPORAL_FEATURES_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure you have run the data_preprocessing.py script first.")
    exit()

# DATA PREPARATION & SEQUENCING 
print("Preparing data and creating sequences...")
# One-hot encode severity and select features
temporal_df = pd.get_dummies(temporal_df, columns=['severity'], drop_first=True)
features_to_scale = ['daily_spending', 'num_transactions', '30d_spending_sum', '7d_spending_avg']
other_features = [col for col in temporal_df.columns if 'severity_' in col]
features = features_to_scale + other_features
target = 'target'

# Scale numerical features
scaler = StandardScaler()
temporal_df[features_to_scale] = scaler.fit_transform(temporal_df[features_to_scale])
temporal_df[features] = temporal_df[features].astype(np.float32)

# Create sequences for each borrower
sequences, targets, sequence_info = [], [], []
for borrower_id, group in temporal_df.groupby('borrower_id'):
    if len(group) < SEQUENCE_LENGTH:
        continue
    
    for i in range(len(group) - SEQUENCE_LENGTH):
        seq = group[features].iloc[i:i+SEQUENCE_LENGTH].values
        label = group[target].iloc[i+SEQUENCE_LENGTH]
        sequences.append(seq)
        targets.append(label)

        sequence_info.append({'borrower_id': borrower_id, 'end_date': group['date'].iloc[i+SEQUENCE_LENGTH]})

X = np.array(sequences)
y = np.array(targets)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create PyTorch DataLoaders
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

print(f"Created {len(X)} sequences. Training on {len(X_train)}, testing on {len(X_test)}.")

# TRANSFORMER MODEL DEFINITION 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :]) 
        return output

model = TimeSeriesTransformer(n_features=X.shape[2], d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# TRAINING LOOP 
print("\nStarting Transformer training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for seq, labels in train_loader:
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation step
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for seq, labels in test_loader:
            output = model(seq)
            val_preds.extend(torch.sigmoid(output).squeeze().tolist())
            val_labels.extend(labels.tolist())
    val_auc = roc_auc_score(val_labels, val_preds)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}')

# SAVE MODEL AND PREDICTIONS 
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Transformer model saved to {MODEL_SAVE_PATH}")

# Generate final scores for each borrower by taking the latest sequence
model.eval()
with torch.no_grad():
    all_scores = []
    for borrower_id, group in temporal_df.groupby('borrower_id'):
        if len(group) >= SEQUENCE_LENGTH:
            last_sequence = group[features].iloc[-SEQUENCE_LENGTH:].values
            last_sequence_tensor = torch.from_numpy(last_sequence).float().unsqueeze(0)
            score = torch.sigmoid(model(last_sequence_tensor)).item()
            all_scores.append({'borrower_id': borrower_id, 's_temporal': score})
        else:
            all_scores.append({'borrower_id': borrower_id, 's_temporal': 0.0}) 

scores_df = pd.DataFrame(all_scores)
scores_df.to_csv(SCORES_SAVE_PATH, index=False)
print(f"✅ Temporal risk scores (S_temporal) saved to {SCORES_SAVE_PATH}")
