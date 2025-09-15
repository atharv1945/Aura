import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import numpy as np

# CONFIGURATION & DATA LOADING
PROCESSED_DIR = "data/processed"
NODE_FEATURES_PATH = os.path.join(PROCESSED_DIR, "node_features.csv")
EDGE_LIST_PATH = os.path.join(PROCESSED_DIR, "edge_list.csv")
OUTPUT_DIR = "models"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "gnn_model.pth")
SCORES_SAVE_PATH = os.path.join(PROCESSED_DIR, "network_risk_scores.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading graph data for GNN...")
try:
    # Set borrower_id as the index column 
    node_features_df = pd.read_csv(NODE_FEATURES_PATH, index_col='borrower_id')
    edge_list_df = pd.read_csv(EDGE_LIST_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}. Please run preprocessing.py first.")
    exit()

# DATA PREPARATION 
feature_columns = [col for col in node_features_df.columns if col != 'target_contagion']
node_features_df[feature_columns] = node_features_df[feature_columns].astype(np.float32)

x = torch.tensor(node_features_df[feature_columns].values, dtype=torch.float)
y = torch.tensor(node_features_df['target_contagion'].values, dtype=torch.float).unsqueeze(1)
edge_index = torch.tensor(edge_list_df.values, dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index, y=y)

# Use a stratified split to handle class imbalance
indices = np.arange(data.num_nodes)
y_np = y.numpy()
positive_class_count = np.sum(y_np)

MIN_SAMPLES_FOR_STRATIFY = 2
if positive_class_count >= MIN_SAMPLES_FOR_STRATIFY:
    train_indices, test_indices, y_train, y_test = train_test_split(indices, y_np, test_size=0.2, random_state=42, stratify=y_np)
    if np.sum(y_train) >= MIN_SAMPLES_FOR_STRATIFY:
        train_indices, val_indices, _, _ = train_test_split(train_indices, y_train, test_size=0.125, random_state=42, stratify=y_train)
        print("Graph data prepared with STRATIFIED split.")
    else:
        print("Warning: Not enough positive samples for a stratified validation split. Falling back to random split for validation set.")
        train_indices, val_indices = train_test_split(train_indices, test_size=0.125, random_state=42)
else:
    print(f"Warning: Only {int(positive_class_count)} positive samples found. Cannot stratify. Falling back to random split.")
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.125, random_state=42)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_indices] = True
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask[val_indices] = True
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[test_indices] = True

print(data)
print(f"Positive samples -> Train: {int(data.y[data.train_mask].sum())}, Val: {int(data.y[data.val_mask].sum())}, Test: {int(data.y[data.test_mask].sum())}")

# GNN MODEL DEFINITION 
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GNN(num_node_features=data.num_node_features, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

pos_samples = data.y[data.train_mask].sum()
if pos_samples > 0:
    neg_samples = data.train_mask.sum() - pos_samples
    pos_weight = neg_samples / pos_samples
    print(f"Calculated positive class weight for loss function: {pos_weight:.2f}")
else:
    pos_weight = torch.tensor(1.0)
    print("No positive samples in training set. Using default weight of 1.0.")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# TRAINING LOOP 
print("\nStarting GNN training...")
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        probs = torch.sigmoid(out).squeeze()
        y_val_np = data.y[data.val_mask].cpu().numpy()
        val_auc = roc_auc_score(y_val_np, probs[data.val_mask].cpu().numpy()) if len(np.unique(y_val_np)) > 1 else 0.5
        y_test_np = data.y[data.test_mask].cpu().numpy()
        test_auc = roc_auc_score(y_test_np, probs[data.test_mask].cpu().numpy()) if len(np.unique(y_test_np)) > 1 else 0.5
        return val_auc, test_auc

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        val_auc, test_auc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

# SAVE MODEL AND PREDICTIONS 
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ GNN model saved to {MODEL_SAVE_PATH}")

model.eval()
with torch.no_grad():
    final_scores = torch.sigmoid(model(data)).squeeze().cpu().numpy()

# Use the DataFrame's index (which is borrower_id) to correctly map scores
scores_df = pd.DataFrame({
    'borrower_id': node_features_df.index,
    's_network': final_scores
})
scores_df.to_csv(SCORES_SAVE_PATH, index=False)
print(f"✅ Network risk scores (S_network) saved to {SCORES_SAVE_PATH}")

