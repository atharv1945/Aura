import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
LOOKAHEAD_DAYS = 30 # How many days in the future to predict a critical event

# --- HELPER FUNCTIONS ---
def load_data(data_dir):
    """Loads all necessary CSV files into pandas DataFrames."""
    print("Loading data...")
    try:
        borrowers_df = pd.read_csv(os.path.join(data_dir, "borrowers.csv"))
        transactions_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
        events_df = pd.read_csv(os.path.join(data_dir, "events.csv"))
        return borrowers_df, transactions_df, events_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure all CSV files are in the '{data_dir}' directory.")
        return None, None, None

def engineer_temporal_features(transactions_df, events_df, borrowers_df):
    """Engineers time-series features for the Transformer model."""
    print("Engineering temporal features...")
    
    # Convert date columns to datetime objects
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    events_df['event_date'] = pd.to_datetime(events_df['event_date'])

    # 1. Process Transactions: Create daily summaries for each borrower
    daily_transactions = transactions_df.set_index('date').groupby('borrower_id').resample('D').agg(
        daily_spending=('amount', 'sum'),
        num_transactions=('amount', 'count')
    ).reset_index()

    # Calculate rolling features
    daily_transactions = daily_transactions.sort_values(by=['borrower_id', 'date'])
    daily_transactions['30d_spending_sum'] = daily_transactions.groupby('borrower_id')['daily_spending'].transform(lambda s: s.rolling(window=30, min_periods=1).sum())
    daily_transactions['7d_spending_avg'] = daily_transactions.groupby('borrower_id')['daily_spending'].transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    
    # 2. Process Events: Create event flags for each day
    events_df['event_flag'] = 1
    events_df['is_critical_event'] = events_df['severity'].isin(['High', 'Critical']).astype(int)
    
    daily_events = events_df.rename(columns={'event_date': 'date'})
    
    # 3. Combine DataFrames
    temporal_df = pd.merge(daily_transactions, daily_events[['borrower_id', 'date', 'severity', 'is_critical_event']], on=['borrower_id', 'date'], how='left')
    temporal_df['severity'] = temporal_df['severity'].fillna('None')
    temporal_df['is_critical_event'] = temporal_df['is_critical_event'].fillna(0)
    
    # 4. Create the Target Variable
    # For each day, check if a critical event happens in the next LOOKAHEAD_DAYS
    temporal_df = temporal_df.sort_values(by=['borrower_id', 'date'])
    temporal_df['future_critical_event'] = temporal_df.groupby('borrower_id')['is_critical_event'].transform(
        lambda s: s.rolling(window=LOOKAHEAD_DAYS, min_periods=1).sum().shift(-LOOKAHEAD_DAYS)
    ).fillna(0).astype(int)
    
    temporal_df['target'] = (temporal_df['future_critical_event'] > 0).astype(int)
    
    return temporal_df.drop(columns=['future_critical_event', 'is_critical_event'])

def engineer_graph_features(borrowers_df, events_df):
    """Engineers node features and an edge list for the GNN model."""
    print("Engineering graph features...")
    
    # 1. Create Node Features from borrower profiles
    node_features = borrowers_df.copy()
    
    # One-hot encode categorical features
    categorical_cols = ['occupation', 'education', 'marital_status', 'income_tier', 'behavioral_archetype']
    node_features = pd.get_dummies(node_features, columns=categorical_cols, drop_first=True)
    
    # Normalize numerical features
    numerical_cols = ['age', 'monthly_income', 'savings_balance', 'credit_score', 'dependents', 'dti_ratio']
    scaler = MinMaxScaler()
    node_features[numerical_cols] = scaler.fit_transform(node_features[numerical_cols])

    # Create target variable: Did the borrower experience contagion?
    contagion_events = events_df[events_df['event_type'] == 'Linked Borrower Critical Stress']['borrower_id'].unique()
    node_features['target_contagion'] = node_features['borrower_id'].isin(contagion_events).astype(int)
    
    node_features = node_features.set_index('borrower_id')
    node_features = node_features.drop(columns=['name', 'city', 'linked_borrower_id']) # Drop non-feature columns
    
    # 2. Create Edge List
    edge_list = borrowers_df[['borrower_id', 'linked_borrower_id']].dropna()
    # Create mapping from borrower_id string to integer index
    borrower_map = {borrower_id: i for i, borrower_id in enumerate(borrowers_df['borrower_id'])}
    edge_list['source'] = edge_list['borrower_id'].map(borrower_map)
    edge_list['target'] = edge_list['linked_borrower_id'].map(borrower_map)
    
    return node_features, edge_list[['source', 'target']]

def calculate_resilience_factor(borrowers_df):
    """Calculates the Borrower Resilience Factor (Rf) and adds it to the borrowers table."""
    print("Calculating Resilience Factor...")
    df = borrowers_df.copy()
    
    # Normalize credit score and savings-to-income ratio
    df['credit_score_norm'] = df['credit_score'] / 850
    df['savings_ratio_norm'] = df['savings_balance'] / (df['monthly_income'] * 12 + 1) # Annualized
    
    # Calculate Rf as a weighted average
    resilience_score = 0.6 * df['credit_score_norm'] + 0.4 * df['savings_ratio_norm']
    
    # Scale between 0 and 1
    scaler = MinMaxScaler()
    df['resilience_factor'] = scaler.fit_transform(resilience_score.values.reshape(-1, 1))
    
    return df.drop(columns=['credit_score_norm', 'savings_ratio_norm'])


def main():
    """Main function to run the full preprocessing pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    borrowers, transactions, events = load_data(DATA_DIR)
    
    if borrowers is not None:
        # --- Task 1: Engineer Temporal Features ---
        temporal_features_df = engineer_temporal_features(transactions, events, borrowers)
        temporal_output_path = os.path.join(OUTPUT_DIR, "temporal_features.csv")
        temporal_features_df.to_csv(temporal_output_path, index=False)
        print(f"✅ Temporal features saved to {temporal_output_path}")

        # --- Task 2: Engineer Graph Features ---
        node_features_df, edge_list_df = engineer_graph_features(borrowers, events)
        node_output_path = os.path.join(OUTPUT_DIR, "node_features.csv")
        edge_output_path = os.path.join(OUTPUT_DIR, "edge_list.csv")
        node_features_df.to_csv(node_output_path)
        edge_list_df.to_csv(edge_output_path, index=False)
        print(f"✅ Graph features saved to {node_output_path} and {edge_output_path}")

        # --- Task 3: Calculate Resilience Factor ---
        borrowers_enhanced_df = calculate_resilience_factor(borrowers)
        borrower_output_path = os.path.join(OUTPUT_DIR, "borrowers_enhanced.csv")
        borrowers_enhanced_df.to_csv(borrower_output_path, index=False)
        print(f"✅ Enhanced borrower profiles saved to {borrower_output_path}")

if __name__ == "__main__":
    main()
    
"""
### **What This Script Does and Your New Datasets**

1.  `temporal_features.csv`:
    * **Purpose:** The direct input for the **Time-Series Transformer**.
    * **Structure:** Each row represents a single day in a borrower's life, with columns for their rolling spending habits, any events that occurred, and a `target` column indicating if a critical event is imminent.

2.  `node_features.csv`:
    * **Purpose:** Provides the features for each node (borrower) in our graph for the **GNN**.
    * **Structure:** Each row is a borrower, and the columns are their normalized and encoded attributes (credit score, income, behavioral archetype, etc.), plus a `target_contagion` column.

3.  `edge_list.csv`:
    * **Purpose:** Defines the connections between nodes for the **GNN**.
    * **Structure:** A simple two-column file listing the `source` and `target` for each link between borrowers.

4.  `borrowers_enhanced.csv`:
    * **Purpose:** Your original `borrowers.csv` file, now with the calculated **`resilience_factor` ($R_f$)** column.
    * **Use:** This will be used at the very end when we combine the model outputs to calculate the final AURA Risk Score.


"""