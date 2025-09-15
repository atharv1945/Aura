import pandas as pd
import os

class Config:
    W_BASE = 0.4
    RISK_THRESHOLD = 0.6

config = Config()

#  CONFIGURATION 
PROCESSED_DIR = "data/processed"
BORROWERS_ENHANCED_PATH = os.path.join(PROCESSED_DIR, "borrowers_enhanced.csv")
NETWORK_SCORES_PATH = os.path.join(PROCESSED_DIR, "network_risk_scores.csv")
TEMPORAL_SCORES_PATH = os.path.join(PROCESSED_DIR, "temporal_risk_scores.csv")
FINAL_SCORES_PATH = os.path.join(PROCESSED_DIR, "aura_risk_scores.csv")

def calculate_aura_score(s_temporal, s_network, r_f):
    """Applies the AURA risk score formula."""
    s_network_modifier = s_network * (1 - r_f)
    aura_score = s_temporal * ((1 - config.W_BASE) + config.W_BASE * s_network_modifier)
    return min(1.0, aura_score) 

def main():
    """Main function to load scores, calculate the final AURA score, and run analysis."""
    print("Loading all necessary data for final score calculation...")
    try:
        borrowers_df = pd.read_csv(BORROWERS_ENHANCED_PATH)
        network_scores_df = pd.read_csv(NETWORK_SCORES_PATH)
        temporal_scores_df = pd.read_csv(TEMPORAL_SCORES_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you have run both train_gnn.py and train_transformer.py successfully.")
        return

    print("Ensuring consistent data types for merge key 'borrower_id'...")
    borrowers_df['borrower_id'] = borrowers_df['borrower_id'].astype(str)
    network_scores_df['borrower_id'] = network_scores_df['borrower_id'].astype(str)
    temporal_scores_df['borrower_id'] = temporal_scores_df['borrower_id'].astype(str)

    print("Merging borrower data with model predictions...")
    final_df = pd.merge(borrowers_df, network_scores_df, on='borrower_id')
    final_df = pd.merge(final_df, temporal_scores_df, on='borrower_id')
    
    if final_df.empty:
        print("\nError: Merge resulted in an empty DataFrame. Please check the 'borrower_id' columns in your processed CSV files.")
        return

    print("Calculating the final AURA Risk Score for all borrowers...")
    final_df['aura_risk_score'] = final_df.apply(
        lambda row: calculate_aura_score(row['s_temporal'], row['s_network'], row['resilience_factor']),
        axis=1
    )
    
    final_df = final_df.sort_values(by='aura_risk_score', ascending=False)
    final_df.to_csv(FINAL_SCORES_PATH, index=False)
    print(f"\nFinal AURA risk scores saved to {FINAL_SCORES_PATH}")

    # PORTFOLIO RISK SUMMARY 
    print("\n--- Portfolio Risk Summary ---")
    flagged_borrowers = final_df[final_df['aura_risk_score'] > config.RISK_THRESHOLD]
    print(f"Total Borrowers: {len(final_df)}")
    print(f"Borrowers Flagged for Review (Score > {config.RISK_THRESHOLD}): {len(flagged_borrowers)}")
    print(f"Average AURA Risk Score: {final_df['aura_risk_score'].mean():.4f}")
    print(f"Maximum AURA Risk Score: {final_df['aura_risk_score'].max():.4f}")

    if not flagged_borrowers.empty:
        print("\n--- Top 10 Most At-Risk Borrowers (from model output) ---")
        print(final_df[['borrower_id', 'aura_risk_score', 's_temporal', 's_network', 'resilience_factor']].head(10).round(4))
        print("\nFlagged Borrower IDs for Stage 2:", flagged_borrowers['borrower_id'].tolist())
    else:
        print("\nNo borrowers exceeded the risk threshold for flagging.")

    #  DYNAMIC WHAT-IF ANALYSIS 
    print("\n--- What-If Analysis: Simulating a High-Risk Scenario ---")
    
    sim_borrower = final_df.sort_values(by='s_temporal', ascending=False).iloc[0]
    
    print(f"Analyzing borrower with highest temporal risk: {sim_borrower['borrower_id']}")
    print(f"Their Resilience Factor (Rf): {sim_borrower['resilience_factor']:.4f}")

    s_temporal_sim, s_network_sim = 0.90, 0.80
    print(f"\nSimulated S_temporal: {s_temporal_sim}")
    print(f"Simulated S_network: {s_network_sim}")

    print("\n--- Formula Breakdown ---")
    s_net_mod_sim = s_network_sim * (1 - sim_borrower['resilience_factor'])
    print(f"1. Network Modifier = {s_network_sim:.2f} * (1 - {sim_borrower['resilience_factor']:.2f}) = {s_net_mod_sim:.4f}")
    
    temporal_component = s_temporal_sim * (1 - config.W_BASE)
    print(f"2. Temporal Component = {s_temporal_sim:.2f} * {1 - config.W_BASE:.2f} = {temporal_component:.4f}")
    
    network_component = s_temporal_sim * config.W_BASE * s_net_mod_sim
    print(f"3. Network Interaction = {s_temporal_sim:.2f} * {config.W_BASE:.2f} * {s_net_mod_sim:.4f} = {network_component:.4f}")

    aura_risk_score_sim = calculate_aura_score(s_temporal_sim, s_network_sim, sim_borrower['resilience_factor'])
    
    print(f"\nCalculated Simulated AURA Risk Score: {aura_risk_score_sim:.4f}")
    print("-----------------------------------------")

    if aura_risk_score_sim > config.RISK_THRESHOLD:
        print(f"Conclusion: Borrower would be flagged for Stage 2.")
    else:
        print("Conclusion: Borrower's resilience keeps them below the threshold.")

if __name__ == "__main__":
    main()
