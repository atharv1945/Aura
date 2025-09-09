import pandas as pd
import os
from datetime import datetime
import random
import uuid

import config
import generator_utils as utils

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    borrower_ids = [f"borrower_{i:03}" for i in range(1, config.NUM_BORROWERS + 1)]
    
    print("Generating Macro-Economic Timeline")
    date_range = pd.date_range(config.START_DATE, config.END_DATE, freq='D')
    macro_data = [utils.get_macroeconomic_signals(d) for d in date_range]
    macro_timeline = pd.DataFrame(macro_data, index=date_range)

    print(" Generate Deep Profiles & Macro-Triggered Event Timelines ")
    profiles_df = utils.generate_borrower_profiles(borrower_ids)
    all_data = {bid: {'profile': p, 'events': [], 'transactions': pd.DataFrame()} for bid, p in profiles_df.set_index('borrower_id').to_dict('index').items()}
    for bid, data in all_data.items():
        scenario = config.STRESS_SCENARIOS.get(bid)
        data['events'] = utils.generate_full_event_timeline(bid, scenario, macro_timeline)

    print("\nSimulate Network Contagion (Recursive)")
    for bid, data in all_data.items():
        if any(evt['severity'] == 'Critical' for evt in data['events']):
            linked_id = data['profile']['linked_borrower_id']
            if pd.notna(linked_id) and linked_id in all_data and random.random() < 0.3: 
                contagion_date = utils.create_random_date(datetime(2025, 6, 1), config.END_DATE)
                all_data[linked_id]['events'].append({"event_id": str(uuid.uuid4()), "borrower_id": linked_id, "event_date": contagion_date, "event_type": "Linked Borrower Critical Stress", "severity": "Medium"})
                print(f"  - Network contagion applied to {linked_id} from {bid}")

    print("\nGenerate All Transactions")
    for bid, data in all_data.items():
        data['transactions'] = utils.generate_transactions_for_borrower(bid, data['profile'], data['events'], config.TRANSACTIONS_PER_BORROWER)

    print("\nFinal Assembly, Feature Engineering & Save ")
    final_transactions = pd.concat([d['transactions'] for d in all_data.values()]).sort_values(by=['borrower_id', 'date']).reset_index(drop=True)
    final_events_list = [evt for d in all_data.values() for evt in d['events']]
    for evt in final_events_list: 
        if isinstance(evt['event_date'], datetime):
            evt['event_date'] = evt['event_date'].strftime('%Y-%m-%d')
    final_events = pd.DataFrame(final_events_list)
    
    comms_df, narratives_df = utils.generate_communications_and_narratives(final_events)
    
    # Save files
    profiles_df.to_csv(os.path.join(config.OUTPUT_DIR, "borrowers.csv"), index=False)
    final_transactions.to_csv(os.path.join(config.OUTPUT_DIR, "transactions.csv"), index=False)
    final_events.to_csv(os.path.join(config.OUTPUT_DIR, "events.csv"), index=False)
    comms_df.to_csv(os.path.join(config.OUTPUT_DIR, "communications.csv"), index=False)
    narratives_df.to_csv(os.path.join(config.OUTPUT_DIR, "risk_narratives.csv"), index=False)
    
    print(f"   Files saved: borrowers.csv, transactions.csv, events.csv, communications.csv, risk_narratives.csv")

if __name__ == "__main__":
    main()