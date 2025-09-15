import pandas as pd
import os
from datetime import datetime, timedelta
import random
import uuid

import data_config as config
import generator_utils as utils

def main():
    """Main function to orchestrate the V7.0 dataset generation."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    borrower_ids = [f"borrower_{i:03}" for i in range(1, config.NUM_BORROWERS + 1)]
    
    print("PRE-COMPUTE: Generating Macro-Economic Timeline ")
    macro_timeline = pd.DataFrame([utils.get_macroeconomic_signals(d) for d in pd.date_range(config.START_DATE, config.END_DATE, freq='D')])
    macro_timeline.index = pd.to_datetime(macro_timeline.index)

    print("  Generate Deep Profiles & Primary Event Timelines ")
    profiles_df = utils.generate_borrower_profiles(borrower_ids)
    all_data = {bid: {'profile': p, 'events': [], 'transactions': pd.DataFrame()} for bid, p in profiles_df.set_index('borrower_id').to_dict('index').items()}
    for bid, data in all_data.items():
        scenario = config.STRESS_SCENARIOS.get(bid)
        data['events'] = utils.generate_full_event_timeline(bid, scenario, macro_timeline)
        print(f"  - Generated {len(data['events'])} primary events for {bid}")

    print("\n Simulate Event-Driven Network Contagion ---")
    initial_events_snapshot = {bid: list(data['events']) for bid, data in all_data.items()}
    for bid, data in all_data.items():
        linked_id = data['profile']['linked_borrower_id']
        if pd.notna(linked_id) and linked_id in all_data:
            for event in initial_events_snapshot[bid]:
                if 'contagion_probability' in event and random.random() < event['contagion_probability']:
                    contagion_event_date = event['event_date'] + timedelta(days=random.randint(15, 45))
                    new_event = {
                        "event_id": str(uuid.uuid4()),
                        "borrower_id": linked_id,
                        "event_date": contagion_event_date,
                        "event_type": "Linked Borrower Critical Stress",
                        "severity": "High",
                        "impact_duration": 120
                    }
                    all_data[linked_id]['events'].append(new_event)
                    print(f"  - Contagion! Event '{event['event_type']}' from {bid} triggered stress for {linked_id}")

    print("\n Generate Behavior-Driven Transactions ")
    for bid, data in all_data.items():
        data['transactions'] = utils.generate_transactions_for_borrower(bid, data['profile'], data['events'], config.TRANSACTIONS_PER_BORROWER)
        print(f"  - Generated transactions for {bid}")

    print("\n--- PASS 4: Final Assembly & Save ---")
    final_transactions = pd.concat([d['transactions'] for d in all_data.values()]).sort_values(by=['borrower_id', 'date']).reset_index(drop=True)
    
    final_events_list = [evt for d in all_data.values() for evt in d['events']]
    final_events = pd.DataFrame(final_events_list)
    
    comms_df, narratives_df = utils.generate_communications_and_narratives(final_events)
    
    if not final_events.empty:
        if 'event_date' in final_events.columns:
            final_events['event_date'] = pd.to_datetime(final_events['event_date']).dt.strftime('%Y-%m-%d')

    # Save files
    profiles_df.to_csv(os.path.join(config.OUTPUT_DIR, "borrowers.csv"), index=False)
    final_transactions.to_csv(os.path.join(config.OUTPUT_DIR, "transactions.csv"), index=False)
    final_events.to_csv(os.path.join(config.OUTPUT_DIR, "events.csv"), index=False)
    comms_df.to_csv(os.path.join(config.OUTPUT_DIR, "communications.csv"), index=False)
    narratives_df.to_csv(os.path.join(config.OUTPUT_DIR, "risk_narratives.csv"), index=False)

if __name__ == "__main__":
    main()