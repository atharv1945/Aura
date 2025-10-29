import pandas as pd
import os
from datetime import datetime, timedelta
import random
import uuid
import sys # For path manipulation if needed

# Import configurations and utility functions
# Ensure data_config and generator_utils are importable
try:
    # Assuming scripts run from project root or IDE handles paths
    # If run_pipeline.py is in scripts/, this should work directly
    import data_config as config
    import generator_utils as utils
except ImportError:
    # Fallback if run directly from scripts folder (less ideal)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Assumes script is in scripts/
    # Add project root to path if needed, though direct import should work if run_pipeline is correct
    # sys.path.insert(0, PROJECT_ROOT)
    try:
        import data_config as config
        import generator_utils as utils
    except ImportError as e:
        print(f"Fatal Error: Could not import config or utils: {e}")
        print("Ensure data_config.py and generator_utils.py are in the same directory or accessible via PYTHONPATH.")
        sys.exit(1)
    # finally: # Avoid removing paths unless absolutely necessary and managed carefully
    #      # Clean up sys.path modification
    #     if SCRIPT_DIR in sys.path:
    #         try:
    #             sys.path.remove(SCRIPT_DIR)
    #         except ValueError:
    #             pass

def main():
    """Main function to orchestrate the V7.0 dataset generation."""
    # Ensure output directory exists, handle potential errors
    try:
        # Use absolute path for output directory based on config location
        output_dir_path = os.path.abspath(config.OUTPUT_DIR)
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir_path}")
    except OSError as e:
        print(f"Fatal Error: Could not create output directory '{config.OUTPUT_DIR}': {e}")
        sys.exit(1)
    except AttributeError:
        print("Fatal Error: OUTPUT_DIR not found in config. Check data_config.py.")
        sys.exit(1)


    borrower_ids = [f"borrower_{i:03}" for i in range(1, config.NUM_BORROWERS + 1)]

    print("PRE-COMPUTE: Generating Macro-Economic Timeline ")
    try:
        if not isinstance(config.START_DATE, datetime) or not isinstance(config.END_DATE, datetime):
             raise TypeError("START_DATE and END_DATE must be datetime objects in data_config.py")
        date_range = pd.date_range(config.START_DATE, config.END_DATE, freq='D')
        macro_data = [utils.get_macroeconomic_signals(d) for d in date_range]
        macro_timeline = pd.DataFrame(macro_data, index=date_range)
        if macro_timeline.empty:
             print("Warning: Macro timeline generation resulted in an empty DataFrame.")
        elif macro_timeline.isnull().values.any():
             print("Warning: Macro timeline contains missing (NaN) values.")
    except Exception as e:
        print(f"Fatal Error: Failed to generate macro timeline: {e}")
        sys.exit(1)


    print("  Generate Deep Profiles & Primary Event Timelines ")
    try:
        profiles_df = utils.generate_borrower_profiles(borrower_ids)
        if profiles_df.empty:
             print("Fatal Error: Borrower profile generation resulted in an empty DataFrame.")
             sys.exit(1)
        all_data = {bid: {'profile': p, 'events': [], 'transactions': pd.DataFrame()}
                    for bid, p in profiles_df.set_index('borrower_id').to_dict('index').items()}
    except Exception as e:
        print(f"Fatal Error: Failed during profile generation or dictionary conversion: {e}")
        sys.exit(1)

    # Generate primary events
    for bid, data in all_data.items():
        scenario = config.STRESS_SCENARIOS.get(bid)
        try:
            generated_events = utils.generate_full_event_timeline(bid, scenario, macro_timeline)
            data['events'] = generated_events if isinstance(generated_events, list) else []
            if scenario or data['events']:
                 print(f"  - Generated {len(data['events'])} primary events for {bid}")
        except Exception as e:
            print(f"Warning: Failed to generate events for {bid} (Scenario: {scenario}): {e}")
            data['events'] = []


    print("\n Simulate Event-Driven Network Contagion ---")
    initial_events_snapshot = {bid: list(data['events']) for bid, data in all_data.items()}
    contagion_count = 0
    for bid, data in all_data.items():
        profile_data = data.get('profile', {})
        linked_id = profile_data.get('linked_borrower_id')

        if pd.notna(linked_id) and linked_id in all_data:
            for event in initial_events_snapshot[bid]:
                 contagion_prob = event.get('contagion_probability', 0)
                 event_type_str = event.get('event_type', 'Unknown Event Type')
                 event_date = event.get('event_date')

                 if isinstance(event_date, datetime) and isinstance(contagion_prob, (int, float)) and contagion_prob > 0 and random.random() < contagion_prob:
                    try:
                        contagion_event_date = event_date + timedelta(days=random.randint(15, 45))
                        if contagion_event_date <= config.END_DATE:
                            new_event = {
                                "event_id": str(uuid.uuid4()),
                                "borrower_id": linked_id,
                                "event_date": contagion_event_date,
                                "event_type": "Linked Borrower Critical Stress",
                                "severity": "High",
                                "impact_duration": 120
                            }
                            if linked_id in all_data and isinstance(all_data[linked_id].get('events'), list):
                                all_data[linked_id]['events'].append(new_event)
                                print(f"  - Contagion! Event '{event_type_str}' from {bid} triggered stress for {linked_id} on {contagion_event_date.strftime('%Y-%m-%d')}")
                                contagion_count += 1
                            else:
                                print(f"Warning: Target borrower {linked_id} not found or events list missing during contagion.")
                    except Exception as e:
                        print(f"Warning: Error creating contagion event from {bid} to {linked_id}: {e}")

    print(f"Total contagion events simulated: {contagion_count}")

    print("\n Generate Behavior-Driven Transactions ")
    total_transactions_generated = 0
    for bid, data in all_data.items():
        try:
            # --- FIX: Call with correct 3 arguments ---
            transactions_df = utils.generate_transactions_for_borrower(bid, data['profile'], data['events'])
            # --- END FIX ---
            data['transactions'] = transactions_df if isinstance(transactions_df, pd.DataFrame) else pd.DataFrame()
            num_generated = len(data['transactions'])
            total_transactions_generated += num_generated
            print(f"  - Generated {num_generated} transactions for {bid}")
        except Exception as e:
            print(f"Warning: Failed to generate transactions for {bid}: {e}")
            data['transactions'] = pd.DataFrame()


    print("\n--- PASS 4: Final Assembly & Save ---")
    try:
        # Concatenate transactions safely
        all_transactions_dfs = [d['transactions'] for d in all_data.values() if isinstance(d.get('transactions'), pd.DataFrame) and not d['transactions'].empty]
        if not all_transactions_dfs:
             print("Warning: No transactions were generated for any borrower.")
             final_transactions = pd.DataFrame()
        else:
             final_transactions = pd.concat(all_transactions_dfs).sort_values(by=['borrower_id', 'date']).reset_index(drop=True)

        # Concatenate events safely
        final_events_list = [evt for d in all_data.values() for evt in d.get('events', []) if isinstance(evt, dict)]
        if not final_events_list:
             print("Warning: No events were generated or recorded for any borrower.")
             final_events = pd.DataFrame()
             comms_df = pd.DataFrame()
             narratives_df = pd.DataFrame()
        else:
            final_events = pd.DataFrame(final_events_list)
            # Generate comms and narratives only if events exist
            # Pass the DataFrame with datetime objects
            comms_df, narratives_df = utils.generate_communications_and_narratives(final_events.copy()) # Pass copy

            # Convert event dates to strings *after* comms/narratives generation for saving
            if 'event_date' in final_events.columns:
                # Coerce errors during conversion for robustness
                final_events['event_date'] = pd.to_datetime(final_events['event_date'], errors='coerce').dt.strftime('%Y-%m-%d')
                # Handle potential NaNs created by coercion
                final_events['event_date'].fillna('Invalid Date', inplace=True) # Apply fillna correctly


        # Define file paths using config.OUTPUT_DIR
        output_dir = config.OUTPUT_DIR
        borrowers_path = os.path.join(output_dir, "borrowers.csv")
        transactions_path = os.path.join(output_dir, "transactions.csv")
        events_path = os.path.join(output_dir, "events.csv")
        comms_path = os.path.join(output_dir, "communications.csv")
        narratives_path = os.path.join(output_dir, "risk_narratives.csv")

        # Save files, checking if DataFrames are not empty
        if not profiles_df.empty:
             profiles_df.to_csv(borrowers_path, index=False)
             print(f"Saved borrowers data ({len(profiles_df)} rows) to {borrowers_path}")
        else:
             print("Warning: Borrowers DataFrame is empty. Not saving.")

        if not final_transactions.empty:
             final_transactions.to_csv(transactions_path, index=False)
             print(f"Saved transactions data ({len(final_transactions)} rows) to {transactions_path}")
        else:
             print("Warning: Transactions DataFrame is empty. Not saving.")

        if not final_events.empty:
             final_events.to_csv(events_path, index=False)
             print(f"Saved events data ({len(final_events)} rows) to {events_path}")
        else:
             print("Warning: Events DataFrame is empty. Not saving.")

        if isinstance(comms_df, pd.DataFrame) and not comms_df.empty:
             comms_df.to_csv(comms_path, index=False)
             print(f"Saved communications data ({len(comms_df)} rows) to {comms_path}")
        else:
             print("Warning: Communications DataFrame is empty or invalid. Not saving.")

        if isinstance(narratives_df, pd.DataFrame) and not narratives_df.empty:
             narratives_df.to_csv(narratives_path, index=False)
             print(f"Saved risk narratives data ({len(narratives_df)} rows) to {narratives_path}")
        else:
             print("Warning: Narratives DataFrame is empty or invalid. Not saving.")


    except Exception as e:
        print(f"Fatal Error: Failed during final assembly or saving: {e}")
        # Consider logging traceback here for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    print("\nData generation script finished.")