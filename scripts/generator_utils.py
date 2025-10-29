import pandas as pd
from faker import Faker
import random
import numpy as np
from datetime import datetime, timedelta
import uuid
import json
import os

import data_config as config

#Faker instance
fake = Faker()
if config.RANDOM_SEED:
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    Faker.seed(config.RANDOM_SEED)

def get_macroeconomic_signals(date):
    """Generates macro-economic indicators for a given date."""
    inflation_rate = round(4.5 + 2.5 * np.sin(date.month / 12 * 2 * np.pi), 2)
    unemployment_rate = round(5.0 + 1.5 * np.cos((date.year - 2024) * np.pi + date.month / 12 * 2 * np.pi), 2)
    interest_rate = round(6.5 + 1.0 * np.sin((date.year - 2024) * np.pi + date.month / 6 * np.pi), 2)
    market_index = round(18000 + 3000 * np.sin((date.year - 2024) * np.pi + date.month / 12 * 2 * np.pi) + random.uniform(-500, 500), 0)
    return {'inflation': inflation_rate, 'unemployment': unemployment_rate, 'interest_rate': interest_rate, 'market_index': market_index}

def generate_borrower_profiles(borrower_ids):
    """Generates deep, realistic borrower profiles."""
    # (This function remains unchanged - already updated for density)
    profiles = []
    for bid in borrower_ids:
        base_income = random.choice([30000, 50000, 80000, 120000])
        profiles.append({
            "borrower_id": bid, "name": fake.name(), "age": random.randint(22, 60), "city": fake.city(),
            "occupation": random.choice(["Software Engineer", "Teacher", "Doctor", "Freelancer", "Business Owner"]),
            "education": random.choice(["Graduate", "Post-Graduate", "PhD"]), "marital_status": random.choice(["Single", "Married"]),
            "income_tier": random.choice(["Low", "Middle", "High"]), "monthly_income": base_income,
            "income_volatility": round(random.uniform(0.05, 0.4), 2),
            "savings_balance": base_income * random.uniform(1, 6), "credit_score": random.randint(300, 850),
            "dependents": random.randint(0, 4), "linked_borrower_id": None,
            "behavioral_archetype": random.choice(config.BEHAVIORAL_ARCHETYPES), "dti_ratio": round(random.uniform(0.2, 0.6), 2)
        })

    # Ensure denser network
    linked_count = 0
    max_links = config.NUM_BORROWERS // 2 # Aim to link about half
    indices = list(range(len(profiles)))
    random.shuffle(indices) # Shuffle to avoid bias
    
    used_indices = set()
    for i in range(0, len(indices) - 1, 2):
        if linked_count >= max_links: break
        p1_idx = indices[i]
        p2_idx = indices[i+1]
        
        # Ensure indices are valid and not already used
        if p1_idx < len(profiles) and p2_idx < len(profiles) and \
           p1_idx not in used_indices and p2_idx not in used_indices:
             profiles[p1_idx]['linked_borrower_id'] = profiles[p2_idx]['borrower_id']
             profiles[p2_idx]['linked_borrower_id'] = profiles[p1_idx]['borrower_id']
             used_indices.add(p1_idx)
             used_indices.add(p2_idx)
             linked_count += 1
             
    df = pd.DataFrame(profiles)
    # Handle potential missing credit scores before calculating limit
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce').fillna(300) # Default low score if missing
    df['credit_limit'] = df['credit_score'] * np.random.uniform(50, 150, size=len(df))
    return df

def generate_full_event_timeline(borrower_id, scenario, macro_timeline):
    """Generates a sequence of events for a borrower based on scenarios and macro triggers."""
    events = []
    if not scenario: return events

    # --- UPGRADE: Precise start date for late crisis ---
    if scenario == "late_stage_crisis":
        # Start exactly on July 1st, 2025 (+/- a few days) to ensure events are recent for prediction
        current_date = datetime(2025, 7, 1) + timedelta(days=random.randint(0, 5))
        print(f"  -> INFO: Starting LATE STAGE crisis for {borrower_id} on {current_date.strftime('%Y-%m-%d')}")
    else:
        # Default start date for all other scenarios (early 2025)
        current_date = create_random_date(datetime(2025, 1, 1), datetime(2025, 4, 1))
    # --- END UPGRADE ---

    chain = config.EVENT_CHAINS.get(scenario)
    if not chain:
        # Handle cases where scenario name might be misspelled or directly used as event type
        print(f"Warning: Scenario '{scenario}' not found in EVENT_CHAINS. Treating as single event.")
        events.append({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date,
                       "event_type": scenario, "severity": "High", "impact_duration": 90, "contagion_probability": 0.1})
        return events

    for event_def in chain:
        # --- UPGRADE: Robust Macro Trigger Logic ---
        trigger = event_def.get("macro_trigger")
        triggered = True # Assume triggered unless proven otherwise
        actual_value = 'N/A' # Initialize for logging
        if trigger:
            triggered = False # Reset if trigger exists
            macro_key, (op, val) = list(trigger.items())[0]
            # Ensure current_date matches the macro_timeline index type (handle timezone awareness if necessary)
            current_date_naive = current_date.replace(tzinfo=None) if current_date.tzinfo else current_date
            macro_timeline_index_dates = macro_timeline.index.date

            if current_date_naive.date() in macro_timeline_index_dates:
                # Use .loc for safe access
                macro_value_series = macro_timeline.loc[macro_timeline.index.date == current_date_naive.date(), macro_key]
                if not macro_value_series.empty:
                    actual_value = macro_value_series.iloc[0]
                    # Check operator safely
                    try:
                        if op == '>' and actual_value > val: triggered = True
                        elif op == '<' and actual_value < val: triggered = True
                        elif op == '==' and actual_value == val: triggered = True
                        elif op == '>=' and actual_value >= val: triggered = True
                        elif op == '<=' and actual_value <= val: triggered = True
                        elif op == '!=' and actual_value != val: triggered = True
                    except TypeError:
                         print(f"  -> Warning: Type mismatch comparing macro value {actual_value} ({type(actual_value)}) with {val} ({type(val)}) for {macro_key}")
                         # Decide how to handle type mismatch - here we skip trigger
                         triggered = False # Safer to not trigger if types mismatch

            if not triggered:
                print(f"  -> Skipping event '{event_def.get('type', 'Unknown')}' for {borrower_id}: Macro condition not met on {current_date_naive.strftime('%Y-%m-%d')} ({macro_key} {op} {val} vs actual {actual_value})")
                continue # Skip this event if macro condition not met
        # --- END UPGRADE ---

        # Create Event Dictionary (Ensure 'event_type' consistency)
        event_def_copy = event_def.copy()
        # --- UPGRADE: Ensure event_type key exists ---
        if 'type' in event_def_copy and 'event_type' not in event_def_copy:
            event_def_copy['event_type'] = event_def_copy.pop('type')
        if 'event_type' not in event_def_copy: # Safety net if 'type' was also missing
             event_def_copy['event_type'] = "Unknown Event Type"
             print(f"Warning: Event definition missing 'type' or 'event_type' key for borrower {borrower_id}.")
        # --- END UPGRADE ---

        event_def_copy.update({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date})
        events.append(event_def_copy)

        # Branching Logic (Robust handling)
        if event_def.get("branch"):
            branch_options = list(event_def["branch"].keys())
            branch_probs = list(event_def["branch"].values())
            # Ensure probabilities sum to 1 (or close enough)
            if not np.isclose(sum(branch_probs), 1.0):
                 print(f"Warning: Branch probabilities for event '{event_def_copy['event_type']}' do not sum to 1. Normalizing.")
                 branch_probs = (np.array(branch_probs) / sum(branch_probs)).tolist() # Ensure list output

            try:
                branch_choice = np.random.choice(branch_options, p=branch_probs)
            except ValueError as e:
                print(f"Error choosing branch for event '{event_def_copy['event_type']}': {e}. Check probabilities.")
                break # Stop processing this chain

            if branch_choice == "Loan Restructuring":
                restructure_chain = config.EVENT_CHAINS.get("loan_restructuring_path", [])
                for restructure_event_def in restructure_chain:
                    restructure_delay = restructure_event_def.get("delay")
                    if not restructure_delay or not (isinstance(restructure_delay, (tuple, list)) and len(restructure_delay) == 2):
                         print(f"Warning: Invalid or missing delay in loan_restructuring_path. Skipping event.")
                         continue
                    current_date += timedelta(days=random.randint(*restructure_delay))
                    restructure_event_def_copy = restructure_event_def.copy()
                    # Ensure event_type consistency
                    if 'type' in restructure_event_def_copy and 'event_type' not in restructure_event_def_copy:
                         restructure_event_def_copy['event_type'] = restructure_event_def_copy.pop('type')
                    if 'event_type' not in restructure_event_def_copy:
                         restructure_event_def_copy['event_type'] = "Unknown Restructure Event"
                    restructure_event_def_copy.update({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date})
                    events.append(restructure_event_def_copy)
                break # Exit the main chain after branching
            elif branch_choice == "Recovery":
                 # Potentially trigger recovery later, but for now, just end this path
                 break # Exit the main chain after branching
            # Else (e.g., Payday Loan Taken), continue the main chain normally

        # Calculate next event date
        delay = event_def.get("delay")
        if delay is None: break # End chain if no more delays specified
        # Ensure delay is a valid tuple/list of two integers
        if isinstance(delay, (tuple, list)) and len(delay) == 2 and all(isinstance(d, int) for d in delay) and delay[0] <= delay[1]:
             try:
                 current_date += timedelta(days=random.randint(*delay))
             except ValueError as e:
                 print(f"Warning: Invalid delay range {delay} for event '{event_def_copy['event_type']}': {e}. Ending chain.")
                 break
        else:
             print(f"Warning: Invalid delay format {delay} for event '{event_def_copy['event_type']}'. Ending chain.")
             break

    return events


def generate_transactions_for_borrower(borrower_id, profile, events):
    """Generates a realistic set of transactions based on profile, events, and behavior."""
    # (Keep this function as is - already robust)
    transactions = []
    home_city = profile.get('city', 'Unknown City') # Safe access
    archetype = profile.get('behavioral_archetype', 'Standard') # Default archetype
    monthly_income = profile.get('monthly_income', 50000) # Default income
    dti = profile.get('dti_ratio', 0.4) # Default DTI

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        # Safely convert to datetime, coercing errors
        events_df['event_date'] = pd.to_datetime(events_df['event_date'], errors='coerce')
        events_df.dropna(subset=['event_date'], inplace=True) # Remove rows where date conversion failed
        # Calculate impact_end_date only if event_date is valid
        if not events_df.empty:
            events_df['impact_end_date'] = events_df.apply(
                lambda row: row['event_date'] + timedelta(days=row.get('impact_duration', 0)) if pd.notna(row['event_date']) else pd.NaT,
                axis=1
            )

    for dt in pd.date_range(start=config.START_DATE, end=config.END_DATE, freq='D'):
        active_events = pd.DataFrame()
        if not events_df.empty:
            # Ensure dt is compatible (timezone-naive)
            dt_naive = dt.replace(tzinfo=None)
            # Filter active events safely
            try:
                active_events = events_df[
                    (events_df['event_date'] <= dt_naive) &
                    (events_df['impact_end_date'] >= dt_naive) &
                    (pd.notna(events_df['event_date'])) # Ensure dates are valid
                ]
            except TypeError as e:
                # Handle potential timezone comparison errors if they arise
                print(f"Warning: Error comparing dates during active event check: {e}")
                # Fallback: convert events_df dates to naive if needed, but should be handled by earlier coerce
                events_df['event_date_naive'] = events_df['event_date'].dt.tz_localize(None)
                events_df['impact_end_date_naive'] = events_df['impact_end_date'].dt.tz_localize(None)
                active_events = events_df[
                    (events_df['event_date_naive'] <= dt_naive) &
                    (events_df['impact_end_date_naive'] >= dt_naive)
                ]


        stress_factor = 1.0
        if not active_events.empty:
            # Check severity safely, handling potential NaNs
            severities = active_events['severity'].astype(str).tolist()
            if any(s in severities for s in ['High', 'Critical']):
                stress_factor = 0.6

        # Generate standard transactions
        if dt.day == 1:
            transactions.append(create_transaction(borrower_id, dt, "credit", "Salary", monthly_income, home_city, profile))
        if dt.day == 5:
            # Ensure EMI is positive
            emi_amount = max(0, monthly_income * dti)
            transactions.append(create_transaction(borrower_id, dt, "debit", "Rent/EMI", emi_amount, home_city, profile))

        # Generate discretionary transactions
        prob_transact = 0.5
        if archetype == "Spender": prob_transact = 0.6
        elif archetype == "Frugal": prob_transact = 0.4

        if random.random() < prob_transact:
            category, amount = generate_archetype_spend(archetype, dt, stress_factor)
            if amount > 0: # Only record positive amounts
                 transactions.append(create_transaction(borrower_id, dt, "debit", category, amount, home_city, profile))

    df = pd.DataFrame(transactions)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


def generate_archetype_spend(archetype, date, stress_factor=1.0):
    """Models spending amount and category based on behavioral archetype and stress."""
    is_weekend = date.weekday() >= 5
    if archetype == "Frugal":
        category = random.choices(config.DISCRETIONARY_CATEGORIES, weights=[0.4, 0.1, 0.2, 0.2, 0.01, 0.04, 0.05], k=1)[0]
        amount = random.uniform(100, 2000)
    elif archetype == "Spender":
        category = random.choices(config.DISCRETIONARY_CATEGORIES, weights=[0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1], k=1)[0]
        amount = random.uniform(500, 8000)
    else: # Standard or Impulsive
        category = random.choice(config.DISCRETIONARY_CATEGORIES)
        amount = random.uniform(200, 5000)

    if archetype == "Impulsive": amount *= 1.2
    if is_weekend and category in ["Restaurant", "Entertainment", "Travel"]: amount *= 1.5 # Adjusted categories

    # Apply stress factor more granularly
    stress_reduction_map = {"Luxury": 0.8, "Entertainment": 0.6, "Restaurant": 0.5, "Travel": 0.7, "Online Shopping": 0.3}
    if category in stress_reduction_map:
         reduction = stress_reduction_map[category]
         effective_stress_factor = 1.0 - (1.0 - stress_factor) * reduction # Apply partial reduction based on stress
         amount *= effective_stress_factor

    return category, max(0, amount)

def create_transaction(borrower_id, date, trans_type, category, amount, location, profile, description=None):
    """Helper function to create a single transaction dictionary."""
    archetype = profile.get('behavioral_archetype', 'Standard')
    channel_weights_map = {"Frugal": [0.1, 0.4, 0.4, 0.1, 0.0], "Spender": [0.5, 0.1, 0.2, 0.1, 0.1], "Standard": [0.3, 0.3, 0.3, 0.1, 0.0], "Impulsive": [0.4, 0.1, 0.4, 0.1, 0.0]}
    weights = channel_weights_map.get(archetype, channel_weights_map['Standard']) # Default weights

    amount *= random.uniform(0.95, 1.05)
    return {
        "transaction_id": str(uuid.uuid4()), "borrower_id": borrower_id, "date": date.strftime('%Y-%m-%d %H:%M:%S'),
        "type": trans_type, "category": category, "amount": round(amount, 2), "location": location,
        "description": description or category, "payment_channel": random.choices(config.PAYMENT_CHANNELS, weights=weights, k=1)[0],
        **get_macroeconomic_signals(date)
    }

def create_random_date(start, end):
    """Generates a random datetime within a given range."""
    try:
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
    except ValueError as e:
        print(f"Warning: Error creating random date between {start} and {end}: {e}. Returning start date.")
        return start

def generate_communications_and_narratives(events_df):
    """Generates post-hoc communications and risk narratives from the final event log."""
    # (Keep this function as is - already robust)
    comms, narratives = [], []
    if events_df.empty:
        return pd.DataFrame(comms), pd.DataFrame(narratives)

    # Ensure 'event_date' is datetime before calculations
    try:
        # Use errors='coerce' to handle potential invalid date strings
        events_df['event_date'] = pd.to_datetime(events_df['event_date'], errors='coerce')
        # Drop rows where date conversion failed
        events_df.dropna(subset=['event_date'], inplace=True)
        if events_df.empty: return pd.DataFrame(comms), pd.DataFrame(narratives)
    except Exception as e:
         print(f"Error converting event_date to datetime in generate_comms: {e}. Skipping comms/narrative generation.")
         return pd.DataFrame(comms), pd.DataFrame(narratives)


    for borrower_id, group in events_df.groupby('borrower_id'):
        if group.empty: continue

        # Calculate severity score safely
        group['severity_score'] = group['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}).fillna(0).astype(int)
        # If multiple events have the max score, idxmax returns the first occurrence
        idx_max_severity = group['severity_score'].idxmax()
        # Handle case where idxmax might return NaN if all scores are 0 or group is empty after filtering
        if pd.isna(idx_max_severity): continue
        most_severe_event = group.loc[idx_max_severity]

        # Ensure event_type is a string
        event_type_str = str(most_severe_event.get('event_type', 'Unknown Event'))
        severity_str = str(most_severe_event.get('severity', 'Unknown Severity'))

        # Calculate dates safely
        comm_date = most_severe_event['event_date'] + timedelta(days=random.randint(3, 7))
        narrative_date = most_severe_event['event_date']

        comms.append({"comm_id": str(uuid.uuid4()), "borrower_id": borrower_id, "comm_date": comm_date.strftime('%Y-%m-%d'),
                      "comm_channel": "Email", "comm_text": f"Regarding your account status after event: {event_type_str}",
                      "sentiment": "Negative" if most_severe_event['severity'] in ["High", "Critical"] else "Neutral"})
        narratives.append({"narrative_id": str(uuid.uuid4()), "borrower_id": borrower_id, "narrative_date": narrative_date.strftime('%Y-%m-%d'),
                           "narrative_text": f"Risk profile escalated for {borrower_id} due to a '{event_type_str}' event of severity '{severity_str}'.",
                           "evidence_event_ids": json.dumps(group['event_id'].tolist())}) # Evidence includes all events for context
    return pd.DataFrame(comms), pd.DataFrame(narratives)

