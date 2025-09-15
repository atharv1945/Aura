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

    for _ in range(config.NUM_BORROWERS // 3):
        p1_idx, p2_idx = random.sample(range(len(profiles)), 2)
        if profiles[p1_idx]['linked_borrower_id'] is None and profiles[p2_idx]['linked_borrower_id'] is None:
            profiles[p1_idx]['linked_borrower_id'] = profiles[p2_idx]['borrower_id']
            profiles[p2_idx]['linked_borrower_id'] = profiles[p1_idx]['borrower_id']
    df = pd.DataFrame(profiles)
    df['credit_limit'] = df['credit_score'] * np.random.uniform(50, 150, size=len(df))
    return df

def generate_full_event_timeline(borrower_id, scenario, macro_timeline):
    """Generates a sequence of events for a borrower based on scenarios and macro triggers."""
    events = []
    if not scenario: return events
    current_date = create_random_date(datetime(2025, 1, 1), datetime(2025, 4, 1))
    chain = config.EVENT_CHAINS.get(scenario)
    if not chain:
        events.append({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date, "event_type": scenario, "severity": "High", "impact_duration": 90, "contagion_probability": 0.1})
        return events

    for event_def in chain:
        trigger = event_def.get("macro_trigger")
        if trigger:
            macro_key, (op, val) = list(trigger.items())[0]
            macro_value = macro_timeline.loc[macro_timeline.index.date == current_date.date(), macro_key]
            if not macro_value.empty and not macro_value.iloc[0] > val:
                continue

        event_def_copy = event_def.copy()
        if 'type' in event_def_copy:
            event_def_copy['event_type'] = event_def_copy.pop('type')
        event_def_copy.update({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date})
        events.append(event_def_copy)

        if event_def.get("branch"):
            branch_choice = np.random.choice(list(event_def["branch"].keys()), p=list(event_def["branch"].values()))
            if branch_choice == "Loan Restructuring":
                restructure_chain = config.EVENT_CHAINS["loan_restructuring_path"]
                for restructure_event_def in restructure_chain:
                    current_date += timedelta(days=random.randint(*restructure_event_def["delay"]))
                    restructure_event_def_copy = restructure_event_def.copy()
                    restructure_event_def_copy.update({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date})
                    events.append(restructure_event_def_copy)
                break 
            elif branch_choice == "Recovery": break

        if not event_def["delay"]: break
        current_date += timedelta(days=random.randint(*event_def["delay"]))
    return events


def generate_transactions_for_borrower(borrower_id, profile, events, num_transactions):
    """Generates a realistic set of transactions based on profile, events, and behavior."""
    transactions = []
    home_city = profile['city']
    archetype = profile['behavioral_archetype']
    monthly_income = profile['monthly_income']

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df['event_date'] = pd.to_datetime(events_df['event_date'])
        events_df['impact_end_date'] = events_df.apply(lambda row: row['event_date'] + timedelta(days=row.get('impact_duration', 0)), axis=1)

    for dt in pd.date_range(start=config.START_DATE, end=config.END_DATE, freq='D'):
        active_events = pd.DataFrame()
        if not events_df.empty:
            active_events = events_df[(events_df['event_date'] <= dt) & (events_df['impact_end_date'] >= dt)]
        
        stress_factor = 1.0
        if not active_events.empty:
            if any(s in active_events['severity'].tolist() for s in ['High', 'Critical']):
                stress_factor = 0.6 

        if dt.day == 1:
            transactions.append(create_transaction(borrower_id, dt, "credit", "Salary", monthly_income, home_city, profile))
        if dt.day == 5:
            transactions.append(create_transaction(borrower_id, dt, "debit", "Rent/EMI", monthly_income * profile['dti_ratio'], home_city, profile))
        
        if random.random() < 0.5:
            category, amount = generate_archetype_spend(archetype, dt, stress_factor)
            transactions.append(create_transaction(borrower_id, dt, "debit", category, amount, home_city, profile))

    df = pd.DataFrame(transactions)
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
    else: 
        category = random.choice(config.DISCRETIONARY_CATEGORIES)
        amount = random.uniform(200, 5000)
    
    if archetype == "Impulsive": amount *= 1.2
    if is_weekend and category in ["Restaurant", "Entertainment"]: amount *= 1.5
    
    if category in ["Luxury", "Entertainment", "Restaurant", "Online Shopping"]:
        amount *= stress_factor
        
    return category, amount

def create_transaction(borrower_id, date, trans_type, category, amount, location, profile, description=None):
    """Helper function to create a single transaction dictionary."""
    channel_weights = {"Frugal": [0.1, 0.4, 0.4, 0.1, 0.0], "Spender": [0.5, 0.1, 0.2, 0.1, 0.1], "Standard": [0.3, 0.3, 0.3, 0.1, 0.0], "Impulsive": [0.4, 0.1, 0.4, 0.1, 0.0]}
    return {
        "transaction_id": str(uuid.uuid4()), "borrower_id": borrower_id, "date": date.strftime('%Y-%m-%d %H:%M:%S'),
        "type": trans_type, "category": category, "amount": round(amount, 2), "location": location,
        "description": description or category, "payment_channel": random.choices(config.PAYMENT_CHANNELS, weights=channel_weights[profile['behavioral_archetype']], k=1)[0],
        **get_macroeconomic_signals(date)
    }

def create_random_date(start, end):
    """Generates a random datetime within a given range."""
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_communications_and_narratives(events_df):
    """Generates post-hoc communications and risk narratives from the final event log."""
    comms, narratives = [], []
    for borrower_id, group in events_df.groupby('borrower_id'):
        if group.empty: continue
        
        group['event_date'] = pd.to_datetime(group['event_date'])

        group['severity_score'] = group['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}).fillna(0)
        most_severe_event = group.loc[group['severity_score'].idxmax()]

        comm_date = most_severe_event['event_date'] + timedelta(days=5)
        narrative_date = most_severe_event['event_date']

        comms.append({"comm_id": str(uuid.uuid4()), "borrower_id": borrower_id, "comm_date": comm_date.strftime('%Y-%m-%d'),
                      "comm_channel": "Email", "comm_text": f"Regarding your account status after event: {most_severe_event['event_type']}", "sentiment": "Negative" if most_severe_event['severity'] in ["High", "Critical"] else "Neutral"})
        narratives.append({"narrative_id": str(uuid.uuid4()), "borrower_id": borrower_id, "narrative_date": narrative_date.strftime('%Y-%m-%d'),
                           "narrative_text": f"Risk profile escalated for {borrower_id} due to a '{most_severe_event['event_type']}' event of severity '{most_severe_event['severity']}'.",
                           "evidence_event_ids": json.dumps(group['event_id'].tolist())})
    return pd.DataFrame(comms), pd.DataFrame(narratives)

