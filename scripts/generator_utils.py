import pandas as pd
from faker import Faker
import random
import numpy as np
from datetime import datetime, timedelta
import uuid
import json
import os

# Import configurations
import config

# Initialize Faker instance
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
    """Creates a DataFrame of deeply realistic borrower profiles."""
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
            "behavioral_archetype": random.choice(config.BEHAVIORAL_ARCHETYPES), "network_risk_flag": 0, "dti_ratio": round(random.uniform(0.2, 0.6), 2)
        })

    for _ in range(config.NUM_BORROWERS // 4):
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
        events.append({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date, "event_type": scenario, "severity": "High"})
        return events

    for event_def in chain:
        trigger = event_def.get("macro_trigger")
        if trigger:
            macro_key, (op, val) = list(trigger.items())[0]
            if current_date.date() in macro_timeline.index.date:
                if not macro_timeline.loc[macro_timeline.index.date == current_date.date(), macro_key].iloc[0] > val:
                    continue
            else:
                continue
        
        event_type = event_def["type"]
        events.append({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date, "event_type": event_type, "severity": event_def["severity"]})
        
        if event_def.get("branch"):
            branch_choice = np.random.choice(list(event_def["branch"].keys()), p=list(event_def["branch"].values()))
            if branch_choice == "Loan Restructuring":
                restructure_chain = config.EVENT_CHAINS["loan_restructuring_path"]
                for restructure_event_def in restructure_chain:
                    current_date += timedelta(days=random.randint(*restructure_event_def["delay"]))
                    events.append({"event_id": str(uuid.uuid4()), "borrower_id": borrower_id, "event_date": current_date, "event_type": restructure_event_def["type"], "severity": restructure_event_def["severity"]})
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
    
    for dt in pd.date_range(start=config.START_DATE, end=config.END_DATE, freq='MS'):
        transactions.append(create_transaction(borrower_id, dt, "credit", "Salary", monthly_income * (1 + profile['income_volatility'] * (random.random() - 0.5)), home_city))
        transactions.append(create_transaction(borrower_id, dt + timedelta(days=5), "debit", "Rent/EMI", monthly_income * profile['dti_ratio'], home_city))

    for _ in range(num_transactions):
        dt = create_random_date(config.START_DATE, config.END_DATE)
        category, amount = generate_archetype_spend(archetype, dt)
        transactions.append(create_transaction(borrower_id, dt, "debit", category, amount, home_city))

    for event in events:
        if event['event_type'] == 'Loan Restructuring': 
            for dt in pd.date_range(start=event['event_date'], end=config.END_DATE, freq='MS'):
                transactions.append(create_transaction(borrower_id, dt + timedelta(days=5), "debit", "Restructured EMI", monthly_income * profile['dti_ratio'] * 0.6, home_city))

    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    return df

def generate_archetype_spend(archetype, date, stress_factor=1.0):
    """Models spending amount and category based on behavioral archetype."""
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
    if is_weekend: amount *= 1.5
    return category, amount * stress_factor

def create_transaction(borrower_id, date, trans_type, category, amount, location, description=None):
    """Helper function to create a single transaction dictionary."""
    archetype = random.choice(config.BEHAVIORAL_ARCHETYPES) 
    channel_weights = {"Frugal": [0.1, 0.4, 0.4, 0.1, 0.0], "Spender": [0.5, 0.1, 0.2, 0.1, 0.1], "Standard": [0.3, 0.3, 0.3, 0.1, 0.0], "Impulsive": [0.4, 0.1, 0.4, 0.1, 0.0]}
    return {
        "transaction_id": str(uuid.uuid4()), "borrower_id": borrower_id, "date": date.strftime('%Y-%m-%d %H:%M:%S'),
        "type": trans_type, "category": category, "amount": round(amount, 2), "location": location,
        "description": description or category, "payment_channel": random.choices(config.PAYMENT_CHANNELS, weights=channel_weights[archetype], k=1)[0],
        **get_macroeconomic_signals(date)
    }

def create_random_date(start, end):
    """Generates a random datetime within a given range."""
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_communications_and_narratives(events):
    """Generates post-hoc communications and risk narratives from the final event log."""
    comms, narratives = [], []
    for borrower_id, group in events.groupby('borrower_id'):
        if group.empty: continue
        # Convert 'event_date' to datetime objects for sorting
        group['event_date'] = pd.to_datetime(group['event_date'])
        critical_event = group.sort_values('event_date').iloc[-1]
        
        comm_date = critical_event['event_date'] + timedelta(days=5)
        narrative_date = critical_event['event_date']
        
        comms.append({"comm_id": str(uuid.uuid4()), "borrower_id": borrower_id, "comm_date": comm_date.strftime('%Y-%m-%d'),
                      "comm_channel": "Email", "comm_text": f"Regarding event: {critical_event['event_type']}", "sentiment": "Negative" if critical_event['severity'] in ["High", "Critical"] else "Neutral"})
        narratives.append({"narrative_id": str(uuid.uuid4()), "borrower_id": borrower_id, "narrative_date": narrative_date.strftime('%Y-%m-%d'),
                           "narrative_text": f"Risk detected for {borrower_id} due to a critical event '{critical_event['event_type']}' of severity {critical_event['severity']}.",
                           "evidence_event_ids": json.dumps(group['event_id'].tolist())})
    return pd.DataFrame(comms), pd.DataFrame(narratives)