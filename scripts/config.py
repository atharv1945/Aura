from datetime import datetime

# --- 1. SIMULATION PARAMETERS ---
NUM_BORROWERS = 50
TRANSACTIONS_PER_BORROWER = 200
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 9, 1)
OUTPUT_DIR = "data/raw/"
RANDOM_SEED = 42

# --- 2. BORROWER & BEHAVIOR DEFINITIONS ---
DISCRETIONARY_CATEGORIES = ["Groceries", "Restaurant", "Fuel", "Online Shopping", "Luxury", "Entertainment", "Travel"]
BEHAVIORAL_ARCHETYPES = ["Frugal", "Standard", "Spender", "Impulsive"]
PAYMENT_CHANNELS = ["Credit Card", "Debit Card", "UPI", "Wallet", "Net Banking"]

# --- 3. EVENT & SCENARIO DEFINITIONS ---
EVENT_CHAINS = {
    "cascading_risk_pro": [
        {"type": "Income Reduction", "severity": "Medium", "delay": (20, 40), "impact_duration": 90, "macro_trigger": {"unemployment": ('>', 6.0)}},
        {"type": "EMI Missed", "severity": "High", "delay": (15, 30), "impact_duration": 60},
        {"type": "Credit Utilization Spike", "severity": "High", "delay": (30, 60), "impact_duration": 120, "branch": {"Payday Loan Taken": 0.6, "Loan Restructuring": 0.4}},
        {"type": "Payday Loan Taken", "severity": "Critical", "delay": (20, 40), "impact_duration": 90},
        {"type": "Default", "severity": "Critical", "delay": None, "impact_duration": 365},
    ],
    "loan_restructuring_path": [
        {"type": "Loan Restructuring", "severity": "Medium", "delay": (10, 20), "impact_duration": 180},
        {"type": "Partial Recovery", "severity": "Low", "delay": (60, 90), "impact_duration": 180},
    ],
    "recovery_path_pro": [
        {"type": "Salary Hike", "severity": "Low", "delay": (30, 90), "impact_duration": 365},
        {"type": "Bonus Received", "severity": "Low", "delay": (15, 30), "impact_duration": 60},
        {"type": "Loan Prepayment", "severity": "Low", "delay": None, "impact_duration": 0},
    ], 
    "high risk spending": [
        {"type": "High Risk Spending", "severity": "High", "delay": (5,10), "impact_duration": 30}
    ],
    "job_loss": [
        {"type": "Job Loss", "severity": "Critical", "delay": None, "impact_duration": 180}
    ]
}

# Assign scenarios to specific borrowers
STRESS_SCENARIOS = {
    "borrower_002": "recovery_path_pro", "borrower_005": "high_risk_spending", "borrower_009": "job_loss",
    "borrower_015": "fraud_anomaly", "borrower_022": "cascading_risk_pro", "borrower_028": "cascading_risk_pro",
    "borrower_035": "cascading_risk_pro", "borrower_042": "cascading_risk_pro",
}