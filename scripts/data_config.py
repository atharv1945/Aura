from datetime import datetime

#  SIMULATION PARAMETERS
NUM_BORROWERS = 250
TRANSACTIONS_PER_BORROWER = 200
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 9, 1)
OUTPUT_DIR = "data/raw/"
RANDOM_SEED = 42


#  BORROWER & BEHAVIOR DEFINITIONS 
DISCRETIONARY_CATEGORIES = ["Groceries", "Restaurant", "Fuel", "Online Shopping", "Luxury", "Entertainment", "Travel"]
BEHAVIORAL_ARCHETYPES = ["Frugal", "Standard", "Spender", "Impulsive"]
PAYMENT_CHANNELS = ["Credit Card", "Debit Card", "UPI", "Wallet", "Net Banking"]

#  EVENT & SCENARIO DEFINITIONS
EVENT_CHAINS = {
    "cascading_risk_pro": [
        {"type": "Income Reduction", "severity": "Medium", "delay": (20, 40), "impact_duration": 90, "macro_trigger": {"unemployment": ('>', 6.0)}},
        {"type": "EMI Missed", "severity": "High", "delay": (15, 30), "impact_duration": 60},
        {"type": "Credit Utilization Spike", "severity": "High", "delay": (30, 60), "impact_duration": 120, "branch": {"Payday Loan Taken": 0.6, "Loan Restructuring": 0.4}},
        {"type": "Payday Loan Taken", "severity": "Critical", "delay": (20, 40), "impact_duration": 90, "contagion_probability": 0.2},
        {"type": "Default", "severity": "Critical", "delay": None, "impact_duration": 365, "contagion_probability": 0.4},
    ],
    "loan_restructuring_path": [
        {"type": "Loan Restructuring", "severity": "Medium", "delay": (10, 20), "impact_duration": 180},
        {"type": "Partial Recovery", "severity": "Low", "delay": (60, 90), "impact_duration": 180},
    ],
    "late_stage_crisis": [
        {"type": "Medical Emergency", "severity": "High", "delay": (5, 10), "impact_duration": 60}, # Shorter delay
        {"type": "EMI Missed", "severity": "High", "delay": (10, 20), "impact_duration": 60}, # Shorter delay
        {"type": "Payday Loan Taken", "severity": "Critical", "delay": (5, 10), "impact_duration": 90, "contagion_probability": 0.9}, # Higher contagion
        {"type": "Default", "severity": "Critical", "delay": None, "impact_duration": 180, "contagion_probability": 0.95}, # Added Default, High contagion
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
        {"type": "Job Loss", "severity": "Critical", "delay": None, "impact_duration": 180, "contagion_probability": 0.3} 
    ],
    "fraud_anomaly": [
        {"type": "Fraudulent Transaction Detected", "severity": "High", "delay": (0, 0), "impact_duration": 30, "contagion_probability": 0.1} # Added delay and small contagion prob
    ]
}

# Assign scenarios to specific borrowers
STRESS_SCENARIOS = {
    "borrower_002": "recovery_path_pro", "borrower_005": "job_loss",
    "borrower_009": "job_loss", "borrower_015": "fraud_anomaly",
    "borrower_022": "cascading_risk_pro", "borrower_028": "cascading_risk_pro",
    "borrower_035": "cascading_risk_pro", "borrower_042": "cascading_risk_pro",
    "borrower_050": "job_loss", "borrower_061": "job_loss", "borrower_075": "cascading_risk_pro",
    "borrower_080": "cascading_risk_pro", "borrower_085": "cascading_risk_pro",
    "borrower_091": "job_loss", "borrower_092": "job_loss", "borrower_099": "cascading_risk_pro", # borrower_018 was duplicated
    "borrower_105": "job_loss", "borrower_115": "recovery_path_pro", "borrower_120": "cascading_risk_pro",
    "borrower_122": "job_loss", "borrower_130": "job_loss", "borrower_133": "cascading_risk_pro",
    "borrower_140": "cascading_risk_pro", "borrower_145": "job_loss", "borrower_149": "recovery_path_pro",
    "borrower_010": "fraud_anomaly", "borrower_018": "job_loss",
    "borrower_027": "recovery_path_pro", "borrower_036": "fraud_anomaly",
    "borrower_048": "cascading_risk_pro",
    "borrower_060": "fraud_anomaly", "borrower_066": "recovery_path_pro",
    "borrower_070": "job_loss", "borrower_082": "fraud_anomaly",
    "borrower_088": "cascading_risk_pro", "borrower_095": "job_loss",
    "borrower_100": "recovery_path_pro", "borrower_108": "fraud_anomaly",
    "borrower_112": "cascading_risk_pro", 
    "borrower_125": "fraud_anomaly", "borrower_132": "recovery_path_pro",
    "borrower_138": "job_loss", "borrower_147": "cascading_risk_pro",
    "borrower_152": "fraud_anomaly", "borrower_155": "job_loss",
    "borrower_160": "recovery_path_pro", 
    "borrower_165": "job_loss", "borrower_168": "fraud_anomaly",
    "borrower_170": "cascading_risk_pro", "borrower_175": "job_loss",
    "borrower_178": "recovery_path_pro", "borrower_180": "fraud_anomaly",
    "borrower_185": "cascading_risk_pro", "borrower_188": "job_loss",
    "borrower_190": "recovery_path_pro", "borrower_195": "fraud_anomaly",
    "borrower_198": "cascading_risk_pro", "borrower_200": "job_loss",

    "borrower_202": "fraud_anomaly", "borrower_205": "recovery_path_pro",
    "borrower_210": "job_loss", "borrower_212": "cascading_risk_pro",
    "borrower_215": "fraud_anomaly", "borrower_218": "job_loss",
    "borrower_220": "recovery_path_pro",
    "borrower_228": "fraud_anomaly", "borrower_230": "job_loss",
    "borrower_235": "recovery_path_pro", "borrower_238": "cascading_risk_pro",
    "borrower_240": "job_loss", "borrower_245": "fraud_anomaly",
    "borrower_248": "cascading_risk_pro", "borrower_250": "recovery_path_pro",
    "borrower_018": "late_stage_crisis",
    "borrower_052": "late_stage_crisis", 
    "borrower_118": "late_stage_crisis",
    "borrower_162": "late_stage_crisis",
    "borrower_225": "late_stage_crisis",
}

W_BASE = 0.4
RISK_THRESHOLD = 0.6