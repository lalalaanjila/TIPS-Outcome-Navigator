"""Web labels only. VERIFY ALL UNITS AGAINST YOUR OWN TRAINING CSV BEFORE RELEASE.
The Notebook provides feature names but not physical measurement units.
"""
FEATURES = {
    "Age": {
        "zh": "Age", "en": "Age", "unit": "years", "step": 1.0,
        "minimum": 18.0, "maximum": 75.0, "help": "适用Age18–75岁",
    },
    "PLT": {
        "zh": "Platelet count", "en": "Platelet count", "unit": "×10⁹/L", "step": 1.0,
        "minimum": 10.0, "maximum": 2000.0, "help": "Units: ×10⁹/L",
    },
    "Hb": {
        "zh": "Hemoglobin", "en": "Hemoglobin", "unit": "g/L", "step": 1.0,
        "minimum": 40.0, "maximum": 300.0, "help": "Units: g/L",
    },
    "Alb": {
        "zh": "Albumin", "en": "Albumin", "unit": "g/L", "step": 0.1,
        "minimum": 15.0, "maximum": 100.0, "help": "Units: g/L",
    },
    "TBIL": {
        "zh": "Total bilirubin", "en": "Total bilirubin", "unit": "μmol/L", "step": 0.1,
        "minimum": 0.01, "maximum": 2000.0, "help": "Units: μmol/L",
    },
    "Scr": {
        "zh": "Serum creatinine", "en": "Serum creatinine", "unit": "μmol/L", "step": 1.0,
        "minimum": 0.01, "maximum": 2500.0, "help": "Units: μmol/L",
    },
    "INR": {
        "zh": "International normalized ratio", "en": "International normalized ratio", "unit": "", "step": 0.01,
        "minimum": 0.01, "maximum": 20.0, "help": "INR (unitless)",
    },
    "PT_prolongation": {
        "zh": "PT prolongation", "en": "PT prolongation", "unit": "s", "step": 0.1,
        "minimum": -5.0, "maximum": 180.0, "help": "Prothrombin time prolongation in seconds; negative values are permitted",
    },
}

OUTCOME_LABELS = {
    "Rebleeding_365": ("Rebleeding", "Rebleeding", "#2563eb"),
    "OHE_365": ("Overt hepatic encephalopathy", "Overt hepatic encephalopathy", "#8b5cf6"),
    "Mortality_365": ("All-cause mortality", "All-cause mortality", "#ea580c"),
}
