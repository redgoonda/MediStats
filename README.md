# MediStats App

MediStats is a lightweight Streamlit application for common medical statistics workflows:

- Data upload (CSV)
- Cohort overview and missingness profile
- Descriptive statistics
- Group comparison:
  - Numeric outcome: independent two-sample t-test
  - Categorical outcome: chi-square test
- Simple logistic regression for binary outcomes

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

## Expected dataset format

- CSV with column headers
- Numeric and categorical columns supported
- For logistic regression, outcome must be binary (`0/1`, `yes/no`, `true/false`)

## Notes

This app is for educational and exploratory analysis. It does **not** replace clinical statistical oversight.
