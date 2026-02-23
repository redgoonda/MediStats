import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import statsmodels.api as sm


def parse_binary(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "y": 1,
        "n": 0,
    }
    mapped = normalized.map(mapping)

    if mapped.isna().any():
        # Try numeric fallback
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().isin([0, 1]).all():
            return numeric
        raise ValueError(
            "Outcome column could not be mapped to a binary variable. "
            "Use values like 0/1, yes/no, true/false."
        )

    return mapped


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        missing_pct = (missing / len(df)) * 100 if len(df) else 0

        if pd.api.types.is_numeric_dtype(s):
            rows.append(
                {
                    "variable": col,
                    "type": "numeric",
                    "count": int(s.notna().sum()),
                    "missing_n": missing,
                    "missing_%": round(missing_pct, 2),
                    "mean": s.mean(),
                    "std": s.std(),
                    "median": s.median(),
                    "q1": s.quantile(0.25),
                    "q3": s.quantile(0.75),
                }
            )
        else:
            top = s.mode(dropna=True)
            top_val = top.iloc[0] if not top.empty else np.nan
            top_freq = int((s == top_val).sum()) if pd.notna(top_val) else 0
            rows.append(
                {
                    "variable": col,
                    "type": "categorical",
                    "count": int(s.notna().sum()),
                    "missing_n": missing,
                    "missing_%": round(missing_pct, 2),
                    "top_category": top_val,
                    "top_freq": top_freq,
                    "n_unique": int(s.nunique(dropna=True)),
                }
            )

    return pd.DataFrame(rows)


def run_ttest(df: pd.DataFrame, group_col: str, outcome_col: str):
    data = df[[group_col, outcome_col]].dropna()
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError("Group column must contain exactly 2 groups for t-test.")

    g1 = data.loc[data[group_col] == groups[0], outcome_col]
    g2 = data.loc[data[group_col] == groups[1], outcome_col]

    stat, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")

    return {
        "group_1": groups[0],
        "group_2": groups[1],
        "n_1": len(g1),
        "n_2": len(g2),
        "mean_1": float(np.mean(g1)),
        "mean_2": float(np.mean(g2)),
        "t_stat": float(stat),
        "p_value": float(p),
    }


def run_chi_square(df: pd.DataFrame, col_a: str, col_b: str):
    data = df[[col_a, col_b]].dropna()
    table = pd.crosstab(data[col_a], data[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(table)

    return {
        "contingency_table": table,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
    }


def run_logistic_regression(df: pd.DataFrame, outcome: str, predictors: list[str]):
    data = df[[outcome] + predictors].dropna().copy()
    y = parse_binary(data[outcome])

    X = pd.get_dummies(data[predictors], drop_first=True)
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    params = result.params
    conf = result.conf_int()
    or_table = pd.DataFrame(
        {
            "coef": params,
            "odds_ratio": np.exp(params),
            "ci_lower": np.exp(conf[0]),
            "ci_upper": np.exp(conf[1]),
            "p_value": result.pvalues,
        }
    )

    return result, or_table


def main():
    st.set_page_config(page_title="MediStats", layout="wide")
    st.title("🩺 MediStats: Medical Statistics App")
    st.caption("Upload a CSV and run common analyses for clinical datasets.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    st.subheader("Dataset Preview")
    st.write(f"Rows: **{len(df)}**, Columns: **{len(df.columns)}**")
    st.dataframe(df.head(20), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Descriptive", "T-test", "Chi-square", "Logistic Regression"]
    )

    with tab1:
        st.markdown("### Descriptive Summary")
        summary = descriptive_table(df)
        st.dataframe(summary, use_container_width=True)

    with tab2:
        st.markdown("### Independent Two-Sample T-test")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available for outcome selection.")
        else:
            group_col = st.selectbox("Group column (2 groups)", cat_cols, key="tt_group")
            outcome_col = st.selectbox("Numeric outcome column", numeric_cols, key="tt_outcome")
            if st.button("Run T-test"):
                try:
                    result = run_ttest(df, group_col, outcome_col)
                    st.json(result)
                except Exception as e:
                    st.error(str(e))

    with tab3:
        st.markdown("### Chi-square Test of Independence")
        cat_cols = df.columns.tolist()
        col_a = st.selectbox("Categorical variable A", cat_cols, key="chi_a")
        col_b = st.selectbox("Categorical variable B", cat_cols, key="chi_b")

        if st.button("Run Chi-square"):
            try:
                result = run_chi_square(df, col_a, col_b)
                st.write("Contingency table")
                st.dataframe(result["contingency_table"], use_container_width=True)
                st.write(
                    {
                        "chi2": result["chi2"],
                        "dof": result["dof"],
                        "p_value": result["p_value"],
                    }
                )
                st.write("Expected frequencies")
                st.dataframe(result["expected"], use_container_width=True)
            except Exception as e:
                st.error(str(e))

    with tab4:
        st.markdown("### Logistic Regression (Binary Outcome)")
        cols = df.columns.tolist()
        outcome = st.selectbox("Binary outcome column", cols, key="lr_outcome")
        predictors = st.multiselect(
            "Predictor columns",
            [c for c in cols if c != outcome],
            default=[c for c in cols if c != outcome][:2],
            key="lr_predictors",
        )

        if st.button("Run Logistic Regression"):
            if not predictors:
                st.warning("Select at least one predictor.")
            else:
                try:
                    result, or_table = run_logistic_regression(df, outcome, predictors)
                    st.write("Model summary")
                    st.text(result.summary().as_text())
                    st.write("Odds ratios")
                    st.dataframe(or_table, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


if __name__ == "__main__":
    main()
