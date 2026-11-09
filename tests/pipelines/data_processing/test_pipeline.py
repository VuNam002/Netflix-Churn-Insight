import pandas as pd

from netflix.pipelines.data_processing.nodes import clean_data


def test_clean_data_drops_duplicates_and_invalid_rows(raw_customers, dp_params):
    raw = raw_customers.head(10)
    raw = pd.concat([raw, raw.iloc[[0]]])  # duplicated customer
    raw.loc[1, "age"] = -5  # impossible age
    raw.loc[2, "monthly_fee"] = None  # missing value

    clean = clean_data(raw, dp_params)

    assert len(clean) == 8  # 10 customers minus the invalid age and the missing fee
    assert clean["customer_id"].is_unique
    assert clean["churned"].dtype == "int64"


def test_clean_data_works_without_target_column(raw_customers, dp_params):
    clean = clean_data(raw_customers.drop(columns="churned"), dp_params)

    assert "churned" not in clean.columns
    assert len(clean) == len(raw_customers)
