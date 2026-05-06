import pandas as pd
from src.generate_sample_df import generate_sample_df

def test_generate_sample_df():
    df = generate_sample_df(n_rows=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    expected_cols = ['index', 'timestamp', 'uniform', 'normal', 'exponential', 'gender', 'risk_level', 'categories']
    for col in expected_cols:
        assert col in df.columns
