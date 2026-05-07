import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_initial_df
import pandas as pd

def test_create_initial_df():
    '''test that it retuns existing df from data/raw/'''
    # setup 
    df = create_initial_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    assert "Unnamed: 0" not in df.columns
    print(df.head())