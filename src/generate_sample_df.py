from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# config
n_rows = 1000
seed = 12345

def generate_sample_df(
        n_rows: int = 5,
        seed: int = 12345,
) -> pd.DataFrame:
    """
    Generate a sample dataframe with n rows and 8 columns:
    - index
    - timestamp
    - uniform
    - normal
    - exponential
    - gender
    - risk_level
    - categories
    """
    # create empty dict to store the data
    df = {}
    # set the random seed for reproducibility
    rng = np.random.default_rng(seed)

    # create the columns of data
    index = range(0, n_rows)
    timestamp = [datetime(2020,1,1) + timedelta(days= days) for days in range(0, n_rows)]
    uniform = rng.uniform(low = 0, high = 1, size = n_rows)
    normal = rng.normal(loc = 0, scale = 1, size = n_rows)
    exponential = rng.exponential(scale = 2, size = n_rows)
    gender = rng.choice(['female', 'male'], size = n_rows, p=[0.05, 0.95])
    categories = rng.choice(['category_'+str(n) for n in range(0,5)], size = n_rows)
    risk_level = rng.choice(['low', 'med','high'], size = n_rows)

    # create a dict to store the data
    df_dict = {
        'index':index,
        'timestamp':timestamp,
        'uniform':uniform,
        'normal':normal,
        'exponential':exponential,
        'gender':gender,
        'risk_level':risk_level,
        'categories':categories
    }

    # convert the dict to a dataframe
    df = pd.DataFrame(df_dict)

    # convert the risk_level column to a categorical variable with an order
    df['risk_level'] = pd.Categorical(
        df['risk_level'],
        categories = ['low','med','high'],
        ordered = True
        )

    return df
