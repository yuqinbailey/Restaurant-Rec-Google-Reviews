import pandas as pd


def get_resturants(data, res_df):
    pass
    pd.merge(data, res_df, on='user_id', how='inner')
    return data