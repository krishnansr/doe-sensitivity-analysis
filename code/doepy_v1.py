import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from doepy import build

"""
from pyDOE import fullfact, ff2n, fracfact
print(fullfact([2, 4, 3]))
print(ff2n(3))
print(fracfact('a b ab'))
"""

CSV_PATH = r"data/syncrel_fasr_data_cleaned.csv"
factor_dict = {
    'f1' : [1, -1], 'f2': [1, -1], 'f3': [1, -1], 'f4': [1, -1], 'f5': [1, -1],
    'f6' : [1, -1], 'f7': [1, -1], 'f8': [1, -1], 'f9': [1, -1], 'f10': [1, -1],
    'f11': [1, -1], 'f12': [1, -1], 'f13': [1, -1]
}


df_labelled = pd.read_csv(CSV_PATH)
def map_data(df_doe, num_factors=13):
    # when output = nan we have to collect data since its not present
    df_intersection = pd.merge(df_labelled, df_doe, how='right', on=[f'f{i+1}' for i in range(num_factors)])
    return df_intersection


# 1. PLACKETT BURMAN
df_plackett = build.build_plackett_burman(factor_dict)
df_plackett_mapped = map_data(df_plackett.astype(np.int64))
print(f"{df_plackett_mapped['output1'].isna().sum()} rows not filled in {len(df_plackett_mapped)}")

# 2. FRACTIONAL FACTORIAL
df_frac_fact = build.frac_fact_res(factor_dict, res=4)
df_frac_fact_mapped = map_data(df_frac_fact.astype(np.int64))
print(f"{df_frac_fact_mapped['output1'].isna().sum()} rows not filled in {len(df_frac_fact_mapped)}")

# 3. FULL FACTORIAL
df_full_fact = build.build_full_fact(factor_dict)
df_full_fact_mapped = map_data(df_full_fact.astype(np.int64))
print(f"{df_full_fact_mapped['output1'].isna().sum()} rows not filled in {len(df_full_fact_mapped)}")


"""
# 4. LATIN HYPERCUBE
df_lhs = build.build_space_filling_lhs(factor_dict)
df_lhs_mapped = map_data(df_lhs.astype(np.int64))
print(df_lhs_mapped)
"""
