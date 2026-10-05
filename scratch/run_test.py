import sys
from pathlib import Path
sys.path.insert(0, r"D:\ml_amazon\code\business_entity_resolution\src")

import pandas as pd
from business_entity_resolution.blocking.token import token_candidate_pairs

s1 = pd.DataFrame({"entity_id": [1, 2], "country": ["US", "US"], "name_normalized": ["apple banana inc", "apple cherry ltd"]})
other = pd.DataFrame({"entity_id": [3, 4, 5], "country": ["US", "US", "US"], "name_normalized": ["apple banana llc", "apple date corp", "banana cherry inc"]})

res = token_candidate_pairs(
    s1, other, "name_normalized", "test",
    min_token_length=2, max_df_fraction=1.0, max_df_abs_other=10, max_df_abs_s1=10, top_k_per_s1=5
)
print(res)
