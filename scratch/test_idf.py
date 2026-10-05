import pandas as pd
import numpy as np

def _explode(df: pd.DataFrame, text_col: str, min_token_length: int) -> pd.DataFrame:
    tokens = df[text_col].str.split(" ")
    long = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(), "token": tokens.to_numpy()})
    long = long.explode("token", ignore_index=True)
    long = long[long["token"].notna() & (long["token"].str.len() >= min_token_length)]
    return long

s1_c = pd.DataFrame({"entity_id": [1, 2], "text": ["apple banana", "apple cherry"]})
other_c = pd.DataFrame({"entity_id": [3, 4, 5], "text": ["apple banana", "apple date", "banana cherry"]})

text_col = "text"
min_token_length = 2

other_long = _explode(other_c, text_col, min_token_length)
part_size_other = other_c["entity_id"].nunique()
token_df = other_long.groupby("token")["entity_id"].nunique()

print(token_df)
cap_other = 10
valid_tokens_other = token_df[token_df <= cap_other].index
other_long = other_long[other_long["token"].isin(valid_tokens_other)]

idf = np.log((part_size_other + 1) / (token_df[valid_tokens_other] + 1)) + 1.0
idf_df = idf.reset_index()
idf_df.columns = ["token", "idf"]

print(idf_df)

s1_long = _explode(s1_c, text_col, min_token_length)
part_size_s1 = s1_c["entity_id"].nunique()
s1_token_df = s1_long.groupby("token")["entity_id"].nunique()

cap_s1 = 10
valid_tokens_s1 = s1_token_df[s1_token_df <= cap_s1].index
s1_long = s1_long[s1_long["token"].isin(valid_tokens_s1)]

s1_long = s1_long.merge(idf_df, on="token", how="inner")
print(s1_long)
