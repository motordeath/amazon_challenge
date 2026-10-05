import gc
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

def vector_candidate_pairs(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    text_col: str,
    channel_name: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Retrieve top-K candidates using character n-gram TF-IDF vectors + SVD + Dense NearestNeighbors."""
    
    countries = sorted(set(s1_df["country"].dropna().astype(str)) & set(other_df["country"].dropna().astype(str)))
    results = []
    
    for country in countries:
        s1_c = s1_df.loc[s1_df["country"] == country, ["entity_id", text_col]].dropna(subset=[text_col])
        other_c = other_df.loc[other_df["country"] == country, ["entity_id", text_col]].dropna(subset=[text_col])
        
        if s1_c.empty or other_c.empty:
            continue
            
        logger.info(f"[{channel_name}] Processing country {country}: {len(s1_c)} queries vs {len(other_c)} targets")
        
        # 1. Build TF-IDF vectors
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=0.5, min_df=2)
        other_matrix = vectorizer.fit_transform(other_c[text_col])
        s1_matrix = vectorizer.transform(s1_c[text_col])
        
        # 2. Dimensionality reduction
        n_components = min(64, other_matrix.shape[1] - 1)
        if n_components < 1:
            continue
            
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        other_dense = svd.fit_transform(other_matrix).astype(np.float32)
        s1_dense = svd.transform(s1_matrix).astype(np.float32)
        
        # 3. Exact Nearest Neighbors on dense 64D vectors (very fast via BLAS)
        n_neighbors = min(top_k, len(other_dense))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(other_dense)
        
        distances, indices = nn.kneighbors(s1_dense)
        
        # 4. Assemble results
        s1_ids = s1_c["entity_id"].to_numpy()
        other_ids = other_c["entity_id"].to_numpy()
        
        s1_id_repeated = s1_ids.repeat(n_neighbors)
        other_id_flat = other_ids[indices.flatten()]
        # metric='cosine' returns distance, so score is 1 - distance
        score_flat = 1.0 - distances.flatten()
        
        part = pd.DataFrame({
            "source1_entity_id": s1_id_repeated,
            "candidate_entity_id": other_id_flat,
            "score": score_flat
        })
        
        results.append(part)
        
        del s1_c, other_c, vectorizer, other_matrix, s1_matrix, svd, other_dense, s1_dense, nn, distances, indices
        gc.collect()

    if not results:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "channel", "score"])
        
    out = pd.concat(results, ignore_index=True)
    out["channel"] = channel_name
    return out[["source1_entity_id", "candidate_entity_id", "channel", "score"]]
