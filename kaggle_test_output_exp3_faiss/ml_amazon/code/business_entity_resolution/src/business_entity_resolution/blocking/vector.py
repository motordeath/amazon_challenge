import gc
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

def vector_candidate_pairs(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    text_col: str,
    channel_name: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Retrieve top-K candidates using character n-gram TF-IDF vectors + SVD + FAISS."""
    import faiss  # Import here to avoid failing if not installed locally
    
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
        
        # L2 normalize for cosine similarity
        faiss.normalize_L2(other_dense)
        faiss.normalize_L2(s1_dense)
        
        # 3. FAISS Exact Inner Product (Cosine) Search
        index = faiss.IndexFlatIP(n_components)
        index.add(other_dense)
        
        n_neighbors = min(top_k, len(other_dense))
        distances, indices = index.search(s1_dense, n_neighbors)
        
        # 4. Assemble results
        s1_ids = s1_c["entity_id"].to_numpy()
        other_ids = other_c["entity_id"].to_numpy()
        
        s1_id_repeated = s1_ids.repeat(n_neighbors)
        other_id_flat = other_ids[indices.flatten()]
        score_flat = distances.flatten()  # IP distance of normalized vectors is cosine similarity
        
        part = pd.DataFrame({
            "source1_entity_id": s1_id_repeated,
            "candidate_entity_id": other_id_flat,
            "score": score_flat
        })
        
        results.append(part)
        
        del s1_c, other_c, vectorizer, other_matrix, s1_matrix, svd, other_dense, s1_dense, index, distances, indices
        gc.collect()

    if not results:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "channel", "score"])
        
    out = pd.concat(results, ignore_index=True)
    out["channel"] = channel_name
    return out[["source1_entity_id", "candidate_entity_id", "channel", "score"]]
