import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

# Generate dummy data
print("Generating data...")
N_s1 = 10000
N_s2 = 500000
vocab_size = 20000

# Random sparse matrices
np.random.seed(42)
s1_matrix = sp.random(N_s1, vocab_size, density=0.005, format='csr')
s2_matrix = sp.random(N_s2, vocab_size, density=0.005, format='csr')

print("Fitting NN...")
t0 = time.time()
nn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute', n_jobs=-1)
nn.fit(s2_matrix)
print(f"Fit took {time.time()-t0:.2f}s")

print("Querying NN...")
t0 = time.time()
distances, indices = nn.kneighbors(s1_matrix)
print(f"Query took {time.time()-t0:.2f}s")
