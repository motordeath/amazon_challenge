path = r'D:\ml_amazon\EXPERIMENT_LOG.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('NearestNeighbors(metric="cosine", algorithm="brute")', 'TruncatedSVD(n_components=64) + faiss.IndexFlatIP(64)')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
