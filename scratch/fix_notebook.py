import json

with open('kaggle_test/abb-aar-ya-paar.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Put pip install cell as a new cell right before cell 3
pip_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": "!pip install -q \"rapidfuzz>=3.6\""
}

nb['cells'].insert(3, pip_cell)

with open('kaggle_test/abb-aar-ya-paar.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Inserted pip install cell at index 3 successfully. Total cells:", len(nb['cells']))
