import json

with open('kaggle_test/abb-aar-ya-paar.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_8_code = """from pathlib import Path
import shutil

SRC = Path("/kaggle/input/datasets/fuckinggpu/business-entity-resolution-source")
if not SRC.exists():
    SRC = Path("/kaggle/input/business-entity-resolution-source")
WORK = Path("/kaggle/working/ml_amazon")

# Remove only the partially-created code directory
if (WORK / "code").exists():
    shutil.rmtree(WORK / "code")

(WORK / "code").mkdir(parents=True, exist_ok=True)

if (SRC / "code").exists():
    shutil.copytree(SRC / "code", WORK / "code", dirs_exist_ok=True)
elif (SRC / "business_entity_resolution").exists():
    shutil.copytree(SRC / "business_entity_resolution", WORK / "code" / "business_entity_resolution", dirs_exist_ok=True)
else:
    for p in SRC.glob("*"):
        if p.is_dir() and "business_entity_resolution" in p.name:
            shutil.copytree(p, WORK / "code" / "business_entity_resolution", dirs_exist_ok=True)
            break
        elif (p / "src").exists():
            shutil.copytree(p, WORK / "code" / p.name, dirs_exist_ok=True)

print("Source copied successfully.")
print("\\nPython files:")

for p in sorted((WORK / "code").rglob("*.py")):
    print(p.relative_to(WORK))
"""

nb['cells'][8]['source'] = cell_8_code

with open('kaggle_test/abb-aar-ya-paar.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated cell 8 successfully.")
