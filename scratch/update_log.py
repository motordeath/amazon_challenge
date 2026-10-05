import sys

path = r'D:\ml_amazon\EXPERIMENT_LOG.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

results = """- **Results**:
  - **Blocking Metrics**:
    - Uncapped blocking recall: **0.6782** (up from 0.6675 in Exp1)
    - Capped blocking recall: **0.5452** (up from 0.5308 in Exp1)
    - Uncapped candidate pairs: 29,629,557 (avg 60.2 per S1)
    - Capped candidate pairs: 11,424,166 (avg 23.2 per S1)
    - Positive candidate pairs: 943,578 (up from 918,614)
    - Negative candidate pairs: 10,480,588
  - **Validation Metrics**:
    - Decision threshold $\\tau$: **0.9100**
    - **Macro $F_{0.5}$**: **0.6904** (up from 0.6833 in Exp1, and 0.6484 in Baseline)
    - Macro Precision: **0.7876** (up from 0.7841)
    - Macro Recall: **0.5470** (up from 0.5344)
    - Singleton Accuracy: **92.95%** (down slightly from 93.25%)
  - **Runtime & Memory**:
    - Blocking Runtime: 1175s (~19.6m)
    - Total Runtime: 2604s (~43.4m)
    - Blocking Peak RSS: ~16.3 GB
- **Conclusion**:
  - **Hypothesis Validated**: Implementing IDF weighting successfully penalized hyper-common tokens and rewarded discriminative tokens. This improved candidate ranking significantly, rescuing an additional 1.5% (absolute) of true positive pairs from being dropped during the top-25 truncation (and even improved the pre-cap channel truncation).
  - F0.5 climbed from 0.683 to 0.690 with virtually no increase in candidate volume or training runtime.
- **Next Action**:
  - Wait for user authorization for the next experiment. The pipeline is now operating at a solid 0.690 F0.5."""

content = content.replace('- **Results**:\n  - [PENDING]', results)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
