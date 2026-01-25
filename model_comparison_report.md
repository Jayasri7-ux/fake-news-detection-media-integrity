# Model Comparison Report

| Model | Accuracy | F1 Score | ROC AUC | Log Loss |
| --- | --- | --- | --- | --- |
| LogisticRegression | 0.8431540437456215 | 0.9092873542520146 | 0.8806566251585388 | 0.3076748599556398 |
| RandomForest | 0.8511714797228925 | 0.9170714781401804 | 0.8704415530140538 | 0.3507596831510379 |
| GradientBoosting | 0.8525725850393088 | 0.9174079888365603 | 0.8628540853592244 | 0.3292088006550001 |
| RandomForest_Calibrated | 0.8425313302716587 | 0.9089845683178117 | 0.8714527204215181 | 0.3354956029576034 |


## Statistical Analysis
Note: Detailed McNemar's test or T-test would require per-sample predictions.

**Best Model based on F1 Score:** GradientBoosting
