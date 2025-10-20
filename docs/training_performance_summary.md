# Training Performance Summary

This note explains the content of the *"Training performance summary"* block that appears when you run `main.py` or call `RunLogisticRegression.evaluate_*` directly. It helps you understand the current model without jumping into notebooks or separate analysis scripts.

## How to run it

1. Make sure the training data is available under `data/train.jsonl`.
2. Start the training run:
   ```bash
   python main.py
   ```
3. Right after the model finishes fitting, the performance block is printed to the terminal.

### Example output

```
Training performance summary:

Accuracy: 0.8249

Coefficients:

        Estimate Std. Error  z value  Pr(>|z|)   
(Intercept)     -0.1655     0.0609   -2.718   0.00658 **
p2_avg_status     2.0629     0.2790    7.400   1.35e-13 ***
p1_avg_status    -2.0349     0.2879   -7.069   1.55e-12 ***
p2_hp_loss        1.9740     0.2565    7.696   1.42e-14 ***
p1_hp_loss       -1.3188     0.2114   -6.240   4.33e-10 ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Confusion Matrix:

        pred_0  pred_1
true_0    4155     845
true_1     906    4094

Classification Report:

              precision    recall  f1-score   support

           0       0.82      0.83      0.83      5000
           1       0.83      0.82      0.82      5000

    accuracy                           0.82     10000
   macro avg       0.82      0.82      0.82     10000
weighted avg       0.82      0.82      0.82     10000
```

## What each section tells you

### Accuracy
- Scalar between 0 and 1, computed on the same features/targets used for training.
- Useful for a quick snapshot, but remember: without a held-out validation set this measures training performance, not generalisation.

### Confusion Matrix
- Shows the prediction counts for each combination of true class (`true_*`) and predicted class (`pred_*`).
- Good models produce large diagonal values (`true_0`/`pred_0`, `true_1`/`pred_1`) and small off-diagonal values.
- You can further process it using the `pandas.DataFrame` returned by `report.confusion_matrix`.

### Classification Report
- Text output from `sklearn.metrics.classification_report` with `precision`, `recall`, `f1-score`, and `support` per class.
- `macro avg` and `weighted avg` help assess performance on imbalanced datasets.
- Watch for large gaps between precision and recall, which can highlight specific misclassification patterns.
- **Precision**: share of predicted positives that are actually correct (TP / (TP + FP)).
- **Recall**: share of actual positives that the model recovers (TP / (TP + FN)).
- **F1-score**: harmonic mean of precision and recall, balancing both error types.
- **Support**: number of samples belonging to each class; useful for weighting metrics.

### Coefficients
- Displays per-feature coefficient estimates together with their standard errors, Wald z-scores and two-sided p-values.
- Significance codes follow the common R-style legend (`***`, `**`, `*`, `.`) to highlight strong effects quickly.
- Values are derived from the Fisher information matrix of the fitted logistic regression model (no regularisation). Interpret large |z| as stronger evidence that the coefficient differs from zero.

## Additional tips

- To log accuracy to disk, call `trainer.train_model(log_result=True)`. Each run appends a row to `results/accuracy_results.csv` (including Git commit and timestamp).
- For any other labelled dataset (validation/test), use `trainer.evaluate_dataset(df, target_column="player_won")`.
- If you need additional metrics (AUC, log loss, etc.), extend `ModelPerformanceReport` in `analysis/model_performance.py` or add your own helpers.
