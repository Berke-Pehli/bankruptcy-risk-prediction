# Bankruptcy Risk Prediction

A reproducible machine-learning project that evaluates whether tree-based models
improve one-year-ahead corporate bankruptcy prediction relative to an interpretable
Logistic Regression benchmark.

The project uses annual accounting data for US public companies from 1999 to 2018.
Its empirical design emphasizes temporal validation, rare-event evaluation, financial
interpretation, and honest discussion of model limitations.

## Research question

> Can tree-based machine-learning models improve out-of-time corporate bankruptcy
> prediction compared with Logistic Regression?

## Planned methodology

- Construct a one-year-ahead bankruptcy outcome from the firm-year panel.
- Engineer financially meaningful profitability, leverage, liquidity, and efficiency ratios.
- Use expanding-window validation and a final untouched temporal test period.
- Compare Logistic Regression, regularized Logistic Regression, a pruned decision tree,
  Random Forest, and Gradient Boosting.
- Evaluate rare-event predictions using Precision–Recall AUC, ROC-AUC, recall, precision,
  F-scores, calibration, and confusion matrices.
- Use Pixi, Pytask, and Pytest to make the complete workflow reproducible and testable.

## Project status

The repository is being developed incrementally. Each milestone is implemented, tested,
and documented before the next modelling component is added.

