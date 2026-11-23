# CIE-2 Activity 2: Binary Classification with Logistic Regression (Loan Approval)

**Student roll number:** AD3125
**Course:** Regression and Classification Algorithms (RCA), T.Y. B.Tech AI & Data Science, Zeal College of Engineering and Research
**Submission GitHub ID:** [github.com/complexiota](https://github.com/complexiota)

## Dataset
- **Source:** taweilo, *Loan Approval Classification Dataset*, Kaggle (synthetic data for binary classification). https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- **File:** `loan_data.csv`, 45,000 rows, 14 columns. The source describes the data as synthetic, so results describe this dataset and may not transfer to real lending.
- **Target (y):** `loan_status` (1 = approved, 0 = rejected)
- **Excluded input:** `loan_int_rate`, because it is set after the lending decision and would leak the outcome.

## What is in this repo
- `AD3125_Activity2_Logistic_Regression_Classification.ipynb`: the executed notebook. It covers class balance, feature preparation, a stratified 80:20 split, the sigmoid function, logistic regression, the confusion matrix, accuracy, precision, recall, specificity, F1, ROC-AUC, a majority-class baseline, threshold comparison (0.3, 0.5, 0.7), a leakage sensitivity check, an overfitting check, a decision-behaviour plot, a new-applicant prediction, and a decision log.

## Key results (test set, 8,999 applicants, threshold 0.5)

| Metric | Majority-class baseline | Logistic regression |
|---|---|---|
| Accuracy | 0.778 | 0.865 |
| Precision (approved) | 0.000 | 0.717 |
| Recall (approved) | 0.000 | 0.652 |
| F1-score | 0.000 | 0.683 |
| Specificity | 1.000 | 0.926 |
| ROC-AUC | 0.5 | 0.930 |

Confusion matrix at threshold 0.5: TP = 1,303, FP = 515, FN = 697, TN = 6,484.

**Threshold trade-off:** 0.3 gives higher recall (0.84) but lower precision (0.59). 0.7 gives higher precision (0.86) but lower recall (0.46). I recommend **0.5** as the default.

## Important caveats
- Most of the performance comes from `previous_loan_defaults_on_file`, which separates the classes almost perfectly in this data. Without it, ROC-AUC falls to 0.79 and recall to 0.40.
- Two coefficients (loan-to-income ratio and credit score) have signs opposite to what a lender would expect. They are flagged as likely data artefacts.
- The data is synthetic. Results should not be used for real credit decisions.

## How to run
1. Place `loan_data.csv` in the same folder as the notebook.
2. Install: `pandas numpy matplotlib seaborn scikit-learn`.
3. Open the notebook and run all cells from top to bottom. A fixed random seed (42) makes results reproducible.
