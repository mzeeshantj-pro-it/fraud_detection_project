# Credit Card Fraud Detection — Complete Project
**Muhammad Zeeshan | B01799050 | UWS COMP11128 MSc Masters Pathway Project**
**Detection of Fraudulent Online Banking Transactions Using Supervised Machine Learning**
**Dataset: Dal Pozzolo et al. (2015) — ULB Credit Card Fraud**

---

## Quick Start (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests (verify everything works)
python tests.py

# 3. Train all models (generates all 8 figures)
python train.py
```

---

## Project Structure

```
fraud_detection_project/
│
├── data/
│   └── creditcard.csv          ← PLACE YOUR DATASET HERE
│
├── models/                     ← Created by train.py
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
├── reports/                    ← Created by train.py
│   ├── fig1_class_distribution.png
│   ├── fig2_amount_by_class.png
│   ├── fig3_time_by_class.png
│   ├── fig4_confusion_matrices.png
│   ├── fig5_roc_curves.png
│   ├── fig6_metrics_comparison.png
│   ├── fig7_learning_curves.png
│   ├── fig8_feature_importance.png
│   └── results_summary.csv
│
├── train.py                    ← Main training pipeline
├── predict.py                  ← Single + batch prediction
├── tests.py                    ← Full test suite (19 tests)
├── requirements.txt
└── README.md
```

---

## Dataset Setup

Download the ULB Credit Card Fraud Detection Dataset:
```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Save the downloaded file as:
```
data/creditcard.csv
```

**About the dataset (Dal Pozzolo et al., 2015):**
- 284,807 real European credit card transactions (September 2013)
- 492 confirmed fraud cases (0.1727%)
- Features V1–V28: PCA-transformed (anonymised for privacy)
- Features: Amount, Time (original)
- Target: Class — 1=fraud, 0=legitimate

---

## Step 1 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

Python 3.8 or above required.

---

## Step 2 — Run Tests

```bash
python tests.py
```

Expected output:
```
============================================================
  FRAUD DETECTION — TEST SUITE
  Muhammad Zeeshan | B01799050 | UWS COMP11128
============================================================
...
============================================================
  ALL 19 TESTS PASSED ✓
============================================================
```

Tests cover:
- Dataset shape and fraud count (6 tests)
- Feature engineering correctness (6 tests)
- Stratified split preservation (4 tests)
- StandardScaler behaviour (3 tests)
- Evaluation metric correctness (5 tests)
- End-to-end integration smoke test (2 tests)

---

## Step 3 — Train All Models

```bash
python train.py
```

With a custom dataset path:
```bash
python train.py --data /path/to/creditcard.csv
```

What happens:
1. Loads 284,807 transactions
2. Generates EDA figures (Figures 1–3)
3. Engineers 3 new features
4. Splits 80/20 stratified
5. Creates balanced training sets (10:1 for LR/RF, 3:1 for SVM)
6. Trains Logistic Regression, Random Forest, SVM
7. Evaluates on the full imbalanced test set
8. Generates result figures (Figures 4–8)
9. Saves results_summary.csv

**Expected training time:** ~3–8 minutes depending on hardware.

---

## Step 4 — Make Predictions

### Interactive mode (type one transaction at a time)
```bash
python predict.py
```

### Quick single prediction
```bash
python predict.py --amount 1500
python predict.py --amount 50 --time 3600
```

### Batch prediction on a CSV file
```bash
python predict.py --file data/new_transactions.csv
```

The batch input CSV needs at minimum an `Amount` column.
V1–V28 and Time default to 0 and 50000 if not provided.

Output: `new_transactions_predictions.csv` with columns:
- fraud_probability
- is_fraud_predicted
- decision (APPROVE / BLOCK)
- risk_level (LOW / MEDIUM / HIGH)

### Use a different model
```bash
python predict.py --model models/logistic_regression.pkl --amount 200
python predict.py --model models/svm.pkl --amount 200
```

---

## Expected Results

After running `train.py` on the full ULB dataset:

| Model               | Accuracy | Precision | Recall | F1     | AUC    | CV F1  |
|---------------------|----------|-----------|--------|--------|--------|--------|
| Logistic Regression | 97.45%   | 5.81%     | 90.82% | 10.93% | 0.9719 | 84.66% |
| **Random Forest**   | **99.88%** | **59.31%** | **87.76%** | **70.78%** | **0.9782** | **90.09%** |
| SVM                 | 98.17%   | 7.80%     | 88.78% | 14.33% | 0.9805 | 91.33% |

**Key finding:** Random Forest generates only 59 false alarms versus
1,442 for Logistic Regression — 24x fewer false alarms while catching
nearly the same number of fraud cases.

---

## Reference

Dal Pozzolo, A., Caelen, O., Le Borgne, Y.A., Waterschoot, S.
and Bontempi, G. (2015) 'Calibrating probability with undersampling
for unbalanced classification', IEEE Symposium Series on
Computational Intelligence. Available at:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
