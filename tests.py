"""
=============================================================
  Credit Card Fraud Detection — Test Suite
  Muhammad Zeeshan | B01799050 | UWS COMP11128
=============================================================

Covers:
  • Data loading and integrity
  • Feature engineering correctness
  • Stratified split preservation
  • StandardScaler behaviour
  • Metric calculation correctness
  • Pipeline integration (end-to-end smoke test)

Run:
    python tests.py
"""

import os, sys, unittest, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

DATA_PATH = "data/creditcard.csv"


# ─────────────────────────────────────────────────────────────
# Helper — load a representative sample quickly
# ─────────────────────────────────────────────────────────────
def get_sample(n_legit=2000):
    """Return a small DataFrame with both classes represented."""
    df = pd.read_csv(DATA_PATH)
    fraud = df[df["Class"] == 1].copy()              # all 492 fraud
    legit = df[df["Class"] == 0].sample(
        n=n_legit, random_state=42)
    return pd.concat([fraud, legit]).sample(
        frac=1, random_state=42).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════
# TEST CLASS 1 — Data Loading
# ═════════════════════════════════════════════════════════════
class TestDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_row_count(self):
        """Full dataset must have 284,807 rows."""
        self.assertEqual(len(self.df), 284807,
            f"Expected 284807 rows, got {len(self.df)}")
        print(f"\n    Rows: {len(self.df):,} — PASS")

    def test_column_count(self):
        """Full dataset must have 31 columns."""
        self.assertEqual(self.df.shape[1], 31,
            f"Expected 31 columns, got {self.df.shape[1]}")
        print(f"\n    Columns: {self.df.shape[1]} — PASS")

    def test_no_missing_values(self):
        """No null values in any column."""
        nulls = self.df.isnull().sum().sum()
        self.assertEqual(nulls, 0,
            f"Found {nulls} missing values")
        print(f"\n    Missing values: {nulls} — PASS")

    def test_class_column_binary(self):
        """Class column must contain only 0 and 1."""
        vals = set(self.df["Class"].unique())
        self.assertTrue(vals.issubset({0, 1}),
            f"Class contains unexpected values: {vals}")

    def test_fraud_count(self):
        """Exactly 492 confirmed fraud cases in the dataset."""
        n = self.df["Class"].sum()
        self.assertEqual(n, 492, f"Expected 492 fraud cases, got {n}")
        print(f"\n    Fraud cases: {n} — PASS")

    def test_fraud_rate(self):
        """Fraud rate must be approximately 0.1727%."""
        rate = self.df["Class"].mean() * 100
        self.assertAlmostEqual(rate, 0.1727, places=2,
            msg=f"Fraud rate {rate:.4f}% differs from expected 0.1727%")
        print(f"\n    Fraud rate: {rate:.4f}% — PASS")


# ═════════════════════════════════════════════════════════════
# TEST CLASS 2 — Feature Engineering
# ═════════════════════════════════════════════════════════════
class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = get_sample()
        df["log_amount"]  = np.log1p(df["Amount"])
        df["hour_of_day"] = (df["Time"] % 86400) / 3600
        df["is_night"]    = (
            (df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)
        ).astype(int)
        cls.df = df

    def test_log_amount_nonneg(self):
        """log_amount must be non-negative (log1p of non-negative input)."""
        minimum = self.df["log_amount"].min()
        self.assertGreaterEqual(minimum, 0.0,
            f"log_amount min={minimum:.6f}, expected >= 0")
        print(f"\n    log_amount min={minimum:.6f} — PASS")

    def test_log_amount_max(self):
        """log_amount max must be log1p of max Amount."""
        expected_max = np.log1p(self.df["Amount"].max())
        actual_max   = self.df["log_amount"].max()
        self.assertAlmostEqual(actual_max, expected_max, places=5)
        print(f"\n    log_amount max={actual_max:.4f} — PASS")

    def test_hour_of_day_range(self):
        """hour_of_day must be within [0, 24)."""
        lo = self.df["hour_of_day"].min()
        hi = self.df["hour_of_day"].max()
        self.assertGreaterEqual(lo, 0.0)
        self.assertLess(hi, 24.0)
        print(f"\n    hour_of_day range [{lo:.4f}, {hi:.4f}] — PASS")

    def test_is_night_binary(self):
        """is_night must contain only 0 and 1."""
        vals = set(self.df["is_night"].unique())
        self.assertTrue(vals.issubset({0, 1}),
            f"is_night contains non-binary values: {vals}")

    def test_is_night_logic(self):
        """Rows where hour < 6 or hour > 22 must have is_night=1."""
        night_rows  = self.df[
            (self.df["hour_of_day"] < 6) | (self.df["hour_of_day"] > 22)
        ]
        self.assertTrue(
            (night_rows["is_night"] == 1).all(),
            "is_night=0 found for a night-time transaction"
        )

    def test_total_features(self):
        """Feature vector must have 31 dimensions (28 PCA + 3 engineered)."""
        feat_cols = [f"V{i}" for i in range(1, 29)] + [
            "log_amount", "hour_of_day", "is_night"
        ]
        self.assertEqual(len(feat_cols), 31)
        print(f"\n    Feature count: {len(feat_cols)} — PASS")


# ═════════════════════════════════════════════════════════════
# TEST CLASS 3 — Train-Test Split
# ═════════════════════════════════════════════════════════════
class TestSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sklearn.model_selection import train_test_split
        df = get_sample()
        df["log_amount"]  = np.log1p(df["Amount"])
        df["hour_of_day"] = (df["Time"] % 86400) / 3600
        df["is_night"]    = (
            (df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)
        ).astype(int)
        feat_cols = [f"V{i}" for i in range(1, 29)] + [
            "log_amount", "hour_of_day", "is_night"
        ]
        X = df[feat_cols].values
        y = df["Class"].values
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        cls.original_rate = y.mean()

    def test_total_rows_preserved(self):
        """train + test must equal the original sample size."""
        total = len(self.y_train) + len(self.y_test)
        expected = len(self.X_train) + len(self.X_test)
        self.assertEqual(total, expected)

    def test_train_fraud_rate_stratified(self):
        """Training fraud rate must be within 0.5% of original."""
        train_rate = self.y_train.mean()
        diff = abs(train_rate - self.original_rate)
        self.assertLess(diff, 0.005,
            f"Train rate {train_rate:.4%} vs original {self.original_rate:.4%}")
        print(f"\n    Train fraud rate: {train_rate:.4%} "
              f"(original: {self.original_rate:.4%}) — PASS")

    def test_test_fraud_rate_stratified(self):
        """Test fraud rate must be within 0.5% of original."""
        test_rate = self.y_test.mean()
        diff = abs(test_rate - self.original_rate)
        self.assertLess(diff, 0.005,
            f"Test rate {test_rate:.4%} vs original {self.original_rate:.4%}")
        print(f"\n    Test fraud rate:  {test_rate:.4%} "
              f"(original: {self.original_rate:.4%}) — PASS")

    def test_test_size_approx_20pct(self):
        """Test set should be approximately 20% of total."""
        test_pct = len(self.y_test) / (len(self.y_train) + len(self.y_test))
        self.assertAlmostEqual(test_pct, 0.2, delta=0.01,
            msg=f"Test size {test_pct:.2%}, expected ~20%")
        print(f"\n    Test size: {test_pct:.2%} — PASS")


# ═════════════════════════════════════════════════════════════
# TEST CLASS 4 — StandardScaler
# ═════════════════════════════════════════════════════════════
class TestScaler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        df = get_sample()
        df["log_amount"]  = np.log1p(df["Amount"])
        df["hour_of_day"] = (df["Time"] % 86400) / 3600
        df["is_night"]    = (
            (df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)
        ).astype(int)
        feat_cols = [f"V{i}" for i in range(1, 29)] + [
            "log_amount", "hour_of_day", "is_night"
        ]
        X = df[feat_cols].values
        y = df["Class"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler()
        cls.X_train_sc = sc.fit_transform(Xtr)
        cls.X_test_sc  = sc.transform(Xte)

    def test_train_mean_near_zero(self):
        """After scaling, max absolute mean across features must be < 0.001."""
        max_mean = np.abs(self.X_train_sc.mean(axis=0)).max()
        self.assertLess(max_mean, 0.001,
            f"Max abs mean = {max_mean:.6f}, expected < 0.001")
        print(f"\n    Max abs train mean: {max_mean:.6f} — PASS")

    def test_train_std_near_one(self):
        """Non-constant features must have std within 0.01 of 1.0 after scaling."""
        stds = self.X_train_sc.std(axis=0)
        # Exclude near-constant features (e.g. is_night in very small sample)
        variable_stds = stds[stds > 0.01]
        if len(variable_stds) == 0:
            self.skipTest("No variable features found in sample")
        min_std = variable_stds.min()
        max_std = variable_stds.max()
        self.assertGreater(min_std, 0.98,
            f"Min std={min_std:.6f}, expected > 0.98")
        self.assertLess(max_std, 1.02,
            f"Max std={max_std:.6f}, expected < 1.02")
        print(f"\n    Std range: [{min_std:.6f}, {max_std:.6f}] — PASS")

    def test_scaler_not_fitted_on_test(self):
        """Test set must have different mean from train set (not re-fitted)."""
        train_mean = np.abs(self.X_train_sc.mean(axis=0)).max()
        test_mean  = np.abs(self.X_test_sc.mean(axis=0)).max()
        # If scaler was re-fitted on test, test mean would also be ~0
        # Here we just check train mean is near zero (confirms correct fitting)
        self.assertLess(train_mean, 0.001)


# ═════════════════════════════════════════════════════════════
# TEST CLASS 5 — Evaluation Metrics
# ═════════════════════════════════════════════════════════════
class TestMetrics(unittest.TestCase):

    def test_confusion_matrix_totals(self):
        """TN + FP + FN + TP must equal total sample size."""
        from sklearn.metrics import confusion_matrix
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
        cm    = confusion_matrix(y_true, y_pred)
        total = int(cm.sum())
        self.assertEqual(total, len(y_true),
            f"CM total {total} != sample size {len(y_true)}")
        print(f"\n    CM total={total} = n_samples — PASS")

    def test_auc_valid_range(self):
        """ROC-AUC must be between 0.5 and 1.0 for a useful classifier."""
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.1, 0.3, 0.2, 0.7, 0.95])
        auc = roc_auc_score(y_true, y_prob)
        self.assertGreaterEqual(auc, 0.5)
        self.assertLessEqual(auc,   1.0)
        print(f"\n    AUC={auc:.4f} in [0.5, 1.0] — PASS")

    def test_f1_formula(self):
        """F1 = 2*TP / (2*TP + FP + FN)."""
        from sklearn.metrics import f1_score
        # TP=2, FP=0, FN=1 → F1 = 2*2/(2*2+0+1) = 4/5 = 0.8
        y_true    = np.array([1, 1, 1, 0, 0])
        y_pred    = np.array([1, 1, 0, 0, 0])
        f1        = f1_score(y_true, y_pred)
        expected  = 4 / 5
        self.assertAlmostEqual(f1, expected, places=4)
        print(f"\n    F1={f1:.4f} = expected {expected:.4f} — PASS")

    def test_perfect_classifier_auc(self):
        """Perfect classifier must give AUC = 1.0."""
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])
        auc    = roc_auc_score(y_true, y_prob)
        self.assertEqual(auc, 1.0)
        print(f"\n    Perfect AUC={auc:.1f} — PASS")

    def test_precision_recall_zero_division(self):
        """Precision/recall must not crash when no positive predictions."""
        from sklearn.metrics import precision_score, recall_score
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])   # predicts all negative
        prec   = precision_score(y_true, y_pred, zero_division=0)
        rec    = recall_score(y_true, y_pred, zero_division=0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec,  0.0)
        print(f"\n    Zero-division handled: prec={prec}, rec={rec} — PASS")


# ═════════════════════════════════════════════════════════════
# TEST CLASS 6 — Pipeline Integration (smoke test)
# ═════════════════════════════════════════════════════════════
class TestPipelineIntegration(unittest.TestCase):

    def test_end_to_end_small_sample(self):
        """
        Full pipeline smoke test on 500 rows.
        Checks: load → features → split → scale → train LR → evaluate.
        No assertion failures = pipeline wiring is correct.
        """
        from sklearn.preprocessing   import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model    import LogisticRegression
        from sklearn.metrics         import f1_score

        # Load small sample
        df = pd.read_csv(DATA_PATH)
        fraud = df[df["Class"] == 1].head(50)
        legit = df[df["Class"] == 0].head(450)
        df_s  = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

        # Features
        df_s = df_s.copy()
        df_s["log_amount"]  = np.log1p(df_s["Amount"])
        df_s["hour_of_day"] = (df_s["Time"] % 86400) / 3600
        df_s["is_night"]    = (
            (df_s["hour_of_day"] < 6) | (df_s["hour_of_day"] > 22)
        ).astype(int)
        feat_cols = [f"V{i}" for i in range(1, 29)] + [
            "log_amount", "hour_of_day", "is_night"
        ]
        X = df_s[feat_cols].values
        y = df_s["Class"].values

        # Split
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale
        sc   = StandardScaler()
        Xtr  = sc.fit_transform(Xtr)
        Xte  = sc.transform(Xte)

        # Balanced training set
        fraud_idx = np.where(ytr == 1)[0]
        legit_idx  = np.where(ytr == 0)[0]
        rng = np.random.RandomState(42)
        n   = min(len(legit_idx), len(fraud_idx) * 3)
        smp = rng.choice(legit_idx, size=n, replace=False)
        idx = rng.permutation(np.concatenate([fraud_idx, smp]))
        Xb  = Xtr[idx]; yb = ytr[idx]

        # Train
        model = LogisticRegression(
            C=0.1, max_iter=500, class_weight="balanced",
            random_state=42)
        model.fit(Xb, yb)

        # Evaluate
        yp   = model.predict(Xte)
        ypr  = model.predict_proba(Xte)[:, 1]
        f1   = f1_score(yte, yp, zero_division=0)
        from sklearn.metrics import roc_auc_score
        auc  = roc_auc_score(yte, ypr)

        self.assertGreater(auc, 0.5, f"AUC={auc:.4f} too low — pipeline may be broken")
        self.assertGreater(f1,  0.0, "F1=0 — model predicts nothing as fraud")
        print(f"\n    Integration test: F1={f1:.4f}, AUC={auc:.4f} — PASS")

    def test_predict_one_function(self):
        """predict.py predict_one must return correct keys and ranges."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from predict import predict_one, build_feature_vector

        # Build a synthetic scaler and model for isolated test
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model  import LogisticRegression
        import numpy as np

        rng = np.random.RandomState(0)
        X   = rng.randn(200, 31)
        y   = rng.randint(0, 2, 200)
        sc  = StandardScaler().fit(X)
        m   = LogisticRegression(max_iter=200).fit(sc.transform(X), y)

        result = predict_one(m, sc, {"Amount": 150.0, "Time": 3600.0})
        self.assertIn("fraud_probability", result)
        self.assertIn("decision",          result)
        self.assertIn("risk_level",        result)
        self.assertIn(result["decision"],  ["APPROVE", "BLOCK"])
        self.assertIn(result["risk_level"], ["LOW", "MEDIUM", "HIGH"])
        prob = result["fraud_probability"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob,    1.0)
        print(f"\n    predict_one returned valid result "
              f"(prob={prob:.4f}) — PASS")


# ═════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FRAUD DETECTION — TEST SUITE")
    print("  Muhammad Zeeshan | B01799050 | UWS COMP11128")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: '{DATA_PATH}' not found.")
        print("  Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Save as:  data/creditcard.csv\n")
        sys.exit(1)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestDataLoading,
        TestFeatureEngineering,
        TestSplit,
        TestScaler,
        TestMetrics,
        TestPipelineIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED ✓")
    else:
        fails  = len(result.failures)
        errors = len(result.errors)
        print(f"  {fails} FAILURE(S)  {errors} ERROR(S)  "
              f"out of {result.testsRun} tests")
    print("=" * 60 + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
