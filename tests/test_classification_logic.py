"""
Testler: Sınıflandırma ve regresyon iş mantığı (Spark bağımsız)
- Binary etiket oluşturma mantığı (rating >= 4.0)
- Overfitting eşiği dinamik hesaplama
- Confusion matrix hücre tanımları (TN/FP/FN/TP)
- Metrik hesaplama (precision, recall, F1)
"""
import pytest


# ── Binary etiket mantığı ─────────────────────────────────────────────────────

def make_label(rating, threshold=4.0):
    return 1.0 if rating >= threshold else 0.0


@pytest.mark.parametrize("rating, expected", [
    (5.0, 1.0),
    (4.0, 1.0),
    (3.9, 0.0),
    (3.5, 0.0),
    (1.0, 0.0),
    (0.5, 0.0),
])
def test_binary_label(rating, expected):
    assert make_label(rating) == expected


def test_label_distribution_balanced():
    ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    labels = [make_label(r) for r in ratings]
    pos = sum(labels)
    neg = len(labels) - pos
    # 4.0 eşiğiyle %50/50'ye yakın olmalı (bu test veri setinde 4/10 = 0.4)
    assert 0.3 <= pos / len(labels) <= 0.7, "Etiket dağılımı çok dengesiz"


# ── Dinamik overfitting eşiği ─────────────────────────────────────────────────

def get_overfit_threshold(n_train):
    if n_train < 5_000:
        return 0.03
    elif n_train < 50_000:
        return 0.05
    else:
        return 0.10


@pytest.mark.parametrize("n_train, expected", [
    (1_000,   0.03),
    (4_999,   0.03),
    (5_000,   0.05),
    (25_000,  0.05),
    (49_999,  0.05),
    (50_000,  0.10),
    (200_000, 0.10),
])
def test_dynamic_overfit_threshold(n_train, expected):
    assert get_overfit_threshold(n_train) == expected


# ── Confusion matrix hücreleri ────────────────────────────────────────────────

def compute_cm(actuals, predictions):
    tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
    fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 0)
    tp = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 1)
    return tn, fp, fn, tp


def test_confusion_matrix_perfect():
    actuals =      [0, 0, 1, 1]
    predictions =  [0, 0, 1, 1]
    tn, fp, fn, tp = compute_cm(actuals, predictions)
    assert tn == 2 and fp == 0 and fn == 0 and tp == 2


def test_confusion_matrix_all_wrong():
    actuals =      [0, 0, 1, 1]
    predictions =  [1, 1, 0, 0]
    tn, fp, fn, tp = compute_cm(actuals, predictions)
    assert tn == 0 and fp == 2 and fn == 2 and tp == 0


def test_confusion_matrix_mixed():
    actuals =     [0, 1, 0, 1, 1]
    predictions = [0, 1, 1, 0, 1]
    tn, fp, fn, tp = compute_cm(actuals, predictions)
    assert tn == 1 and fp == 1 and fn == 1 and tp == 2


# ── Precision / Recall / F1 hesaplama ────────────────────────────────────────

def compute_metrics(tn, fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def test_metrics_perfect():
    prec, rec, f1 = compute_metrics(tn=2, fp=0, fn=0, tp=2)
    assert prec == 1.0 and rec == 1.0 and f1 == 1.0


def test_metrics_zero_tp():
    prec, rec, f1 = compute_metrics(tn=2, fp=2, fn=2, tp=0)
    assert prec == 0.0 and rec == 0.0 and f1 == 0.0


def test_metrics_mixed():
    prec, rec, f1 = compute_metrics(tn=1, fp=1, fn=1, tp=2)
    assert prec == pytest.approx(0.6667, abs=0.001)
    assert rec  == pytest.approx(0.6667, abs=0.001)
