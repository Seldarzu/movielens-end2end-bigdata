"""
Testler: Ön işleme iş mantığı (Spark bağımsız, saf Python/Pandas)
- Geçersiz rating aralığı tespiti
- Hyperactive kullanıcı eşiği mantığı
- Duplikat tespiti (aynı userId+movieId+timestamp)
- Null değer sayımı
"""
import pandas as pd
import pytest


VALID_RATING_MIN = 0.5
VALID_RATING_MAX = 5.0
HYPERACTIVE_THRESHOLD = 10_000


# ── Geçersiz rating aralığı ───────────────────────────────────────────────────

def is_valid_rating(r):
    return VALID_RATING_MIN <= r <= VALID_RATING_MAX


@pytest.mark.parametrize("rating, valid", [
    (0.5,  True),
    (1.0,  True),
    (3.5,  True),
    (5.0,  True),
    (0.0,  False),
    (0.4,  False),
    (5.1,  False),
    (6.0,  False),
    (-1.0, False),
])
def test_rating_range_validation(rating, valid):
    assert is_valid_rating(rating) == valid


# ── Hyperactive kullanıcı eşiği ───────────────────────────────────────────────

def test_hyperactive_threshold():
    df = pd.DataFrame({"userId": [1]*15_000 + [2]*500, "rating": [4.0]*15_500})
    counts = df.groupby("userId").size()
    hyperactive = counts[counts > HYPERACTIVE_THRESHOLD].index.tolist()
    assert 1 in hyperactive
    assert 2 not in hyperactive


# ── Duplikat tespiti ──────────────────────────────────────────────────────────

def test_duplicate_detection():
    df = pd.DataFrame({
        "userId":    [1, 1, 2, 3],
        "movieId":   [10, 10, 20, 30],
        "timestamp": [1000, 1000, 2000, 3000],
        "rating":    [4.0, 4.0, 3.5, 5.0],
    })
    dupes = df.duplicated(subset=["userId", "movieId", "timestamp"])
    assert dupes.sum() == 1
    clean = df.drop_duplicates(subset=["userId", "movieId", "timestamp"])
    assert len(clean) == 3


# ── Null değer sayımı ─────────────────────────────────────────────────────────

def test_null_count():
    df = pd.DataFrame({
        "userId":  [1, None, 3],
        "movieId": [10, 20, None],
        "rating":  [4.0, 3.5, 5.0],
    })
    nulls = df.isnull().sum()
    assert nulls["userId"] == 1
    assert nulls["movieId"] == 1
    assert nulls["rating"] == 0


def test_no_nulls_in_clean_data():
    df = pd.DataFrame({
        "userId":  [1, 2, 3],
        "movieId": [10, 20, 30],
        "rating":  [4.0, 3.5, 5.0],
    })
    assert df.isnull().sum().sum() == 0
