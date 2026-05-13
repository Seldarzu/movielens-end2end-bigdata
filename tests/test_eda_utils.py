"""
Testler: spark/eda.py _fmt() yardımcı fonksiyonu mantığı
- Doğru birim etiketleri ürettiğini doğrular (K, M)
- eda.py pyspark gerektirdiğinden fonksiyon burada yeniden tanımlanmıştır;
  mantık spark/eda.py:_fmt ile birebir aynıdır.
"""
import pytest


def _fmt(v: float) -> str:
    """spark/eda.py:_fmt ile aynı mantık — adaptif birim formatı."""
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return str(int(v))


# ── _fmt() birim formatı ──────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    (0,           "0"),
    (999,         "999"),
    (1_000,       "1.0K"),
    (1_500,       "1.5K"),
    (999_999,     "1000.0K"),
    (1_000_000,   "1.00M"),
    (2_500_000,   "2.50M"),
    (25_000_000,  "25.00M"),
])
def test_fmt_output(value, expected):
    assert _fmt(value) == expected


# ── Adaptif eksen seçimi mantığı ─────────────────────────────────────────────

def test_fmt_small_values_no_suffix():
    assert _fmt(500) == "500"


def test_fmt_thousands_use_K():
    result = _fmt(50_000)
    assert "K" in result
    assert "M" not in result


def test_fmt_millions_use_M():
    result = _fmt(5_000_000)
    assert "M" in result
