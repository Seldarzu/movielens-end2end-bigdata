"""
Testler: spark/config.py
- Tüm path sabitlerinin tanımlı ve string olduğunu doğrular
- require_path() fonksiyonunun yok olan dizinde sys.exit(1) çağırdığını doğrular
- check_mlflow() fonksiyonunun erişilemeyen URL'de exception fırlatmadığını doğrular
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spark"))
import config


# ── Path sabitleri ────────────────────────────────────────────────────────────

PATH_CONSTANTS = [
    "DELTA_RATINGS_PATH",
    "DELTA_CLEANED_PATH",
    "DELTA_FEATURES_BASE",
    "DELTA_MOVIE_STATS_PATH",
    "DELTA_USER_STATS_PATH",
    "DELTA_ENRICHED_PATH",
    "DELTA_FULL_FEATURES_PATH",
    "CHECKPOINT_PATH",
    "PLOT_DIR",
    "EDA_OUTPUT_DIR",
    "MOVIES_CSV_PATH",
    "RATINGS_CSV_PATH",
]

@pytest.mark.parametrize("const", PATH_CONSTANTS)
def test_path_constant_is_string(const):
    value = getattr(config, const)
    assert isinstance(value, str), f"{const} string olmali"
    assert value.startswith("/"), f"{const} mutlak yol olmali (/ ile baslamali)"


def test_mlflow_uri_is_string():
    assert isinstance(config.MLFLOW_URI, str)
    assert "://" in config.MLFLOW_URI


def test_experiment_names_are_strings():
    for attr in ["EXPERIMENT_PREPROCESSING", "EXPERIMENT_ALS",
                 "EXPERIMENT_CLS_REG", "EXPERIMENT_EXTENDED", "EXPERIMENT_LOG_MODEL"]:
        value = getattr(config, attr)
        assert isinstance(value, str) and len(value) > 0, f"{attr} bos olmamali"


# ── require_path() ─────────────────────────────────────────────────────────────

def test_require_path_exits_when_missing(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(SystemExit) as exc_info:
        config.require_path(missing)
    assert exc_info.value.code == 1


def test_require_path_passes_when_exists(tmp_path):
    existing = str(tmp_path)
    config.require_path(existing)  # SystemExit fırlatmamalı


# ── check_mlflow() ────────────────────────────────────────────────────────────

def test_check_mlflow_does_not_raise_on_unreachable():
    # Erişilemeyen URL için sadece warning loglamalı, exception fırlatmamalı
    config.check_mlflow("http://localhost:19999")  # Kapalı port
