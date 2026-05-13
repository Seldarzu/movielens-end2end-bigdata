"""
Testler: Pipeline bütünlük kontrolleri (Docker gerektirmez)
- run_scripts.ps1 ve run_pipeline.ps1 dosyalarının varlığını doğrular
- Tüm beklenen spark script dosyalarının var olduğunu kontrol eder
- docker-compose.yml'in beklenen servisleri tanımladığını kontrol eder
- config.py path'lerinin tutarlı /app/delta prefix'ine sahip olduğunu doğrular
"""
import os
import sys
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SPARK_DIR = os.path.join(ROOT, "spark")

sys.path.insert(0, SPARK_DIR)
import config


# ── Zorunlu dosyalar ──────────────────────────────────────────────────────────

REQUIRED_SPARK_SCRIPTS = [
    "consumer.py", "eda.py", "preprocessing.py", "etl.py",
    "feature_engineering.py", "model.py", "classification.py",
    "scale_analysis.py",
    "extended_metrics.py", "advanced_visualizations.py", "log_model.py",
    "config.py",
]

@pytest.mark.parametrize("script", REQUIRED_SPARK_SCRIPTS)
def test_spark_script_exists(script):
    path = os.path.join(SPARK_DIR, script)
    assert os.path.isfile(path), f"Eksik dosya: spark/{script}"


REQUIRED_ROOT_FILES = [
    "docker-compose.yml",
    "run_pipeline.ps1",
    "run_scripts.ps1",
    "requirements.txt",
    "LICENSE",
    "README.md",
    ".gitignore",
]

@pytest.mark.parametrize("filename", REQUIRED_ROOT_FILES)
def test_root_file_exists(filename):
    path = os.path.join(ROOT, filename)
    assert os.path.isfile(path), f"Eksik dosya: {filename}"


REQUIRED_DOCKERFILES = [
    "Dockerfile.producer",
    "Dockerfile.spark",
    "Dockerfile.mlflow",
    "Dockerfile.streamlit",
]

@pytest.mark.parametrize("df", REQUIRED_DOCKERFILES)
def test_dockerfile_exists(df):
    path = os.path.join(ROOT, df)
    assert os.path.isfile(path), f"Eksik: {df}"


# ── docker-compose.yml servis kontrolü ───────────────────────────────────────

def test_docker_compose_has_required_services():
    compose_path = os.path.join(ROOT, "docker-compose.yml")
    content = open(compose_path).read()
    for service in ["zookeeper", "kafka", "spark-master", "spark-worker", "mlflow", "streamlit"]:
        assert service in content, f"docker-compose.yml'de '{service}' servisi eksik"


# ── config.py path tutarlılığı ────────────────────────────────────────────────

DELTA_PATHS = [
    config.DELTA_RATINGS_PATH,
    config.DELTA_CLEANED_PATH,
    config.DELTA_FEATURES_BASE,
    config.DELTA_ENRICHED_PATH,
    config.DELTA_FULL_FEATURES_PATH,
    config.CHECKPOINT_PATH,
    config.PLOT_DIR,
]

@pytest.mark.parametrize("path", DELTA_PATHS)
def test_delta_paths_have_app_prefix(path):
    assert path.startswith("/app/"), f"'{path}' /app/ ile baslamali (konteyner icinde)"


def test_mlflow_path_starts_with_mlflow():
    # MLflow path /mlflow altında olmalı
    assert config.MLFLOW_URI.startswith("http://"), "MLFLOW_URI http:// ile basmali"


def test_no_duplicate_path_values():
    """Aynı path iki farklı sabite atanmamış olmalı (ETL ve Feature çıktıları farklı)."""
    paths = [
        config.DELTA_RATINGS_PATH,
        config.DELTA_CLEANED_PATH,
        config.DELTA_ENRICHED_PATH,
        config.DELTA_FULL_FEATURES_PATH,
    ]
    assert len(paths) == len(set(paths)), "Çakışan path sabitleri var"
