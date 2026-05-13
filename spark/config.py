# Merkezi yapılandırma — tüm pipeline scriptleri bu dosyadan import eder.
# Yol veya servis adresi değiştiğinde yalnızca bu dosyayı güncelle.

import sys
import os
import logging

# ─── Delta Lake Yolları ───────────────────────────────────────────────────────
DELTA_RATINGS_PATH      = "/app/delta/ratings"
DELTA_CLEANED_PATH      = "/app/delta/ratings_cleaned"

# ETL çıktıları
DELTA_FEATURES_BASE     = "/app/delta/ratings_features"
DELTA_MOVIE_STATS_PATH  = "/app/delta/ratings_features/movie_stats"
DELTA_USER_STATS_PATH   = "/app/delta/ratings_features/user_stats"
DELTA_ENRICHED_PATH     = "/app/delta/ratings_features/enriched_ratings"

# Feature Engineering çıktısı
DELTA_FULL_FEATURES_PATH = "/app/delta/ratings_features/full_features"

# Streaming checkpoint
CHECKPOINT_PATH         = "/app/delta/checkpoints/ratings"

# ─── Grafik / EDA ─────────────────────────────────────────────────────────────
PLOT_DIR       = "/app/delta/eda"
EDA_OUTPUT_DIR = "/app/delta/eda"

# ─── Ham Veri ─────────────────────────────────────────────────────────────────
MOVIES_CSV_PATH  = "/app/data/movies.csv"
RATINGS_CSV_PATH = "/app/data/ratings.csv"

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_URI = "http://mlflow:5000"

EXPERIMENT_PREPROCESSING = "movielens-preprocessing"
EXPERIMENT_ALS           = "movielens-als-experiments"
EXPERIMENT_CLS_REG       = "movielens-classification-regression"
EXPERIMENT_EXTENDED      = "movielens-extended-metrics"
EXPERIMENT_LOG_MODEL     = "movielens-als"
EXPERIMENT_SCALE         = "movielens-scale-analysis"

# ─── Yardımcı: Yol Doğrulama ─────────────────────────────────────────────────

def require_path(path: str, label: str = "") -> None:
    """Verilen Delta/dosya yolu yoksa hata logu yazıp çıkış yapar."""
    if not os.path.exists(path):
        logging.getLogger(__name__).error(
            "HATA: Gerekli yol bulunamadı: %s%s — önceki pipeline adımı tamamlandı mı?",
            path,
            f" ({label})" if label else "",
        )
        sys.exit(1)


# ─── Yardımcı: MLflow Sağlık Kontrolü ────────────────────────────────────────

def check_mlflow(uri: str = MLFLOW_URI) -> None:
    """MLflow sunucusuna bağlantı kontrolü yapar; erişilemezse uyarı loglar."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=5)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "MLflow sunucusuna (%s) erişilemiyor: %s — metrikler loglanmayabilir.", uri, e
        )
