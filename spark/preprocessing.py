# Takım Üyesi: Arzu (Kıdemli Mühendis) & Ayaz (Junior Mühendis)
# Görev: Veri Ön İşleme Pipeline'ı
#   - Missing value analizi (null/NaN tespiti ve raporlama)
#   - Duplikasyon kontrolü (aynı kullanıcı-film-zaman üçlüsü)
#   - Outlier detection (IQR yöntemiyle)
#   - Aşırı aktif kullanıcı tespiti (10.000+ rating)
#   - Şüpheli puanlama kalıpları (tüm filmlere aynı puanı verenler)
#   - Temizlenmiş veriyi Delta Lake'e yazma
#   - Veri kalitesi raporu üretme ve görselleştirme
#   - Tüm adımları MLflow'a loglama (metrik + artifact)

import logging
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, avg, stddev, lit, when,
    round as spark_round, isnan, isnull,
    abs as spark_abs,
)
from pyspark.sql.types import FloatType

# ─── Loglama Ayarları ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
DELTA_RATINGS_PATH   = "/app/delta/ratings"
DELTA_CLEANED_PATH   = "/app/delta/ratings_cleaned"
REPORT_DIR           = "/app/delta/preprocessing_report"
MLFLOW_URI           = "http://mlflow:5000"
EXPERIMENT_NAME      = "movielens-preprocessing"

# Eşik değerleri
HYPERACTIVE_THRESHOLD   = 10_000   # Bu sayıdan fazla rating veren → aşırı aktif
SUSPECT_MIN_RATINGS     = 50       # Şüpheli analizi için minimum rating sayısı
SUSPECT_STDDEV_THRESH   = 0.1      # Stddev bu altındaysa → hep aynı puanı veriyor
VALID_RATING_MIN        = 0.5
VALID_RATING_MAX        = 5.0

# ─── Grafik Stili ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor":   "#2a2a3e",
    "axes.edgecolor":   "#555577",
    "text.color":       "#cdd6f4",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "axes.titlecolor":  "#cba6f7",
    "grid.color":       "#44445a",
    "grid.alpha":       0.4,
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
})
PALETTE = ["#cba6f7", "#89b4fa", "#a6e3a1", "#fab387", "#f38ba8",
           "#94e2d5", "#f9e2af", "#eba0ac", "#b4befe", "#74c7ec"]


def save_fig(name: str) -> None:
    path = os.path.join(REPORT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafik kaydedildi → %s", path)


# ─── Spark Oturumu ────────────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    """Delta Lake destekli Spark oturumu başlatır."""
    try:
        spark = (
            SparkSession.builder
            .appName("MovieLens-Preprocessing")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        os.makedirs(REPORT_DIR, exist_ok=True)
        logger.info("Spark oturumu hazır.")
        return spark
    except Exception as e:
        logger.error("Spark oturumu başlatılamadı: %s", e)
        sys.exit(1)


# ─── ADIM 1: Missing Value Analizi ────────────────────────────────────────────

def analyze_missing_values(df: DataFrame) -> dict:
    """
    Tüm kolonlardaki null ve NaN değerleri tespit eder, raporlar.
    Returns: {kolon_adı: {"null_count": int, "null_pct": float}}
    """
    logger.info("=" * 60)
    logger.info("ADIM 1: Missing Value Analizi")
    logger.info("=" * 60)

    total_rows = df.count()
    report = {}

    for col_name in df.columns:
        col_type = dict(df.dtypes).get(col_name, "unknown")

        # Hem null hem NaN kontrolü
        if col_type in ("float", "double"):
            null_count = df.filter(
                isnull(col(col_name)) | isnan(col(col_name))
            ).count()
        else:
            null_count = df.filter(isnull(col(col_name))).count()

        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0.0
        report[col_name] = {"null_count": null_count, "null_pct": round(null_pct, 4)}

        status = "✓ Temiz" if null_count == 0 else f"⚠ {null_count} eksik ({null_pct:.2f}%)"
        logger.info("  [%s] (%s): %s", col_name, col_type, status)

    total_missing = sum(v["null_count"] for v in report.values())
    logger.info("Toplam eksik değer: %d / %d hücre", total_missing, total_rows * len(df.columns))

    return report


def drop_missing_values(df: DataFrame) -> DataFrame:
    """Kritik kolonlardaki null satırları düşürür."""
    critical_cols = ["userId", "movieId", "rating", "timestamp"]
    before = df.count()

    for c in critical_cols:
        if c in df.columns:
            df = df.filter(col(c).isNotNull())
            col_type = dict(df.dtypes).get(c, "unknown")
            if col_type in ("float", "double"):
                df = df.filter(~isnan(col(c)))

    after = df.count()
    dropped = before - after
    logger.info("Missing value temizliği: %d satır düşürüldü (%d → %d)", dropped, before, after)
    return df


# ─── ADIM 2: Duplikasyon Kontrolü ─────────────────────────────────────────────

def detect_duplicates(df: DataFrame) -> DataFrame:
    """
    Aynı (userId, movieId, timestamp) üçlüsüne sahip duplike satırları tespit eder.
    """
    logger.info("=" * 60)
    logger.info("ADIM 2: Duplikasyon Kontrolü")
    logger.info("=" * 60)

    total = df.count()

    dup_counts = (
        df.groupBy("userId", "movieId", "timestamp")
        .agg(count("*").alias("dup_count"))
        .filter(col("dup_count") > 1)
    )
    dup_groups = dup_counts.count()
    dup_total = dup_counts.select(
        (col("dup_count") - 1)  # Her gruptaki fazlalık
    ).groupBy().sum().first()

    dup_extra = dup_total[0] if dup_total and dup_total[0] else 0

    logger.info("Toplam satır: %d", total)
    logger.info("Duplike grup sayısı: %d", dup_groups)
    logger.info("Fazladan duplike satır: %d", dup_extra)

    if dup_groups > 0:
        logger.info("Örnek duplike kayıtlar:")
        dup_counts.orderBy(col("dup_count").desc()).show(5, truncate=False)

    return dup_counts


def remove_duplicates(df: DataFrame) -> DataFrame:
    """Duplike satırları kaldırır (ilk kaydı tutar)."""
    before = df.count()
    df_clean = df.dropDuplicates(["userId", "movieId", "timestamp"])
    after = df_clean.count()
    logger.info("Duplikasyon temizliği: %d satır kaldırıldı (%d → %d)", before - after, before, after)
    return df_clean


# ─── ADIM 3: Outlier Detection (IQR Yöntemi) ─────────────────────────────────

def detect_rating_outliers(df: DataFrame) -> dict:
    """
    Rating kolonunda IQR (Interquartile Range) yöntemiyle outlier tespit eder.
    Q1 - 1.5*IQR altı veya Q3 + 1.5*IQR üstü → outlier
    """
    logger.info("=" * 60)
    logger.info("ADIM 3: Outlier Detection (IQR — Rating)")
    logger.info("=" * 60)

    quantiles = df.approxQuantile("rating", [0.25, 0.5, 0.75], 0.01)
    q1, median, q3 = quantiles[0], quantiles[1], quantiles[2]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    logger.info("  Q1=%.2f | Median=%.2f | Q3=%.2f | IQR=%.2f", q1, median, q3, iqr)
    logger.info("  Alt sınır: %.2f | Üst sınır: %.2f", lower_bound, upper_bound)

    outlier_count = df.filter(
        (col("rating") < lower_bound) | (col("rating") > upper_bound)
    ).count()

    # Geçersiz rating kontrolü (0.5-5.0 aralığı dışı)
    invalid_count = df.filter(
        (col("rating") < VALID_RATING_MIN) | (col("rating") > VALID_RATING_MAX)
    ).count()

    total = df.count()
    logger.info("  IQR outlier sayısı: %d (%.2f%%)", outlier_count, outlier_count / total * 100)
    logger.info("  Geçersiz rating (%.1f-%.1f dışı): %d",
                VALID_RATING_MIN, VALID_RATING_MAX, invalid_count)

    return {
        "q1": q1, "median": median, "q3": q3, "iqr": iqr,
        "lower_bound": lower_bound, "upper_bound": upper_bound,
        "outlier_count": outlier_count, "invalid_count": invalid_count,
    }


def remove_invalid_ratings(df: DataFrame) -> DataFrame:
    """Geçersiz rating aralığındaki satırları kaldırır."""
    before = df.count()
    df_clean = df.filter(
        (col("rating") >= VALID_RATING_MIN) & (col("rating") <= VALID_RATING_MAX)
    )
    after = df_clean.count()
    logger.info("Geçersiz rating temizliği: %d satır kaldırıldı", before - after)
    return df_clean


# ─── ADIM 4: Aşırı Aktif Kullanıcı Tespiti ───────────────────────────────────

def detect_hyperactive_users(df: DataFrame) -> DataFrame:
    """
    Belirli eşik değerinin üzerinde rating veren kullanıcıları tespit eder.
    Bu kullanıcılar bot veya otomatik sistem olabilir.
    """
    logger.info("=" * 60)
    logger.info("ADIM 4: Aşırı Aktif Kullanıcı Tespiti (>%d rating)", HYPERACTIVE_THRESHOLD)
    logger.info("=" * 60)

    user_counts = (
        df.groupBy("userId")
        .agg(
            count("*").alias("rating_count"),
            spark_round(avg("rating"), 2).alias("avg_rating"),
            spark_round(stddev("rating"), 2).alias("stddev_rating"),
        )
    )

    hyperactive = user_counts.filter(col("rating_count") > HYPERACTIVE_THRESHOLD)
    hyper_count = hyperactive.count()
    total_users = user_counts.count()

    logger.info("Toplam kullanıcı: %d", total_users)
    logger.info("Aşırı aktif kullanıcı (>%d): %d (%.2f%%)",
                HYPERACTIVE_THRESHOLD, hyper_count, hyper_count / total_users * 100)

    if hyper_count > 0:
        logger.info("Aşırı aktif kullanıcılar:")
        hyperactive.orderBy(col("rating_count").desc()).show(10, truncate=False)

    return hyperactive


# ─── ADIM 5: Şüpheli Puanlama Kalıpları ──────────────────────────────────────

def detect_suspicious_patterns(df: DataFrame) -> DataFrame:
    """
    Tüm filmlere aynı (veya neredeyse aynı) puanı veren kullanıcıları tespit eder.
    Kriter: Minimum 50 rating vermiş VE standart sapması < 0.1
    """
    logger.info("=" * 60)
    logger.info("ADIM 5: Şüpheli Puanlama Kalıpları Tespiti")
    logger.info("=" * 60)

    user_stats = (
        df.groupBy("userId")
        .agg(
            count("*").alias("rating_count"),
            spark_round(avg("rating"), 3).alias("avg_rating"),
            spark_round(stddev("rating"), 4).alias("stddev_rating"),
        )
        .filter(col("rating_count") >= SUSPECT_MIN_RATINGS)
    )

    # stddev null olabilir (tek rating verenlerde) — onu da şüpheli say
    suspicious = user_stats.filter(
        (col("stddev_rating") < SUSPECT_STDDEV_THRESH) |
        (col("stddev_rating").isNull())
    )

    suspect_count = suspicious.count()
    total_analyzed = user_stats.count()

    logger.info("Analiz edilen kullanıcı (>=%d rating): %d", SUSPECT_MIN_RATINGS, total_analyzed)
    logger.info("Şüpheli kullanıcı (stddev < %.2f): %d (%.2f%%)",
                SUSPECT_STDDEV_THRESH, suspect_count,
                suspect_count / total_analyzed * 100 if total_analyzed > 0 else 0)

    if suspect_count > 0:
        logger.info("Şüpheli kullanıcı örnekleri:")
        suspicious.orderBy(col("rating_count").desc()).show(10, truncate=False)

    return suspicious


# ─── ADIM 6: Veri Kalitesi Raporu Görselleştirmesi ────────────────────────────

def plot_preprocessing_summary(
    total_before: int,
    total_after: int,
    missing_report: dict,
    outlier_info: dict,
    hyper_count: int,
    suspect_count: int,
    dup_extra: int,
) -> None:
    """Tüm ön işleme sonuçlarını özetleyen grafik üretir."""

    # ── Grafik 1: Temizleme Özet Bar Chart ────────────────────────────────
    categories = [
        "Eksik\nDeğer",
        "Duplike\nSatır",
        "Geçersiz\nRating",
        "Aşırı Aktif\nKullanıcı",
        "Şüpheli\nKullanıcı",
    ]
    total_missing = sum(v["null_count"] for v in missing_report.values())
    values = [
        total_missing,
        dup_extra,
        outlier_info.get("invalid_count", 0),
        hyper_count,
        suspect_count,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Sol: Tespit edilen sorunlar
    bars = axes[0].bar(categories, values, color=PALETTE[:5], edgecolor="#1e1e2e", width=0.6)
    axes[0].bar_label(bars, labels=[f"{v:,}" for v in values],
                      padding=4, color="#cdd6f4", fontsize=10)
    axes[0].set_title("Tespit Edilen Veri Kalitesi Sorunları")
    axes[0].set_ylabel("Sayı")
    axes[0].grid(axis="y")

    # Sağ: Önce/Sonra karşılaştırma
    labels = ["Ham Veri", "Temiz Veri"]
    sizes = [total_before, total_after]
    removed = total_before - total_after
    colors_pie = [PALETTE[4], PALETTE[2]]

    bars2 = axes[1].bar(labels, sizes, color=colors_pie, edgecolor="#1e1e2e", width=0.5)
    axes[1].bar_label(bars2, labels=[f"{v:,}" for v in sizes],
                      padding=4, color="#cdd6f4", fontsize=11)
    axes[1].set_title(f"Veri Temizleme Sonucu (Kaldırılan: {removed:,})")
    axes[1].set_ylabel("Satır Sayısı")
    axes[1].grid(axis="y")

    plt.tight_layout()
    save_fig("preprocessing_summary")

    # ── Grafik 2: Rating Dağılımı (Box Plot) ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    info = outlier_info
    box_data = {
        "Q1": info["q1"], "Median": info["median"], "Q3": info["q3"],
        "Alt Sınır": info["lower_bound"], "Üst Sınır": info["upper_bound"],
    }
    x_pos = range(len(box_data))
    bars3 = ax2.bar(list(box_data.keys()), list(box_data.values()),
                    color=PALETTE[1], edgecolor="#1e1e2e", width=0.5)
    ax2.bar_label(bars3, labels=[f"{v:.2f}" for v in box_data.values()],
                  padding=4, color="#cdd6f4", fontsize=10)
    ax2.set_title("Rating IQR İstatistikleri")
    ax2.set_ylabel("Değer")
    ax2.grid(axis="y")
    plt.tight_layout()
    save_fig("preprocessing_iqr_stats")


# ─── MLflow Loglama ───────────────────────────────────────────────────────────

def log_quality_metrics_to_mlflow(
    total_before: int,
    total_after: int,
    missing_report: dict,
    outlier_info: dict,
    hyper_count: int,
    suspect_count: int,
    dup_extra: int,
) -> None:
    """
    Tüm veri kalitesi metriklerini ve görselleştirme grafiklerini
    MLflow Tracking Server'a loglar.
    """
    logger.info("=" * 60)
    logger.info("MLflow'a Veri Kalitesi Metrikleri Loglanıyor...")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    total_missing = sum(v["null_count"] for v in missing_report.values())
    removed = total_before - total_after
    removed_pct = (removed / total_before * 100) if total_before > 0 else 0.0

    with mlflow.start_run(run_name="data_quality_report"):

        # ── Genel Veri Metrikleri ─────────────────────────────────────────
        mlflow.log_param("pipeline_stage", "preprocessing")
        mlflow.log_param("source_path",    DELTA_RATINGS_PATH)
        mlflow.log_param("output_path",    DELTA_CLEANED_PATH)
        mlflow.log_param("hyperactive_threshold",  HYPERACTIVE_THRESHOLD)
        mlflow.log_param("suspect_min_ratings",     SUSPECT_MIN_RATINGS)
        mlflow.log_param("suspect_stddev_threshold", SUSPECT_STDDEV_THRESH)
        mlflow.log_param("valid_rating_range", f"{VALID_RATING_MIN}-{VALID_RATING_MAX}")

        # ── Satır Sayıları ────────────────────────────────────────────────
        mlflow.log_metric("rows_before_cleaning",  total_before)
        mlflow.log_metric("rows_after_cleaning",   total_after)
        mlflow.log_metric("rows_removed",           removed)
        mlflow.log_metric("rows_removed_pct",       round(removed_pct, 4))

        # ── Missing Value Metrikleri ──────────────────────────────────────
        mlflow.log_metric("missing_total", total_missing)
        for col_name, info in missing_report.items():
            mlflow.log_metric(f"missing_{col_name}_count", info["null_count"])
            mlflow.log_metric(f"missing_{col_name}_pct",   info["null_pct"])

        # ── Duplikasyon Metrikleri ─────────────────────────────────────────
        mlflow.log_metric("duplicate_rows", dup_extra)

        # ── Outlier / IQR Metrikleri ──────────────────────────────────────
        mlflow.log_metric("rating_q1",          outlier_info["q1"])
        mlflow.log_metric("rating_median",      outlier_info["median"])
        mlflow.log_metric("rating_q3",          outlier_info["q3"])
        mlflow.log_metric("rating_iqr",         outlier_info["iqr"])
        mlflow.log_metric("rating_lower_bound", outlier_info["lower_bound"])
        mlflow.log_metric("rating_upper_bound", outlier_info["upper_bound"])
        mlflow.log_metric("outlier_count",      outlier_info["outlier_count"])
        mlflow.log_metric("invalid_rating_count", outlier_info["invalid_count"])

        # ── Kullanıcı Kalite Metrikleri ────────────────────────────────────
        mlflow.log_metric("hyperactive_users",  hyper_count)
        mlflow.log_metric("suspicious_users",   suspect_count)

        # ── Görselleştirme Artifact'larını Logla ──────────────────────────
        summary_plot = os.path.join(REPORT_DIR, "preprocessing_summary.png")
        iqr_plot     = os.path.join(REPORT_DIR, "preprocessing_iqr_stats.png")

        if os.path.exists(summary_plot):
            mlflow.log_artifact(summary_plot, artifact_path="preprocessing_plots")
            logger.info("Artifact loglandı: %s", summary_plot)
        if os.path.exists(iqr_plot):
            mlflow.log_artifact(iqr_plot, artifact_path="preprocessing_plots")
            logger.info("Artifact loglandı: %s", iqr_plot)

        logger.info("MLflow loglama tamamlandı — Experiment: %s", EXPERIMENT_NAME)


# ─── ADIM 7: Temiz Veriyi Yazma ──────────────────────────────────────────────

def write_cleaned_data(df: DataFrame, path: str) -> None:
    """Temizlenmiş veriyi Delta Lake'e yazar."""
    try:
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .save(path)
        )
        final_count = df.count()
        logger.info("Temiz veri Delta Lake'e yazıldı → %s (%d satır)", path, final_count)
    except Exception as e:
        logger.error("Temiz veri yazma hatası: %s", e)
        raise


# ─── ANA FONKSİYON ────────────────────────────────────────────────────────────

def main():
    spark = create_spark_session()

    # ── Ham veriyi yükle ──────────────────────────────────────────────────
    logger.info("Ham veri yükleniyor: %s", DELTA_RATINGS_PATH)
    df = spark.read.format("delta").load(DELTA_RATINGS_PATH)
    total_before = df.count()
    logger.info("Ham veri yüklendi — %d satır, %d kolon", total_before, len(df.columns))
    logger.info("Şema:")
    df.printSchema()

    # ── Adım 1: Missing value analizi ────────────────────────────────────
    missing_report = analyze_missing_values(df)
    df = drop_missing_values(df)

    # ── Adım 2: Duplikasyon kontrolü ─────────────────────────────────────
    dup_df = detect_duplicates(df)
    dup_extra = 0
    if dup_df.count() > 0:
        dup_sum = dup_df.selectExpr("sum(dup_count - 1)").first()
        dup_extra = dup_sum[0] if dup_sum and dup_sum[0] else 0
    df = remove_duplicates(df)

    # ── Adım 3: Outlier detection ────────────────────────────────────────
    outlier_info = detect_rating_outliers(df)
    df = remove_invalid_ratings(df)

    # ── Adım 4: Aşırı aktif kullanıcı tespiti ────────────────────────────
    hyperactive_df = detect_hyperactive_users(df)
    hyper_count = hyperactive_df.count()
    # Not: Aşırı aktif kullanıcıları silmiyoruz, sadece raporluyoruz.
    # İstenirse aşağıdaki satırı aktif edin:
    # hyper_ids = [row.userId for row in hyperactive_df.select("userId").collect()]
    # df = df.filter(~col("userId").isin(hyper_ids))

    # ── Adım 5: Şüpheli puanlama kalıpları ───────────────────────────────
    suspicious_df = detect_suspicious_patterns(df)
    suspect_count = suspicious_df.count()
    # Not: Şüpheli kullanıcıları silmiyoruz, sadece raporluyoruz.
    # İstenirse aşağıdaki satırı aktif edin:
    # suspect_ids = [row.userId for row in suspicious_df.select("userId").collect()]
    # df = df.filter(~col("userId").isin(suspect_ids))

    # ── Adım 6: Görselleştirme ───────────────────────────────────────────
    total_after = df.count()
    logger.info("=" * 60)
    logger.info("VERİ KALİTESİ RAPORU — ÖZET")
    logger.info("=" * 60)
    logger.info("  Ham veri satır sayısı       : %d", total_before)
    logger.info("  Temiz veri satır sayısı      : %d", total_after)
    logger.info("  Kaldırılan satır             : %d (%.2f%%)",
                total_before - total_after,
                (total_before - total_after) / total_before * 100)
    logger.info("  Duplike satır                : %d", dup_extra)
    logger.info("  Geçersiz rating              : %d", outlier_info.get("invalid_count", 0))
    logger.info("  Aşırı aktif kullanıcı        : %d", hyper_count)
    logger.info("  Şüpheli puanlama kullanıcısı : %d", suspect_count)
    logger.info("=" * 60)

    plot_preprocessing_summary(
        total_before, total_after, missing_report,
        outlier_info, hyper_count, suspect_count, dup_extra,
    )

    # ── Adım 7: Temiz veriyi Delta Lake'e yaz ────────────────────────────
    write_cleaned_data(df, DELTA_CLEANED_PATH)

    # ── Adım 8: MLflow'a veri kalitesi metriklerini logla ─────────────
    log_quality_metrics_to_mlflow(
        total_before, total_after, missing_report,
        outlier_info, hyper_count, suspect_count, dup_extra,
    )

    logger.info("Ön işleme pipeline'ı tamamlandı.")
    spark.stop()


if __name__ == "__main__":
    main()
