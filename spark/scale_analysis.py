# Takım Üyesi: Arzu & Ayaz
# Görev: Veri Ölçeği Analizi
#   Farklı veri boyutlarında (100K → 500K → 1M → 5M → tüm veri) model
#   performansını ölçerek "büyük verinin fark yaratıp yaratmadığını" gösterir.
#   Çıktılar: MLflow logları + 2 grafik (RMSE vs ölçek, süre vs ölçek)

import logging, sys, os, time
sys.path.insert(0, "/app/spark")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow

from config import (
    DELTA_FULL_FEATURES_PATH as FEATURES_PATH,
    PLOT_DIR, MLFLOW_URI,
    require_path, check_mlflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

EXPERIMENT = "movielens-scale-analysis"

FEATURE_COLS = [
    "day_of_week", "hour_of_day", "month", "season", "is_weekend", "day_period",
    "user_avg_rating", "user_rating_count",
    "user_rating_variance", "user_activity_days",
    "movie_avg_rating", "movie_rating_count", "movie_age",
    "genre_count", "movie_popularity_trend",
    "user_genre_avg_rating", "rating_order", "time_since_last_rating",
]

# Analiz edilecek ölçekler; None = tüm veri
SCALE_SIZES = [100_000, 500_000, 1_000_000, 5_000_000, None]

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#2a2a3e",
    "axes.edgecolor": "#555577", "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4", "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4", "axes.titlecolor": "#cba6f7",
    "grid.color": "#44445a", "grid.alpha": 0.4, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
})
PALETTE = ["#cba6f7", "#89b4fa", "#a6e3a1", "#fab387", "#f38ba8",
           "#94e2d5", "#f9e2af", "#eba0ac"]


def save_fig(name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafik → %s", path)


def create_spark():
    spark = (
        SparkSession.builder
        .appName("MovieLens-ScaleAnalysis")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(PLOT_DIR, exist_ok=True)
    return spark


def load_full(spark):
    logger.info("Tam veri yükleniyor: %s", FEATURES_PATH)
    df = spark.read.format("delta").load(FEATURES_PATH)
    available = [c for c in FEATURE_COLS if c in df.columns]
    for c in available:
        df = df.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c)).cast("double"))
    assembler = VectorAssembler(inputCols=available, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df).withColumnRenamed("rating", "label").select("features", "label")
    df.cache()
    total = df.count()
    logger.info("Toplam satır: %d", total)
    return df, total, available


def _label(size, total):
    if size is None:
        n = total
        lbl = f"{total/1_000_000:.1f}M (tümü)" if total >= 1_000_000 else f"{total:,}"
    else:
        n = size
        lbl = f"{size//1_000}K" if size < 1_000_000 else f"{size//1_000_000}M"
    return n, lbl


def run_scale_experiment(df, total, model_name, model):
    """Verilen modeli tüm ölçeklerde eğitip sonuçları döndürür."""
    eval_rmse = RegressionEvaluator(labelCol="label", metricName="rmse")
    eval_mae  = RegressionEvaluator(labelCol="label", metricName="mae")
    eval_r2   = RegressionEvaluator(labelCol="label", metricName="r2")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    records = []
    for size in SCALE_SIZES:
        actual_n, label = _label(size, total)
        if size is not None and size >= total:
            # Veri zaten daha küçük, bu ölçeği geç
            logger.info("Ölçek %s >= toplam veri (%d), atlanıyor.", label, total)
            continue

        logger.info("── %s | ölçek=%s (%d satır)", model_name, label, actual_n)

        if size is not None:
            sampled = df.sample(False, size / total, seed=42).limit(size)
        else:
            sampled = df

        train, test = sampled.randomSplit([0.8, 0.2], seed=42)
        train.cache()

        t0 = time.time()
        fitted  = model.fit(train)
        elapsed = time.time() - t0

        preds = fitted.transform(test)
        rmse = eval_rmse.evaluate(preds)
        mae  = eval_mae.evaluate(preds)
        r2   = eval_r2.evaluate(preds)

        train.unpersist()

        run_name = f"scale_{model_name}_{label}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("scale_label", label)
            mlflow.log_metric("n_rows", actual_n)
            mlflow.log_metric("rmse", round(rmse, 4))
            mlflow.log_metric("mae",  round(mae, 4))
            mlflow.log_metric("r2",   round(r2, 4))
            mlflow.log_metric("train_time_sec", round(elapsed, 2))

        logger.info("  RMSE=%.4f  MAE=%.4f  R²=%.4f  Süre=%.1fs", rmse, mae, r2, elapsed)
        records.append({
            "model": model_name, "scale_label": label,
            "n_rows": actual_n,
            "rmse": round(rmse, 4), "mae": round(mae, 4),
            "r2": round(r2, 4), "train_time_sec": round(elapsed, 2),
        })

    return records


def plot_scale_rmse(all_records):
    """RMSE vs veri boyutu çizgi grafiği — ana 'büyük veri fark yaratıyor mu?' görseli."""
    import pandas as pd
    df = pd.DataFrame(all_records)
    models = df["model"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Veri Ölçeği Analizi: Daha Fazla Veri Daha İyi Sonuç Verir mi?",
                 fontsize=15, color="#cba6f7")

    # Sol: RMSE vs n_rows
    ax = axes[0]
    for i, m in enumerate(models):
        sub = df[df["model"] == m].sort_values("n_rows")
        ax.plot(sub["n_rows"], sub["rmse"], marker="o", linewidth=2.5,
                markersize=8, label=m.replace("_", " "), color=PALETTE[i])
        for _, row in sub.iterrows():
            ax.annotate(f"{row['rmse']:.3f}",
                        (row["n_rows"], row["rmse"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=PALETTE[i])
    ax.set_xscale("log")
    ax.set_xlabel("Eğitim Verisi (satır, log ölçek)")
    ax.set_ylabel("RMSE (düşük = iyi)")
    ax.set_title("RMSE vs Veri Boyutu")
    ax.legend()
    ax.grid(True)
    # x eksen etiketleri okunabilir olsun
    xticks = sorted(df["n_rows"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        f"{v//1_000_000:.0f}M" if v >= 1_000_000 else f"{v//1_000:.0f}K"
        for v in xticks
    ], fontsize=9)

    # Sağ: Eğitim süresi vs n_rows
    ax = axes[1]
    for i, m in enumerate(models):
        sub = df[df["model"] == m].sort_values("n_rows")
        ax.plot(sub["n_rows"], sub["train_time_sec"], marker="s", linewidth=2.5,
                markersize=8, label=m.replace("_", " "), color=PALETTE[i], linestyle="--")
        for _, row in sub.iterrows():
            ax.annotate(f"{row['train_time_sec']:.0f}s",
                        (row["n_rows"], row["train_time_sec"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=PALETTE[i])
    ax.set_xscale("log")
    ax.set_xlabel("Eğitim Verisi (satır, log ölçek)")
    ax.set_ylabel("Eğitim Süresi (saniye)")
    ax.set_title("Eğitim Süresi vs Veri Boyutu")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        f"{v//1_000_000:.0f}M" if v >= 1_000_000 else f"{v//1_000:.0f}K"
        for v in xticks
    ], fontsize=9)

    plt.tight_layout()
    save_fig("23_scale_analysis")


def plot_scale_r2(all_records):
    """R² vs veri boyutu — açıklanan varyans ne kadar artıyor?"""
    import pandas as pd
    df = pd.DataFrame(all_records)
    models = df["model"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("R² Skoru vs Veri Boyutu", fontsize=14, color="#cba6f7")

    for i, m in enumerate(models):
        sub = df[df["model"] == m].sort_values("n_rows")
        ax.plot(sub["n_rows"], sub["r2"], marker="^", linewidth=2.5,
                markersize=8, label=m.replace("_", " "), color=PALETTE[i])
        for _, row in sub.iterrows():
            ax.annotate(f"{row['r2']:.3f}",
                        (row["n_rows"], row["r2"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=PALETTE[i])

    ax.set_xscale("log")
    ax.set_xlabel("Eğitim Verisi (satır, log ölçek)")
    ax.set_ylabel("R² (yüksek = iyi)")
    ax.set_title("R² vs Veri Boyutu")
    ax.legend()
    ax.grid(True)
    xticks = sorted(df["n_rows"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        f"{v//1_000_000:.0f}M" if v >= 1_000_000 else f"{v//1_000:.0f}K"
        for v in xticks
    ], fontsize=9)

    plt.tight_layout()
    save_fig("24_scale_r2")


def main():
    require_path(FEATURES_PATH, "full_features (feature_engineering.py çalıştırıldı mı?)")
    check_mlflow()
    spark = create_spark()
    df, total, available = load_full(spark)

    models = {
        "Linear_Regression": LinearRegression(
            featuresCol="features", labelCol="label",
            maxIter=20, regParam=0.1, elasticNetParam=0.3),
        "Random_Forest": RandomForestRegressor(
            featuresCol="features", labelCol="label",
            numTrees=50, maxDepth=5, minInstancesPerNode=10, seed=42),
    }

    all_records = []
    for model_name, model in models.items():
        logger.info("=" * 60)
        logger.info("MODEL: %s", model_name)
        logger.info("=" * 60)
        records = run_scale_experiment(df, total, model_name, model)
        all_records.extend(records)

    if not all_records:
        logger.error("Hiç sonuç üretilemedi — veri yeterli mi?")
        spark.stop()
        return

    logger.info("Grafik üretiliyor...")
    plot_scale_rmse(all_records)
    plot_scale_r2(all_records)

    # Grafikleri MLflow'a artifact olarak logla
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="scale_analysis_summary"):
        import pandas as pd
        df_res = pd.DataFrame(all_records)
        best_row = df_res.loc[df_res["rmse"].idxmin()]
        mlflow.log_param("best_model", best_row["model"])
        mlflow.log_param("best_scale", best_row["scale_label"])
        mlflow.log_metric("best_rmse", best_row["rmse"])
        for png in ["23_scale_analysis", "24_scale_r2"]:
            path = os.path.join(PLOT_DIR, f"{png}.png")
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="scale_plots")

    logger.info("Veri ölçeği analizi tamamlandı.")
    spark.stop()


if __name__ == "__main__":
    main()
