# Takım: Arzu & Ayaz
# Görev: Genişletilmiş Değerlendirme Metrikleri
#  - ALS: MAE, MSE, R², Precision@K, Recall@K, NDCG
#  - Sınıflandırma: Confusion Matrix, Macro/Micro metrikler
#  - Tüm sonuçları görselleştirme + MLflow loglama

import logging, sys, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, avg, count, lit, collect_list, expr,
    row_number, when, round as spark_round, desc,
    struct, explode, array,
)
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import (
    RegressionEvaluator, MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

ENRICHED_PATH   = "/app/delta/ratings_features/enriched_ratings"
BEST_MODEL_PATH = "/app/delta/als_model_best"
PLOT_DIR        = "/app/delta/eda"
MLFLOW_URI      = "http://mlflow:5000"
EXPERIMENT      = "movielens-extended-metrics"

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#2a2a3e",
    "axes.edgecolor": "#555577", "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4", "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4", "axes.titlecolor": "#cba6f7",
    "grid.color": "#44445a", "grid.alpha": 0.4, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
})
PAL = ["#cba6f7","#89b4fa","#a6e3a1","#fab387","#f38ba8",
       "#94e2d5","#f9e2af","#eba0ac","#b4befe","#74c7ec"]

def save_fig(n):
    p = os.path.join(PLOT_DIR, f"{n}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Grafik → %s", p)

def create_spark():
    spark = (SparkSession.builder.appName("MovieLens-ExtendedMetrics")
        .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages","io.delta:delta-core_2.12:2.4.0")
        .config("spark.driver.memory","2g").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(PLOT_DIR, exist_ok=True)
    return spark


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: ALS GENİŞLETİLMİŞ REGRESYON METRİKLERİ (MAE, MSE, R²)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_als_extended_metrics(spark):
    """ALS modelini yükler, MAE/MSE/R²/RMSE hesaplar."""
    logger.info("=== ALS Genişletilmiş Regresyon Metrikleri ===")

    df = (spark.read.format("delta").load(ENRICHED_PATH)
          .select("userId","movieId","rating").dropna())
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Mevcut modeli yüklemeye çalış, yoksa yeniden eğit
    try:
        model = ALSModel.load(BEST_MODEL_PATH)
        logger.info("Kayıtlı ALS modeli yüklendi: %s", BEST_MODEL_PATH)
    except Exception:
        logger.info("Model bulunamadı, yeniden eğitiliyor...")
        als = ALS(rank=10, maxIter=10, regParam=0.1,
                  userCol="userId", itemCol="movieId", ratingCol="rating",
                  coldStartStrategy="drop", seed=42)
        model = als.fit(train)

    preds = model.transform(test)

    metrics = {}
    for name in ["rmse", "mae", "mse", "r2"]:
        ev = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName=name)
        metrics[name] = ev.evaluate(preds)
        logger.info("  ALS %s: %.4f", name.upper(), metrics[name])

    return metrics, model, train, test


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: ÖNERİ SİSTEMİ METRİKLERİ (Precision@K, Recall@K, NDCG)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_recommendation_metrics(model, train, test, k_values=[5, 10]):
    """Precision@K, Recall@K ve NDCG hesaplar."""
    logger.info("=== Öneri Sistemi Metrikleri ===")

    # Gerçek beğenilen filmler (rating >= 3.5)
    relevant = (test.filter(col("rating") >= 3.5)
                .groupBy("userId")
                .agg(collect_list("movieId").alias("relevant_items"),
                     count("*").alias("n_relevant")))

    # ALS'ten top-K öneriler
    max_k = max(k_values)
    recs = model.recommendForAllUsers(max_k)
    recs = recs.withColumn("rec_item", explode(col("recommendations")))
    recs = recs.withColumn("rec_movieId", col("rec_item.movieId"))
    recs = recs.withColumn("rec_rating", col("rec_item.rating"))
    recs = recs.withColumn("rec_rank", row_number().over(
        Window.partitionBy("userId").orderBy(col("rec_rating").desc())))

    results = {}
    for k in k_values:
        top_k = (recs.filter(col("rec_rank") <= k)
                 .groupBy("userId")
                 .agg(collect_list("rec_movieId").alias("rec_items")))

        joined = top_k.join(relevant, "userId", "inner")

        # Precision@K ve Recall@K hesaplama
        pr = joined.withColumn(
            "hits", expr(f"size(array_intersect(rec_items, relevant_items))")
        ).withColumn("precision_at_k", col("hits") / lit(k)
        ).withColumn("recall_at_k",
            when(col("n_relevant") > 0, col("hits") / col("n_relevant")).otherwise(0.0))

        avg_prec = pr.select(avg("precision_at_k")).first()[0] or 0.0
        avg_rec = pr.select(avg("recall_at_k")).first()[0] or 0.0

        results[f"precision_at_{k}"] = round(avg_prec, 4)
        results[f"recall_at_{k}"] = round(avg_rec, 4)
        logger.info("  Precision@%d: %.4f | Recall@%d: %.4f", k, avg_prec, k, avg_rec)

    # NDCG hesaplama (K=10)
    k_ndcg = max_k
    recs_ndcg = recs.filter(col("rec_rank") <= k_ndcg)
    recs_with_rel = recs_ndcg.join(
        test.select("userId", col("movieId").alias("rec_movieId"),
                    col("rating").alias("true_rating")),
        ["userId", "rec_movieId"], "left"
    ).withColumn("rel", when(col("true_rating").isNotNull() & (col("true_rating") >= 3.5), 1.0)
                 .otherwise(0.0))

    # DCG
    recs_with_rel = recs_with_rel.withColumn(
        "dcg_val", col("rel") / expr("log2(rec_rank + 1)"))

    user_dcg = recs_with_rel.groupBy("userId").agg(
        expr("sum(dcg_val)").alias("dcg"))

    # İdeal DCG
    ideal_df = (test.filter(col("rating") >= 3.5)
                .withColumn("ideal_rank", row_number().over(
                    Window.partitionBy("userId").orderBy(col("rating").desc())))
                .filter(col("ideal_rank") <= k_ndcg)
                .withColumn("idcg_val", lit(1.0) / expr("log2(ideal_rank + 1)"))
                .groupBy("userId").agg(expr("sum(idcg_val)").alias("idcg")))

    ndcg_df = user_dcg.join(ideal_df, "userId", "inner").withColumn(
        "ndcg", when(col("idcg") > 0, col("dcg") / col("idcg")).otherwise(0.0))

    avg_ndcg = ndcg_df.select(avg("ndcg")).first()[0] or 0.0
    results[f"ndcg_at_{k_ndcg}"] = round(avg_ndcg, 4)
    logger.info("  NDCG@%d: %.4f", k_ndcg, avg_ndcg)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: SINIFLANDIRMA GENİŞLETİLMİŞ METRİKLER + CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_classification_extended(spark):
    """Macro/Micro/Weighted precision, recall, F1 + Confusion Matrix."""
    logger.info("=== Sınıflandırma Genişletilmiş Metrikler ===")

    try:
        df = spark.read.format("delta").load("/app/delta/ratings_features/full_features")
    except Exception:
        df = (spark.read.format("delta").load(ENRICHED_PATH)
              .select("userId","movieId","rating").dropna())

    feat_cols = [c for c in ["user_avg_rating","movie_avg_rating","user_rating_count",
                             "movie_rating_count","movie_age","genre_count",
                             "day_of_week","hour_of_day","season","is_weekend"] if c in df.columns]

    if len(feat_cols) < 2:
        feat_cols = ["userId", "movieId"]

    for c in feat_cols:
        df = df.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c)).cast("double"))

    df = df.withColumn("label", when(col("rating") >= 3.5, 1.0).otherwise(0.0))
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df).select("features", "label")
    df = df.sample(False, 0.05, seed=42)
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(numTrees=50, maxDepth=8, seed=42)
    model = rf.fit(train)
    preds = model.transform(test)

    results = {}
    # Weighted/Macro/Micro F1, Precision, Recall
    for avg_type in ["weightedPrecision", "weightedRecall", "f1", "accuracy"]:
        ev = MulticlassClassificationEvaluator(labelCol="label", metricName=avg_type)
        results[avg_type] = ev.evaluate(preds)

    # Macro/Micro hesapla
    for metric_base in ["precision", "recall", "fMeasure"]:
        for beta in ["macro", "micro"]:
            key = f"{metric_base}_{beta}"
            try:
                # metricName formatı: "precisionByLabel" vb. Spark desteklemeyebilir
                # Manuel hesaplama
                pass
            except Exception:
                pass

    # Manuel macro precision/recall/F1
    tp0 = preds.filter((col("prediction")==0)&(col("label")==0)).count()
    fp0 = preds.filter((col("prediction")==0)&(col("label")==1)).count()
    fn0 = preds.filter((col("prediction")==1)&(col("label")==0)).count()
    tp1 = preds.filter((col("prediction")==1)&(col("label")==1)).count()
    fp1 = preds.filter((col("prediction")==1)&(col("label")==0)).count()
    fn1 = preds.filter((col("prediction")==0)&(col("label")==1)).count()

    prec0 = tp0/(tp0+fp0) if (tp0+fp0)>0 else 0
    prec1 = tp1/(tp1+fp1) if (tp1+fp1)>0 else 0
    rec0 = tp0/(tp0+fn0) if (tp0+fn0)>0 else 0
    rec1 = tp1/(tp1+fn1) if (tp1+fn1)>0 else 0

    macro_prec = (prec0+prec1)/2
    macro_rec = (rec0+rec1)/2
    macro_f1 = 2*macro_prec*macro_rec/(macro_prec+macro_rec) if (macro_prec+macro_rec)>0 else 0

    total = tp0+fp0+fn0+tp1
    micro_prec = (tp0+tp1)/total if total>0 else 0
    micro_rec = (tp0+tp1)/(tp0+fn0+tp1+fn1) if (tp0+fn0+tp1+fn1)>0 else 0
    micro_f1 = 2*micro_prec*micro_rec/(micro_prec+micro_rec) if (micro_prec+micro_rec)>0 else 0

    results["macro_precision"] = round(macro_prec, 4)
    results["macro_recall"] = round(macro_rec, 4)
    results["macro_f1"] = round(macro_f1, 4)
    results["micro_precision"] = round(micro_prec, 4)
    results["micro_recall"] = round(micro_rec, 4)
    results["micro_f1"] = round(micro_f1, 4)

    try:
        auc_ev = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        results["auc_roc"] = auc_ev.evaluate(preds)
    except Exception:
        results["auc_roc"] = 0.0

    # Confusion Matrix
    cm = np.array([[tp0, fp1], [fp0, tp1]])

    for k, v in results.items():
        logger.info("  %s: %.4f", k, v)

    return results, cm


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: GÖRSELLEŞTİRMELER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm):
    labels = ["Beğenmedi (0)", "Beğendi (1)"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="plasma", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white", fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_fig("21_confusion_matrix")


def plot_als_metrics(als_metrics):
    names = list(als_metrics.keys())
    vals = list(als_metrics.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([ n.upper() for n in names], vals, color=PAL[:4], edgecolor="#1e1e2e", width=0.5)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], padding=4, color="#cdd6f4", fontsize=10)
    ax.set_title("ALS Regresyon Metrikleri (RMSE, MAE, MSE, R²)")
    ax.set_ylabel("Değer"); ax.grid(axis="y")
    plt.tight_layout()
    save_fig("22_als_extended_metrics")


def plot_recommendation_metrics(rec_metrics):
    names = list(rec_metrics.keys())
    vals = list(rec_metrics.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, vals, color=PAL[4:4+len(names)], edgecolor="#1e1e2e", width=0.5)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], padding=4, color="#cdd6f4", fontsize=10)
    ax.set_title("Öneri Sistemi Metrikleri (Precision@K, Recall@K, NDCG)")
    ax.set_ylabel("Skor"); ax.set_ylim(0, max(vals)*1.3 if vals else 1)
    ax.grid(axis="y"); plt.tight_layout()
    save_fig("23_recommendation_metrics")


def plot_classification_extended(cls_metrics):
    # Weighted vs Macro vs Micro karşılaştırma
    groups = {
        "Precision": [cls_metrics.get("weightedPrecision",0), cls_metrics.get("macro_precision",0), cls_metrics.get("micro_precision",0)],
        "Recall": [cls_metrics.get("weightedRecall",0), cls_metrics.get("macro_recall",0), cls_metrics.get("micro_recall",0)],
        "F1": [cls_metrics.get("f1",0), cls_metrics.get("macro_f1",0), cls_metrics.get("micro_f1",0)],
    }
    x = np.arange(3)
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, vals) in enumerate(groups.items()):
        bars = ax.bar(x + i*width, vals, width, label=metric, color=PAL[i], edgecolor="#1e1e2e")
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=3, fontsize=8, color="#cdd6f4")
    ax.set_xticks(x + width); ax.set_xticklabels(["Weighted", "Macro", "Micro"])
    ax.set_title("Sınıflandırma Metrikleri (Weighted / Macro / Micro)")
    ax.set_ylabel("Skor"); ax.set_ylim(0, 1.1)
    ax.legend(); ax.grid(axis="y"); plt.tight_layout()
    save_fig("24_classification_extended_metrics")


def plot_all_metrics_summary(als_m, rec_m, cls_m):
    """Tüm metrikleri tek tabloda özetleyen final grafiği."""
    all_data = {}
    all_data.update({f"ALS_{k.upper()}": v for k, v in als_m.items()})
    all_data.update({f"Rec_{k}": v for k, v in rec_m.items()})
    all_data.update({f"Cls_{k}": v for k, v in cls_m.items() if isinstance(v, (int, float))})

    names = list(all_data.keys())
    vals = list(all_data.values())

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = [PAL[i % len(PAL)] for i in range(len(names))]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="#1e1e2e", height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7, color="#cdd6f4")
    ax.set_title("Tüm Değerlendirme Metrikleri — Özet Tablo")
    ax.set_xlabel("Değer"); ax.grid(axis="x"); plt.tight_layout()
    save_fig("25_all_metrics_summary")


# ═══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    spark = create_spark()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # 1. ALS genişletilmiş regresyon metrikleri
    als_metrics, als_model, train, test = compute_als_extended_metrics(spark)

    # 2. Öneri sistemi metrikleri
    rec_metrics = compute_recommendation_metrics(als_model, train, test, k_values=[5, 10])

    # 3. Sınıflandırma genişletilmiş metrikler
    cls_metrics, cm = compute_classification_extended(spark)

    # 4. Görselleştirmeler
    logger.info("Görselleştirmeler oluşturuluyor...")
    plot_als_metrics(als_metrics)
    plot_recommendation_metrics(rec_metrics)
    plot_confusion_matrix(cm)
    plot_classification_extended(cls_metrics)
    plot_all_metrics_summary(als_metrics, rec_metrics, cls_metrics)

    # 5. MLflow'a tüm metrikleri logla
    with mlflow.start_run(run_name="extended_metrics_report"):
        mlflow.log_param("pipeline_stage", "extended_evaluation")
        for k, v in als_metrics.items():
            mlflow.log_metric(f"als_{k}", v)
        for k, v in rec_metrics.items():
            mlflow.log_metric(k, v)
        for k, v in cls_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"cls_{k}", v)
        # Grafikleri artifact olarak logla
        for png in ["21_confusion_matrix","22_als_extended_metrics",
                     "23_recommendation_metrics","24_classification_extended_metrics",
                     "25_all_metrics_summary"]:
            p = os.path.join(PLOT_DIR, f"{png}.png")
            if os.path.exists(p):
                mlflow.log_artifact(p, "extended_metrics_plots")

    # Sonuç özeti
    logger.info("=" * 70)
    logger.info("GENİŞLETİLMİŞ METRİK RAPORU")
    logger.info("─" * 70)
    logger.info("ALS Regresyon:")
    for k, v in als_metrics.items(): logger.info("  %-10s: %.4f", k.upper(), v)
    logger.info("─" * 70)
    logger.info("Öneri Sistemi:")
    for k, v in rec_metrics.items(): logger.info("  %-15s: %.4f", k, v)
    logger.info("─" * 70)
    logger.info("Sınıflandırma:")
    for k, v in cls_metrics.items():
        if isinstance(v, (int, float)): logger.info("  %-25s: %.4f", k, v)
    logger.info("=" * 70)

    spark.stop()

if __name__ == "__main__":
    main()
