# Takım Üyesi: Arzu & Ayaz
# Görev: Sınıflandırma ve Regresyon Modelleri
#   - 4 Sınıflandırma: LogReg, RF, GBT, DecisionTree
#   - 4 Regresyon: Linear, RF, GBT, DecisionTree
#   - MLflow loglama + model kaydetme
#   - Karşılaştırma tablosu ve görselleştirmeler

import logging, sys, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier,
    GBTClassifier, DecisionTreeClassifier,
)
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor,
    GBTRegressor, DecisionTreeRegressor,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
    RegressionEvaluator,
)
import mlflow
import mlflow.spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

FEATURES_PATH = "/app/delta/ratings_features/full_features"
PLOT_DIR      = "/app/delta/eda"
MODEL_DIR     = "/app/delta/models"
MLFLOW_URI    = "http://mlflow:5000"
EXPERIMENT    = "movielens-classification-regression"

FEATURE_COLS = [
    "day_of_week", "hour_of_day", "month", "season", "is_weekend", "day_period",
    "user_avg_rating", "user_rating_count",
    "user_rating_variance", "user_activity_days",
    "movie_avg_rating", "movie_rating_count", "movie_age",
    "genre_count", "movie_popularity_trend",
    "user_genre_avg_rating", "rating_order", "time_since_last_rating",
]

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#2a2a3e",
    "axes.edgecolor": "#555577", "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4", "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4", "axes.titlecolor": "#cba6f7",
    "grid.color": "#44445a", "grid.alpha": 0.4, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
})
PALETTE = ["#cba6f7", "#89b4fa", "#a6e3a1", "#fab387", "#f38ba8",
           "#94e2d5", "#f9e2af", "#eba0ac", "#b4befe", "#74c7ec"]


def save_fig(name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafik → %s", path)


def create_spark():
    spark = (
        SparkSession.builder
        .appName("MovieLens-Classification-Regression")
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    return spark


def load_and_prepare(spark):
    logger.info("Veri yükleniyor: %s", FEATURES_PATH)
    try:
        df = spark.read.format("delta").load(FEATURES_PATH)
    except Exception:
        logger.warning("Full features bulunamadı, ham veri deneniyor...")
        df = spark.read.format("delta").load("/app/delta/ratings")

    available = [c for c in FEATURE_COLS if c in df.columns]
    logger.info("Kullanılabilir feature: %d / %d", len(available), len(FEATURE_COLS))

    for c in available:
        df = df.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c)).cast("double"))

    # Binary label (sınıflandırma için)
    df = df.withColumn("label_cls", when(col("rating") >= 3.5, 1.0).otherwise(0.0))

    assembler = VectorAssembler(inputCols=available, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df).select("features", "rating", "label_cls")

    # Örneklem (25M çok büyük, %10 al)
    df = df.sample(False, 0.10, seed=42)
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()

    logger.info("Train: %d | Test: %d", train.count(), test.count())
    return train, test, available


# ═══════════════════════════════════════════════════════════════════════════════
# SINIFLANDIRMA MODELLERİ
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifiers(train, test):
    logger.info("=" * 60)
    logger.info("SINIFLANDIRMA MODELLERİ EĞİTİLİYOR")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    train_cls = train.withColumnRenamed("label_cls", "label")
    test_cls = test.withColumnRenamed("label_cls", "label")

    models = {
        "Logistic_Regression": LogisticRegression(
            featuresCol="features", labelCol="label", maxIter=20, regParam=0.01),
        "Random_Forest_Classifier": RandomForestClassifier(
            featuresCol="features", labelCol="label", numTrees=50, maxDepth=8, seed=42),
        "GBT_Classifier": GBTClassifier(
            featuresCol="features", labelCol="label", maxIter=20, maxDepth=5, seed=42),
        "Decision_Tree_Classifier": DecisionTreeClassifier(
            featuresCol="features", labelCol="label", maxDepth=10, seed=42),
    }

    eval_acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    eval_prec = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    eval_rec = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")
    eval_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    results = []
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        logger.info("── %s eğitiliyor...", name)
        t0 = time.time()

        with mlflow.start_run(run_name=f"cls_{name}"):
            fitted = model.fit(train_cls)
            preds       = fitted.transform(test_cls)
            preds_train = fitted.transform(train_cls)
            elapsed = time.time() - t0

            # Test metrikleri
            acc  = eval_acc.evaluate(preds)
            f1   = eval_f1.evaluate(preds)
            prec = eval_prec.evaluate(preds)
            rec  = eval_rec.evaluate(preds)
            # Train metrikleri (overfitting kontrolü)
            train_acc = eval_acc.evaluate(preds_train)
            train_f1  = eval_f1.evaluate(preds_train)

            try:
                auc = eval_auc.evaluate(preds)
            except Exception:
                auc = 0.0

            # Overfitting / underfitting teşhisi
            gap = round(train_f1 - f1, 4)
            if gap > 0.05:
                diagnosis = "OVERFIT"
            elif f1 < 0.5:
                diagnosis = "UNDERFIT"
            else:
                diagnosis = "OK"

            mlflow.log_param("model_type", "classification")
            mlflow.log_param("algorithm", name)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("auc_roc", auc)
            mlflow.log_metric("overfit_gap_f1", gap)
            mlflow.log_metric("train_time_sec", elapsed)

            model_path = os.path.join(MODEL_DIR, f"cls_{name}")
            fitted.write().overwrite().save(model_path)
            mlflow.spark.log_model(fitted, artifact_path=f"cls_{name}",
                                   registered_model_name=f"movielens-{name}")

            logger.info("  [%s] Train_F1=%.4f | Test_F1=%.4f | Gap=%.4f | Acc=%.4f | AUC=%.4f | %.1fs",
                        diagnosis, train_f1, f1, gap, acc, auc, elapsed)

        results.append({
            "Model": name,
            "Train_F1": round(train_f1, 4), "Test_F1": round(f1, 4),
            "Train_Acc": round(train_acc, 4), "Accuracy": round(acc, 4),
            "F1": round(f1, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "AUC-ROC": round(auc, 4),
            "Overfit_Gap": gap, "Diagnosis": diagnosis,
            "Süre (s)": round(elapsed, 1),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = name

    logger.info("En iyi sınıflandırma: %s (F1=%.4f)", best_model, best_f1)
    return pd.DataFrame(results), best_model


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESYON MODELLERİ
# ═══════════════════════════════════════════════════════════════════════════════

def train_regressors(train, test):
    logger.info("=" * 60)
    logger.info("REGRESYON MODELLERİ EĞİTİLİYOR")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    train_reg = train.withColumnRenamed("rating", "label")
    test_reg = test.withColumnRenamed("rating", "label")

    models = {
        "Linear_Regression": LinearRegression(
            featuresCol="features", labelCol="label", maxIter=20, regParam=0.01),
        "Decision_Tree_Regressor": DecisionTreeRegressor(
            featuresCol="features", labelCol="label", maxDepth=10, seed=42),
        "Random_Forest_Regressor": RandomForestRegressor(
            featuresCol="features", labelCol="label", numTrees=50, maxDepth=8, seed=42),
        "GBT_Regressor": GBTRegressor(
            featuresCol="features", labelCol="label", maxIter=20, maxDepth=5, seed=42),
    }

    eval_rmse = RegressionEvaluator(labelCol="label", metricName="rmse")
    eval_mae = RegressionEvaluator(labelCol="label", metricName="mae")
    eval_r2 = RegressionEvaluator(labelCol="label", metricName="r2")
    eval_mse = RegressionEvaluator(labelCol="label", metricName="mse")

    results = []
    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        logger.info("── %s eğitiliyor...", name)
        t0 = time.time()

        with mlflow.start_run(run_name=f"reg_{name}"):
            fitted = model.fit(train_reg)
            preds       = fitted.transform(test_reg)
            preds_train = fitted.transform(train_reg)
            elapsed = time.time() - t0

            rmse       = eval_rmse.evaluate(preds)
            mae        = eval_mae.evaluate(preds)
            r2         = eval_r2.evaluate(preds)
            mse        = eval_mse.evaluate(preds)
            train_rmse = eval_rmse.evaluate(preds_train)
            train_r2   = eval_r2.evaluate(preds_train)

            gap = round(rmse - train_rmse, 4)
            if gap > 0.1:
                diagnosis = "OVERFIT"
            elif r2 < 0.1:
                diagnosis = "UNDERFIT"
            else:
                diagnosis = "OK"

            mlflow.log_param("model_type", "regression")
            mlflow.log_param("algorithm", name)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("overfit_gap_rmse", gap)
            mlflow.log_metric("train_time_sec", elapsed)

            model_path = os.path.join(MODEL_DIR, f"reg_{name}")
            fitted.write().overwrite().save(model_path)
            mlflow.spark.log_model(fitted, artifact_path=f"reg_{name}",
                                   registered_model_name=f"movielens-{name}")

            logger.info("  [%s] Train_RMSE=%.4f | Test_RMSE=%.4f | Gap=%.4f | R²=%.4f | %.1fs",
                        diagnosis, train_rmse, rmse, gap, r2, elapsed)

        results.append({
            "Model": name,
            "Train_RMSE": round(train_rmse, 4), "RMSE": round(rmse, 4),
            "Train_R2": round(train_r2, 4), "MAE": round(mae, 4),
            "R²": round(r2, 4), "MSE": round(mse, 4),
            "Overfit_Gap": gap, "Diagnosis": diagnosis,
            "Süre (s)": round(elapsed, 1),
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = name

    logger.info("En iyi regresyon: %s (RMSE=%.4f)", best_model, best_rmse)
    return pd.DataFrame(results), best_model


# ═══════════════════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRMELER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_classification_comparison(cls_df):
    """Sınıflandırma modellerini 4 metrikte karşılaştıran bar chart."""
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sınıflandırma Modelleri Karşılaştırması", fontsize=16, color="#cba6f7")

    for ax, metric, color in zip(axes.flat, metrics, PALETTE[:4]):
        bars = ax.bar(cls_df["Model"].str.replace("_", "\n"), cls_df[metric],
                      color=color, edgecolor="#1e1e2e", width=0.55)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in cls_df[metric]],
                     padding=4, color="#cdd6f4", fontsize=9)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y")

    plt.tight_layout()
    save_fig("16_classification_comparison")


def plot_regression_comparison(reg_df):
    """Regresyon modellerini RMSE, MAE, R² ile karşılaştıran bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Regresyon Modelleri Karşılaştırması", fontsize=16, color="#cba6f7")

    for ax, metric, color in zip(axes, ["RMSE", "MAE", "R²"], [PALETTE[4], PALETTE[5], PALETTE[2]]):
        vals = reg_df[metric]
        bars = ax.bar(reg_df["Model"].str.replace("_", "\n"), vals,
                      color=color, edgecolor="#1e1e2e", width=0.55)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals],
                     padding=4, color="#cdd6f4", fontsize=9)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(axis="y")

    plt.tight_layout()
    save_fig("17_regression_comparison")


def plot_auc_comparison(cls_df):
    """AUC-ROC karşılaştırma grafiği."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cls_df["Model"].str.replace("_", " "), cls_df["AUC-ROC"],
                   color=PALETTE[6], edgecolor="#1e1e2e", height=0.5)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in cls_df["AUC-ROC"]],
                 padding=4, color="#cdd6f4", fontsize=10)
    ax.set_title("AUC-ROC Karşılaştırması")
    ax.set_xlabel("AUC-ROC")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x")
    plt.tight_layout()
    save_fig("18_auc_roc_comparison")


def plot_training_time(cls_df, reg_df):
    """Tüm modellerin eğitim süresi karşılaştırması."""
    all_models = pd.concat([
        cls_df[["Model", "Süre (s)"]].assign(Tür="Sınıflandırma"),
        reg_df[["Model", "Süre (s)"]].assign(Tür="Regresyon"),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PALETTE[0] if t == "Sınıflandırma" else PALETTE[1] for t in all_models["Tür"]]
    bars = ax.barh(all_models["Model"].str.replace("_", " "),
                   all_models["Süre (s)"], color=colors, edgecolor="#1e1e2e", height=0.5)
    ax.bar_label(bars, labels=[f"{v:.1f}s" for v in all_models["Süre (s)"]],
                 padding=4, color="#cdd6f4", fontsize=9)
    ax.set_title("Model Eğitim Süreleri")
    ax.set_xlabel("Süre (saniye)")
    ax.grid(axis="x")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE[0], label="Sınıflandırma"),
                       Patch(facecolor=PALETTE[1], label="Regresyon")]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    save_fig("19_training_time_comparison")


def plot_overfit_check(cls_df, reg_df):
    """Train vs Test metrik karşılaştırması — overfitting/underfitting görselleştirme."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Overfitting / Underfitting Kontrolü", fontsize=16, color="#cba6f7")

    x = np.arange(len(cls_df))
    w = 0.35
    axes[0].bar(x - w/2, cls_df["Train_F1"], width=w, label="Train F1", color=PALETTE[0], edgecolor="#1e1e2e")
    axes[0].bar(x + w/2, cls_df["Test_F1"],  width=w, label="Test F1",  color=PALETTE[1], edgecolor="#1e1e2e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cls_df["Model"].str.replace("_", "\n"), fontsize=8)
    axes[0].set_title("Sınıflandırma: Train F1 vs Test F1")
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(axis="y")
    for i, (tr, te, diag) in enumerate(zip(cls_df["Train_F1"], cls_df["Test_F1"], cls_df["Diagnosis"])):
        color = "#f38ba8" if diag == "OVERFIT" else ("#fab387" if diag == "UNDERFIT" else "#a6e3a1")
        axes[0].text(i, max(tr, te) + 0.03, diag, ha="center", fontsize=7, color=color, fontweight="bold")

    x = np.arange(len(reg_df))
    axes[1].bar(x - w/2, reg_df["Train_RMSE"], width=w, label="Train RMSE", color=PALETTE[4], edgecolor="#1e1e2e")
    axes[1].bar(x + w/2, reg_df["RMSE"],       width=w, label="Test RMSE",  color=PALETTE[5], edgecolor="#1e1e2e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(reg_df["Model"].str.replace("_", "\n"), fontsize=8)
    axes[1].set_title("Regresyon: Train RMSE vs Test RMSE")
    axes[1].legend()
    axes[1].grid(axis="y")
    for i, (tr, te, diag) in enumerate(zip(reg_df["Train_RMSE"], reg_df["RMSE"], reg_df["Diagnosis"])):
        color = "#f38ba8" if diag == "OVERFIT" else ("#fab387" if diag == "UNDERFIT" else "#a6e3a1")
        axes[1].text(i, max(tr, te) + 0.01, diag, ha="center", fontsize=7, color=color, fontweight="bold")

    plt.tight_layout()
    save_fig("31_overfit_check")


def plot_combined_summary(cls_df, reg_df, best_cls, best_reg):
    """En iyi modelleri özetleyen final grafiği."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("En İyi Model Özeti", fontsize=16, color="#cba6f7")

    # Sol: En iyi sınıflandırma
    best_cls_row = cls_df[cls_df["Model"] == best_cls].iloc[0]
    metrics_cls = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC"]
    vals_cls = [best_cls_row[m] for m in metrics_cls]
    bars1 = axes[0].bar(metrics_cls, vals_cls, color=PALETTE[0], edgecolor="#1e1e2e", width=0.5)
    axes[0].bar_label(bars1, labels=[f"{v:.4f}" for v in vals_cls],
                      padding=4, color="#cdd6f4", fontsize=10)
    axes[0].set_title(f"En İyi Sınıflandırma: {best_cls}")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis="y")

    # Sağ: En iyi regresyon
    best_reg_row = reg_df[reg_df["Model"] == best_reg].iloc[0]
    metrics_reg = ["RMSE", "MAE", "R²"]
    vals_reg = [best_reg_row[m] for m in metrics_reg]
    bars2 = axes[1].bar(metrics_reg, vals_reg, color=PALETTE[2], edgecolor="#1e1e2e", width=0.5)
    axes[1].bar_label(bars2, labels=[f"{v:.4f}" for v in vals_reg],
                      padding=4, color="#cdd6f4", fontsize=10)
    axes[1].set_title(f"En İyi Regresyon: {best_reg}")
    axes[1].grid(axis="y")

    plt.tight_layout()
    save_fig("20_best_models_summary")


# ═══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    spark = create_spark()
    train, test, feature_names = load_and_prepare(spark)

    # Sınıflandırma
    cls_df, best_cls = train_classifiers(train, test)

    # Regresyon
    reg_df, best_reg = train_regressors(train, test)

    # Görselleştirmeler
    logger.info("Görselleştirmeler oluşturuluyor...")
    plot_classification_comparison(cls_df)
    plot_regression_comparison(reg_df)
    plot_auc_comparison(cls_df)
    plot_training_time(cls_df, reg_df)
    plot_overfit_check(cls_df, reg_df)
    plot_combined_summary(cls_df, reg_df, best_cls, best_reg)

    # Sonuç tabloları
    logger.info("=" * 70)
    logger.info("SINIFLANDIRMA SONUÇLARI")
    logger.info("=" * 70)
    for _, r in cls_df.iterrows():
        logger.info("  %-30s [%s] TrainF1=%.4f TestF1=%.4f Gap=%.4f Acc=%.4f AUC=%.4f",
                    r["Model"], r["Diagnosis"], r["Train_F1"], r["Test_F1"], r["Overfit_Gap"], r["Accuracy"], r["AUC-ROC"])
    logger.info("  ★ En İyi: %s", best_cls)

    logger.info("=" * 70)
    logger.info("REGRESYON SONUÇLARI")
    logger.info("=" * 70)
    for _, r in reg_df.iterrows():
        logger.info("  %-30s [%s] TrainRMSE=%.4f TestRMSE=%.4f Gap=%.4f R²=%.4f",
                    r["Model"], r["Diagnosis"], r["Train_RMSE"], r["RMSE"], r["Overfit_Gap"], r["R²"])
    logger.info("  ★ En İyi: %s", best_reg)
    logger.info("=" * 70)

    # MLflow'a sonuç tablolarını da logla
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="final_summary"):
        mlflow.log_param("best_classifier", best_cls)
        mlflow.log_param("best_regressor", best_reg)
        best_cls_row = cls_df[cls_df["Model"] == best_cls].iloc[0]
        best_reg_row = reg_df[reg_df["Model"] == best_reg].iloc[0]
        mlflow.log_metric("best_cls_f1", best_cls_row["F1"])
        mlflow.log_metric("best_cls_accuracy", best_cls_row["Accuracy"])
        mlflow.log_metric("best_reg_rmse", best_reg_row["RMSE"])
        mlflow.log_metric("best_reg_r2", best_reg_row["R²"])

        # Grafikleri artifact olarak logla
        for png in ["16_classification_comparison", "17_regression_comparison",
                     "18_auc_roc_comparison", "19_training_time_comparison",
                     "31_overfit_check", "20_best_models_summary"]:
            path = os.path.join(PLOT_DIR, f"{png}.png")
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="comparison_plots")

    logger.info("Tüm modeller eğitildi, kaydedildi ve görselleştirildi.")
    spark.stop()


if __name__ == "__main__":
    main()
