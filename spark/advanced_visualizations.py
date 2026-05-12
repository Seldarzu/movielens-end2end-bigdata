# Takım: Arzu & Ayaz
# Görev: Zenginleştirilmiş Görselleştirmeler
#  - ROC Curve (tüm sınıflandırma modelleri aynı grafikte)
#  - Learning Curve (veri boyutu vs performans)
#  - Residual Plot (tahmin hataları dağılımı)
#  - Gelişmiş Confusion Matrix
#  - Tüm modellerin birleşik karşılaştırması

import logging, sys, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, count, avg
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
    RegressionEvaluator, MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

FEATURES_PATH = "/app/delta/ratings_features/full_features"
ENRICHED_PATH = "/app/delta/ratings_features/enriched_ratings"
PLOT_DIR      = "/app/delta/eda"

FEATURE_COLS = [
    "day_of_week","hour_of_day","month","season","is_weekend","day_period",
    "user_avg_rating","user_rating_count","user_rating_deviation",
    "user_rating_variance","user_activity_days",
    "movie_avg_rating","movie_rating_count","movie_age",
    "genre_count","movie_popularity_trend",
    "user_genre_avg_rating","rating_order","time_since_last_rating",
]

plt.rcParams.update({
    "figure.facecolor":"#1e1e2e","axes.facecolor":"#2a2a3e",
    "axes.edgecolor":"#555577","text.color":"#cdd6f4",
    "axes.labelcolor":"#cdd6f4","xtick.color":"#cdd6f4",
    "ytick.color":"#cdd6f4","axes.titlecolor":"#cba6f7",
    "grid.color":"#44445a","grid.alpha":0.4,"font.size":11,
    "axes.titlesize":14,"axes.titleweight":"bold",
})
PAL = ["#cba6f7","#89b4fa","#a6e3a1","#fab387","#f38ba8",
       "#94e2d5","#f9e2af","#eba0ac","#b4befe","#74c7ec"]

def save_fig(n):
    p = os.path.join(PLOT_DIR, f"{n}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Grafik → %s", p)

def create_spark():
    spark = (SparkSession.builder.appName("MovieLens-AdvancedViz")
        .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages","io.delta:delta-core_2.12:2.4.0")
        .config("spark.driver.memory","2g").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(PLOT_DIR, exist_ok=True)
    return spark

def load_data(spark):
    try:
        df = spark.read.format("delta").load(FEATURES_PATH)
    except Exception:
        df = spark.read.format("delta").load(ENRICHED_PATH)
        df = df.select("userId","movieId","rating")

    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 2:
        available = ["userId","movieId"]

    for c in available:
        df = df.withColumn(c, when(col(c).isNull(),0).otherwise(col(c)).cast("double"))

    df = df.withColumn("label_cls", when(col("rating")>=3.5,1.0).otherwise(0.0))
    assembler = VectorAssembler(inputCols=available, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df).select("features","rating","label_cls")
    df = df.sample(False, 0.08, seed=42)
    return df, available


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ROC CURVE — Tüm sınıflandırma modelleri aynı grafikte
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(df):
    logger.info("=== ROC Curve Oluşturuluyor ===")
    train, test = df.randomSplit([0.8,0.2], seed=42)
    train_c = train.withColumnRenamed("label_cls","label")
    test_c = test.withColumnRenamed("label_cls","label")

    models = {
        "Logistic Regression": LogisticRegression(maxIter=20, regParam=0.01),
        "Random Forest": RandomForestClassifier(numTrees=50, maxDepth=8, seed=42),
        "GBT": GBTClassifier(maxIter=20, maxDepth=5, seed=42),
        "Decision Tree": DecisionTreeClassifier(maxDepth=10, seed=42),
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    auc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    for i, (name, model) in enumerate(models.items()):
        logger.info("  %s eğitiliyor...", name)
        fitted = model.fit(train_c)
        preds = fitted.transform(test_c)
        auc = auc_eval.evaluate(preds)

        # ROC noktalarını hesapla
        if hasattr(fitted, "summary") and hasattr(fitted.summary, "roc"):
            roc_df = fitted.summary.roc.toPandas()
            ax.plot(roc_df["FPR"], roc_df["TPR"], color=PAL[i],
                    linewidth=2.5, label=f"{name} (AUC={auc:.4f})")
        else:
            # Manuel ROC noktaları
            thresholds = np.arange(0, 1.05, 0.05)
            fprs, tprs = [], []
            preds_pd = preds.select("label","probability").toPandas()

            # probability kolonundan pozitif sınıf olasılığını çıkar
            try:
                preds_pd["prob_1"] = preds_pd["probability"].apply(lambda x: float(x[1]))
            except Exception:
                preds_pd["prob_1"] = preds_pd["probability"].apply(lambda x: float(x))

            for t in thresholds:
                pred_pos = preds_pd["prob_1"] >= t
                tp = ((pred_pos) & (preds_pd["label"]==1)).sum()
                fp = ((pred_pos) & (preds_pd["label"]==0)).sum()
                fn = ((~pred_pos) & (preds_pd["label"]==1)).sum()
                tn = ((~pred_pos) & (preds_pd["label"]==0)).sum()
                fpr = fp/(fp+tn) if (fp+tn)>0 else 0
                tpr = tp/(tp+fn) if (tp+fn)>0 else 0
                fprs.append(fpr); tprs.append(tpr)

            # Sırala
            sorted_pts = sorted(zip(fprs, tprs))
            fprs = [p[0] for p in sorted_pts]
            tprs = [p[1] for p in sorted_pts]
            ax.plot(fprs, tprs, color=PAL[i], linewidth=2.5,
                    label=f"{name} (AUC={auc:.4f})")

    ax.plot([0,1],[0,1], "--", color="#555577", linewidth=1, label="Rastgele (AUC=0.5)")
    ax.set_title("ROC Curve — Tüm Sınıflandırma Modelleri")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.grid(True)
    plt.tight_layout()
    save_fig("26_roc_curves_all_models")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LEARNING CURVE — Eğitim verisi boyutu vs performans
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve(df):
    logger.info("=== Learning Curve Oluşturuluyor ===")
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    full_train, test = df.randomSplit([0.8,0.2], seed=42)
    test_c = test.withColumnRenamed("label_cls","label")

    cls_results = {f: [] for f in fractions}
    reg_results = {f: [] for f in fractions}

    acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    rmse_eval = RegressionEvaluator(labelCol="label", metricName="rmse")

    for frac in fractions:
        logger.info("  Fraction: %.0f%%", frac*100)
        subset = full_train.sample(False, frac, seed=42) if frac < 1.0 else full_train
        train_c = subset.withColumnRenamed("label_cls","label")

        # Sınıflandırma (RF)
        rf = RandomForestClassifier(numTrees=30, maxDepth=6, seed=42)
        preds = rf.fit(train_c).transform(test_c)
        train_preds = rf.fit(train_c).transform(train_c)
        cls_results[frac] = [
            acc_eval.evaluate(train_preds),
            acc_eval.evaluate(preds),
        ]

        # Regresyon (RF)
        train_r = subset.withColumnRenamed("rating","label")
        test_r = test.withColumnRenamed("rating","label")
        rfr = RandomForestRegressor(numTrees=30, maxDepth=6, seed=42)
        preds_r = rfr.fit(train_r).transform(test_r)
        train_preds_r = rfr.fit(train_r).transform(train_r)
        reg_results[frac] = [
            rmse_eval.evaluate(train_preds_r),
            rmse_eval.evaluate(preds_r),
        ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Learning Curve — Eğitim Verisi Boyutu vs Performans", fontsize=14, color="#cba6f7")

    pcts = [f"{int(f*100)}%" for f in fractions]

    # Sınıflandırma
    train_accs = [cls_results[f][0] for f in fractions]
    test_accs = [cls_results[f][1] for f in fractions]
    axes[0].plot(pcts, train_accs, "o-", color=PAL[2], linewidth=2, markersize=7, label="Train Accuracy")
    axes[0].plot(pcts, test_accs, "s-", color=PAL[4], linewidth=2, markersize=7, label="Test Accuracy")
    axes[0].fill_between(range(len(pcts)), train_accs, test_accs, alpha=0.1, color=PAL[0])
    axes[0].set_title("Sınıflandırma (Random Forest)")
    axes[0].set_xlabel("Eğitim Verisi Oranı"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True)

    # Regresyon
    train_rmses = [reg_results[f][0] for f in fractions]
    test_rmses = [reg_results[f][1] for f in fractions]
    axes[1].plot(pcts, train_rmses, "o-", color=PAL[1], linewidth=2, markersize=7, label="Train RMSE")
    axes[1].plot(pcts, test_rmses, "s-", color=PAL[3], linewidth=2, markersize=7, label="Test RMSE")
    axes[1].fill_between(range(len(pcts)), train_rmses, test_rmses, alpha=0.1, color=PAL[0])
    axes[1].set_title("Regresyon (Random Forest)")
    axes[1].set_xlabel("Eğitim Verisi Oranı"); axes[1].set_ylabel("RMSE")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    save_fig("27_learning_curve")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RESIDUAL PLOT — Tahmin hataları dağılımı
# ═══════════════════════════════════════════════════════════════════════════════

def plot_residuals(df):
    logger.info("=== Residual Plot Oluşturuluyor ===")
    train, test = df.randomSplit([0.8,0.2], seed=42)
    train_r = train.withColumnRenamed("rating","label")
    test_r = test.withColumnRenamed("rating","label")

    models = {
        "Linear Regression": LinearRegression(maxIter=20, regParam=0.01),
        "Random Forest": RandomForestRegressor(numTrees=50, maxDepth=8, seed=42),
        "GBT": GBTRegressor(maxIter=20, maxDepth=5, seed=42),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Residual Plot — Tahmin Hataları Dağılımı", fontsize=14, color="#cba6f7")

    for ax, (name, model), color in zip(axes, models.items(), [PAL[0], PAL[1], PAL[2]]):
        logger.info("  %s eğitiliyor...", name)
        fitted = model.fit(train_r)
        preds = fitted.transform(test_r).select("label","prediction").toPandas()

        residuals = preds["label"] - preds["prediction"]
        predicted = preds["prediction"]

        ax.scatter(predicted, residuals, alpha=0.3, s=8, color=color, edgecolors="none")
        ax.axhline(y=0, color="#f9e2af", linewidth=1.5, linestyle="--")
        ax.set_title(f"{name}\nOrtalama Hata: {residuals.mean():.4f}")
        ax.set_xlabel("Tahmin Değeri")
        ax.set_ylabel("Residual (Gerçek - Tahmin)")
        ax.grid(True)

        # Histogram inset
        inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
        inset.hist(residuals, bins=40, color=color, edgecolor="#1e1e2e", alpha=0.8)
        inset.set_facecolor("#2a2a3e")
        inset.tick_params(colors="#cdd6f4", labelsize=6)
        inset.set_title("Hata Dağılımı", fontsize=7, color="#cdd6f4")

    plt.tight_layout()
    save_fig("28_residual_plots")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GELİŞMİŞ CONFUSION MATRIX — Her model için ayrı
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all_confusion_matrices(df):
    logger.info("=== Gelişmiş Confusion Matrix'ler ===")
    train, test = df.randomSplit([0.8,0.2], seed=42)
    train_c = train.withColumnRenamed("label_cls","label")
    test_c = test.withColumnRenamed("label_cls","label")

    models = {
        "Logistic\nRegression": LogisticRegression(maxIter=20, regParam=0.01),
        "Random\nForest": RandomForestClassifier(numTrees=50, maxDepth=8, seed=42),
        "GBT": GBTClassifier(maxIter=20, maxDepth=5, seed=42),
        "Decision\nTree": DecisionTreeClassifier(maxDepth=10, seed=42),
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Confusion Matrix — Tüm Sınıflandırma Modelleri", fontsize=14, color="#cba6f7")
    labels = ["Beğen-\nmedi", "Beğen-\ndi"]

    for ax, (name, model) in zip(axes, models.items()):
        logger.info("  %s...", name.replace("\n"," "))
        fitted = model.fit(train_c)
        preds = fitted.transform(test_c)

        tp = preds.filter((col("prediction")==1)&(col("label")==1)).count()
        fp = preds.filter((col("prediction")==1)&(col("label")==0)).count()
        fn = preds.filter((col("prediction")==0)&(col("label")==1)).count()
        tn = preds.filter((col("prediction")==0)&(col("label")==0)).count()
        cm = np.array([[tn, fp],[fn, tp]])

        im = ax.imshow(cm, cmap="plasma", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Tahmin", fontsize=9)
        ax.set_ylabel("Gerçek", fontsize=9)

        acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0
        ax.set_title(f"{name}\nAcc={acc:.3f}", fontsize=10)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")

    plt.tight_layout()
    save_fig("29_all_confusion_matrices")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BİRLEŞİK MODEL KARŞILAŞTIRMASI
# ═══════════════════════════════════════════════════════════════════════════════

def plot_unified_comparison(df):
    logger.info("=== Birleşik Model Karşılaştırması ===")
    train, test = df.randomSplit([0.8,0.2], seed=42)

    # Sınıflandırma
    train_c = train.withColumnRenamed("label_cls","label")
    test_c = test.withColumnRenamed("label_cls","label")
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

    cls_models = {
        "LogReg": LogisticRegression(maxIter=20, regParam=0.01),
        "RF_Cls": RandomForestClassifier(numTrees=50, maxDepth=8, seed=42),
        "GBT_Cls": GBTClassifier(maxIter=20, maxDepth=5, seed=42),
        "DT_Cls": DecisionTreeClassifier(maxDepth=10, seed=42),
    }

    # Regresyon
    train_r = train.withColumnRenamed("rating","label")
    test_r = test.withColumnRenamed("rating","label")
    rmse_eval = RegressionEvaluator(labelCol="label", metricName="rmse")
    r2_eval = RegressionEvaluator(labelCol="label", metricName="r2")

    reg_models = {
        "LinReg": LinearRegression(maxIter=20, regParam=0.01),
        "RF_Reg": RandomForestRegressor(numTrees=50, maxDepth=8, seed=42),
        "GBT_Reg": GBTRegressor(maxIter=20, maxDepth=5, seed=42),
        "DT_Reg": DecisionTreeRegressor(maxDepth=10, seed=42),
    }

    cls_data = []
    for name, m in cls_models.items():
        logger.info("  %s...", name)
        p = m.fit(train_c).transform(test_c)
        cls_data.append({"Model": name, "Accuracy": acc_eval.evaluate(p), "F1": f1_eval.evaluate(p)})

    reg_data = []
    for name, m in reg_models.items():
        logger.info("  %s...", name)
        p = m.fit(train_r).transform(test_r)
        reg_data.append({"Model": name, "RMSE": rmse_eval.evaluate(p), "R²": r2_eval.evaluate(p)})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tüm Modeller — Birleşik Karşılaştırma", fontsize=16, color="#cba6f7")

    cls_df = pd.DataFrame(cls_data)
    reg_df = pd.DataFrame(reg_data)

    # Accuracy
    b = axes[0,0].bar(cls_df["Model"], cls_df["Accuracy"], color=PAL[0], edgecolor="#1e1e2e")
    axes[0,0].bar_label(b, labels=[f"{v:.4f}" for v in cls_df["Accuracy"]], padding=3, fontsize=9, color="#cdd6f4")
    axes[0,0].set_title("Accuracy"); axes[0,0].set_ylim(0,1.05); axes[0,0].grid(axis="y")

    # F1
    b = axes[0,1].bar(cls_df["Model"], cls_df["F1"], color=PAL[1], edgecolor="#1e1e2e")
    axes[0,1].bar_label(b, labels=[f"{v:.4f}" for v in cls_df["F1"]], padding=3, fontsize=9, color="#cdd6f4")
    axes[0,1].set_title("F1-Score"); axes[0,1].set_ylim(0,1.05); axes[0,1].grid(axis="y")

    # RMSE
    b = axes[1,0].bar(reg_df["Model"], reg_df["RMSE"], color=PAL[4], edgecolor="#1e1e2e")
    axes[1,0].bar_label(b, labels=[f"{v:.4f}" for v in reg_df["RMSE"]], padding=3, fontsize=9, color="#cdd6f4")
    axes[1,0].set_title("RMSE (düşük=iyi)"); axes[1,0].grid(axis="y")

    # R²
    b = axes[1,1].bar(reg_df["Model"], reg_df["R²"], color=PAL[2], edgecolor="#1e1e2e")
    axes[1,1].bar_label(b, labels=[f"{v:.4f}" for v in reg_df["R²"]], padding=3, fontsize=9, color="#cdd6f4")
    axes[1,1].set_title("R² (yüksek=iyi)"); axes[1,1].grid(axis="y")

    plt.tight_layout()
    save_fig("30_unified_model_comparison")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    spark = create_spark()
    df, features = load_data(spark)
    logger.info("Veri yüklendi: %d satır, %d feature", df.count(), len(features))

    plot_roc_curves(df)
    plot_learning_curve(df)
    plot_residuals(df)
    plot_all_confusion_matrices(df)
    plot_unified_comparison(df)

    logger.info("=" * 60)
    logger.info("Tüm gelişmiş görselleştirmeler tamamlandı!")
    logger.info("Çıktılar → %s", PLOT_DIR)
    logger.info("  26_roc_curves_all_models.png")
    logger.info("  27_learning_curve.png")
    logger.info("  28_residual_plots.png")
    logger.info("  29_all_confusion_matrices.png")
    logger.info("  30_unified_model_comparison.png")
    logger.info("=" * 60)
    spark.stop()

if __name__ == "__main__":
    main()
