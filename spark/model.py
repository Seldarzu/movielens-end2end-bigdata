# Takım Üyesi: Arzu (Kıdemli Mühendis)
# Görev: ALS tabanlı film öneri modeli — Grid Search, Cross Validation,
#         Baseline karşılaştırması ve MLflow loglama ile kapsamlı deney süreci.

import logging
import sys
import os
import time
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg, lit, count
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import mlflow
import mlflow.spark

# ─── Loglama ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
ENRICHED_PATH   = "/app/delta/ratings_features/enriched_ratings"
BEST_MODEL_PATH = "/app/delta/als_model_best"
PLOT_DIR        = "/app/delta/eda"
MLFLOW_URI      = "http://mlflow:5000"
EXPERIMENT_NAME = "movielens-als-experiments"

RANK_LIST      = [5, 10, 20]
MAX_ITER_LIST  = [5, 10, 15]
REG_PARAM_LIST = [0.01, 0.1, 0.5]

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


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("MovieLens-Model-Experiments")
        .master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(PLOT_DIR, exist_ok=True)
    logger.info("Spark oturumu hazır.")
    return spark


def save_fig(name: str) -> None:
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafik kaydedildi → %s", path)


def load_data(spark: SparkSession):
    df = (
        spark.read.format("delta").load(ENRICHED_PATH)
        .select("userId", "movieId", "rating")
        .dropna()
    )
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    logger.info("Veri yüklendi. Train: %d | Test: %d", train_df.count(), test_df.count())
    return train_df, test_df


def make_evaluator():
    return RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )


# ─── BÖLÜM 1: Baseline Modeller ───────────────────────────────────────────────

def evaluate_baselines(train_df: DataFrame, test_df: DataFrame) -> dict:
    logger.info("=== Baseline Modeller ===")
    evaluator = make_evaluator()

    global_mean = train_df.select(avg("rating")).first()[0]
    test_gm = test_df.withColumn("prediction", lit(float(global_mean)))
    rmse_gm = evaluator.evaluate(test_gm)
    logger.info("Global Mean RMSE: %.4f", rmse_gm)

    user_means = train_df.groupBy("userId").agg(avg("rating").alias("user_mean"))
    test_um = (test_df.join(user_means, "userId", "left")
               .withColumn("prediction", col("user_mean"))
               .filter(col("prediction").isNotNull()))
    rmse_um = evaluator.evaluate(test_um)
    logger.info("User Mean RMSE: %.4f", rmse_um)

    item_means = train_df.groupBy("movieId").agg(avg("rating").alias("item_mean"))
    test_im = (test_df.join(item_means, "movieId", "left")
               .withColumn("prediction", col("item_mean"))
               .filter(col("prediction").isNotNull()))
    rmse_im = evaluator.evaluate(test_im)
    logger.info("Item Mean RMSE: %.4f", rmse_im)

    return {"global_mean": rmse_gm, "user_mean": rmse_um, "item_mean": rmse_im}


def log_baselines_to_mlflow(baselines: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    for name, rmse in baselines.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("baseline_strategy", name)
            mlflow.log_metric("rmse", rmse)


# ─── BÖLÜM 2: Grid Search ─────────────────────────────────────────────────────

def run_grid_search(train_df: DataFrame, test_df: DataFrame) -> pd.DataFrame:
    n_combos = len(RANK_LIST) * len(MAX_ITER_LIST) * len(REG_PARAM_LIST)
    logger.info("=== Grid Search başlıyor (%d kombinasyon) ===", n_combos)

    evaluator = make_evaluator()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for i, (rank, max_iter, reg) in enumerate(
        itertools.product(RANK_LIST, MAX_ITER_LIST, REG_PARAM_LIST), 1
    ):
        run_name = f"gs_r{rank}_i{max_iter}_reg{reg}"
        logger.info("[%d/%d] %s", i, n_combos, run_name)
        t0 = time.time()

        als = ALS(
            rank=rank, maxIter=max_iter, regParam=reg,
            userCol="userId", itemCol="movieId", ratingCol="rating",
            coldStartStrategy="drop", seed=42,
        )
        with mlflow.start_run(run_name=run_name):
            model   = als.fit(train_df)
            preds   = model.transform(test_df)
            rmse    = evaluator.evaluate(preds)
            elapsed = time.time() - t0

            mlflow.log_param("model_type", "ALS")
            mlflow.log_param("rank",       rank)
            mlflow.log_param("maxIter",    max_iter)
            mlflow.log_param("regParam",   reg)
            mlflow.log_metric("rmse",      rmse)
            mlflow.log_metric("train_time_sec", elapsed)

        results.append({"rank": rank, "maxIter": max_iter, "regParam": reg,
                         "rmse": rmse, "train_time": elapsed})
        logger.info("  RMSE=%.4f | %.1fs", rmse, elapsed)

    return pd.DataFrame(results)


# ─── BÖLÜM 3: Ablation Study ──────────────────────────────────────────────────

def run_ablation_study(train_df: DataFrame, test_df: DataFrame,
                       best_params: dict) -> pd.DataFrame:
    logger.info("=== Ablation Study başlıyor ===")

    evaluator = make_evaluator()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    configs = []
    for rank in RANK_LIST:
        configs.append({"vary": "rank", "rank": rank,
                        "maxIter": best_params["maxIter"], "regParam": best_params["regParam"]})
    for mi in MAX_ITER_LIST:
        configs.append({"vary": "maxIter", "rank": best_params["rank"],
                        "maxIter": mi, "regParam": best_params["regParam"]})
    for reg in REG_PARAM_LIST:
        configs.append({"vary": "regParam", "rank": best_params["rank"],
                        "maxIter": best_params["maxIter"], "regParam": reg})

    results = []
    for cfg in configs:
        run_name = f"ablation_{cfg['vary']}_r{cfg['rank']}_i{cfg['maxIter']}_reg{cfg['regParam']}"
        t0 = time.time()
        als = ALS(
            rank=cfg["rank"], maxIter=cfg["maxIter"], regParam=cfg["regParam"],
            userCol="userId", itemCol="movieId", ratingCol="rating",
            coldStartStrategy="drop", seed=42,
        )
        with mlflow.start_run(run_name=run_name):
            model   = als.fit(train_df)
            preds   = model.transform(test_df)
            rmse    = evaluator.evaluate(preds)
            elapsed = time.time() - t0

            mlflow.log_param("model_type",  "ALS_ablation")
            mlflow.log_param("vary_param",  cfg["vary"])
            mlflow.log_param("rank",        cfg["rank"])
            mlflow.log_param("maxIter",     cfg["maxIter"])
            mlflow.log_param("regParam",    cfg["regParam"])
            mlflow.log_metric("rmse",       rmse)
            mlflow.log_metric("train_time_sec", elapsed)

        results.append({**cfg, "rmse": rmse, "train_time": elapsed})
        logger.info("  %s → RMSE=%.4f", run_name, rmse)

    return pd.DataFrame(results)


# ─── BÖLÜM 4: CrossValidator ──────────────────────────────────────────────────

def run_cross_validation(train_df: DataFrame, test_df: DataFrame,
                         best_params: dict) -> float:
    logger.info("=== Cross Validation (3-fold) başlıyor ===")

    evaluator = make_evaluator()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    als = ALS(
        userCol="userId", itemCol="movieId", ratingCol="rating",
        coldStartStrategy="drop", seed=42,
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank,     [best_params["rank"]])
        .addGrid(als.maxIter,  [best_params["maxIter"]])
        .addGrid(als.regParam, [best_params["regParam"]])
        .build()
    )
    cv = CrossValidator(
        estimator=als, estimatorParamMaps=param_grid,
        evaluator=evaluator, numFolds=3, seed=42, parallelism=1,
    )

    t0 = time.time()
    cv_model   = cv.fit(train_df)
    elapsed    = time.time() - t0
    cv_rmse    = min(cv_model.avgMetrics)
    best_model = cv_model.bestModel

    test_preds = best_model.transform(test_df)
    test_rmse  = evaluator.evaluate(test_preds)

    with mlflow.start_run(run_name="cross_validation_best"):
        mlflow.log_param("model_type", "ALS_CrossValidation")
        mlflow.log_param("num_folds",  3)
        mlflow.log_param("rank",       best_params["rank"])
        mlflow.log_param("maxIter",    best_params["maxIter"])
        mlflow.log_param("regParam",   best_params["regParam"])
        mlflow.log_metric("cv_avg_rmse", cv_rmse)
        mlflow.log_metric("test_rmse",   test_rmse)
        mlflow.log_metric("train_time_sec", elapsed)
        mlflow.spark.log_model(best_model, artifact_path="als_cv_model",
                               registered_model_name="movielens-als-cv")

    best_model.write().overwrite().save(BEST_MODEL_PATH)
    logger.info("CV RMSE: %.4f | Test RMSE: %.4f | %.1fs", cv_rmse, test_rmse, elapsed)
    logger.info("En iyi model kaydedildi → %s", BEST_MODEL_PATH)
    return test_rmse


# ─── BÖLÜM 5: Görselleştirmeler ───────────────────────────────────────────────

def plot_grid_search_heatmaps(gs_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(MAX_ITER_LIST), figsize=(16, 5))
    fig.suptitle("Grid Search RMSE Isı Haritası (rank × regParam)", y=1.02)

    for ax, mi in zip(axes, MAX_ITER_LIST):
        sub = gs_df[gs_df["maxIter"] == mi].pivot(
            index="rank", columns="regParam", values="rmse"
        )
        im = ax.imshow(sub.values, cmap="plasma_r", aspect="auto")
        ax.set_xticks(range(len(REG_PARAM_LIST)))
        ax.set_xticklabels([str(r) for r in REG_PARAM_LIST])
        ax.set_yticks(range(len(RANK_LIST)))
        ax.set_yticklabels([str(r) for r in RANK_LIST])
        ax.set_title(f"maxIter={mi}")
        ax.set_xlabel("regParam")
        ax.set_ylabel("rank")
        for (row_i, col_i), val in np.ndenumerate(sub.values):
            ax.text(col_i, row_i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color="white")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_fig("09_gridsearch_heatmap")


def plot_grid_search_top10(gs_df: pd.DataFrame) -> None:
    top10 = gs_df.nsmallest(10, "rmse").reset_index(drop=True)
    top10["label"] = top10.apply(
        lambda r: f"r={int(r['rank'])}\ni={int(r['maxIter'])}\nreg={r['regParam']}", axis=1
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(top10)), top10["rmse"], color=PALETTE[1], edgecolor="#1e1e2e")
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in top10["rmse"]],
                 padding=3, color="#cdd6f4", fontsize=9)
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10["label"], fontsize=8)
    ax.set_title("Grid Search — En İyi 10 Kombinasyon")
    ax.set_ylabel("RMSE")
    ax.set_ylim(top10["rmse"].min() - 0.02, top10["rmse"].max() + 0.05)
    ax.grid(axis="y")
    plt.tight_layout()
    save_fig("10_gridsearch_top10")


def plot_ablation_study(ablation_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Ablation Study — Parametre Etkisi")

    param_specs = [("rank", "Rank"), ("maxIter", "maxIter"), ("regParam", "regParam")]
    for ax, (param, label), color in zip(axes, param_specs, [PALETTE[0], PALETTE[2], PALETTE[3]]):
        sub = ablation_df[ablation_df["vary"] == param].sort_values(param)
        ax.plot(sub[param].astype(str), sub["rmse"], marker="o",
                color=color, linewidth=2.5, markersize=8)
        for x, y in enumerate(sub["rmse"]):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color="#cdd6f4")
        ax.set_title(f"{label} Etkisi")
        ax.set_xlabel(label)
        ax.set_ylabel("RMSE")
        ax.grid(True)

    plt.tight_layout()
    save_fig("11_ablation_study")


def plot_model_comparison(baselines: dict, best_gs_rmse: float, cv_rmse: float) -> None:
    models = {
        "Baseline\n(Global Mean)": baselines["global_mean"],
        "Baseline\n(User Mean)":   baselines["user_mean"],
        "Baseline\n(Item Mean)":   baselines["item_mean"],
        "ALS\n(Best Grid)":        best_gs_rmse,
        "ALS\n(Cross-Val)":        cv_rmse,
    }
    labels = list(models.keys())
    values = list(models.values())
    colors = [PALETTE[4]] * 3 + [PALETTE[2], PALETTE[0]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1e1e2e", width=0.55)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values],
                 padding=4, color="#cdd6f4", fontsize=10)
    ax.set_title("Model Karşılaştırması — RMSE")
    ax.set_ylabel("RMSE (düşük = iyi)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.axhline(y=cv_rmse, linestyle="--", color=PALETTE[6],
               linewidth=1.5, label=f"CV RMSE: {cv_rmse:.4f}")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    save_fig("12_model_comparison")


def plot_rmse_vs_train_time(gs_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for rank, color in zip(RANK_LIST, PALETTE):
        sub = gs_df[gs_df["rank"] == rank]
        ax.scatter(sub["train_time"], sub["rmse"], color=color,
                   s=80, label=f"rank={rank}", zorder=3)
    ax.set_title("RMSE vs Eğitim Süresi (Grid Search)")
    ax.set_xlabel("Eğitim Süresi (saniye)")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_fig("13_rmse_vs_train_time")


def plot_rank_regparam_lines(gs_df: pd.DataFrame) -> None:
    best_per = gs_df.groupby(["rank", "regParam"])["rmse"].min().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for rank, color in zip(RANK_LIST, PALETTE):
        sub = best_per[best_per["rank"] == rank].sort_values("regParam")
        ax.plot(sub["regParam"].astype(str), sub["rmse"], marker="o",
                color=color, linewidth=2, markersize=7, label=f"rank={rank}")
    ax.set_title("regParam → RMSE (Rank'a Göre)")
    ax.set_xlabel("regParam")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_fig("14_rank_regparam_lines")


# ─── ANA FONKSİYON ────────────────────────────────────────────────────────────

def main():
    spark = create_spark_session()
    train_df, test_df = load_data(spark)

    # 1. Baseline
    baselines = evaluate_baselines(train_df, test_df)
    log_baselines_to_mlflow(baselines)

    # 2. Grid Search
    gs_df = run_grid_search(train_df, test_df)
    best_row = gs_df.loc[gs_df["rmse"].idxmin()]
    best_params = {
        "rank":     int(best_row["rank"]),
        "maxIter":  int(best_row["maxIter"]),
        "regParam": float(best_row["regParam"]),
    }
    logger.info("En iyi Grid Search: %s | RMSE=%.4f", best_params, best_row["rmse"])

    # 3. Ablation Study
    ablation_df = run_ablation_study(train_df, test_df, best_params)

    # 4. Cross Validation
    cv_rmse = run_cross_validation(train_df, test_df, best_params)

    # 5. Görselleştirmeler
    logger.info("Görselleştirmeler oluşturuluyor...")
    plot_grid_search_heatmaps(gs_df)
    plot_grid_search_top10(gs_df)
    plot_ablation_study(ablation_df)
    plot_model_comparison(baselines, float(best_row["rmse"]), cv_rmse)
    plot_rmse_vs_train_time(gs_df)
    plot_rank_regparam_lines(gs_df)

    logger.info("=" * 60)
    logger.info("SONUÇ ÖZETİ")
    logger.info("  Baseline Global Mean RMSE : %.4f", baselines["global_mean"])
    logger.info("  Baseline User Mean RMSE   : %.4f", baselines["user_mean"])
    logger.info("  Baseline Item Mean RMSE   : %.4f", baselines["item_mean"])
    logger.info("  Grid Search En İyi RMSE   : %.4f", best_row["rmse"])
    logger.info("  Cross-Val Test RMSE       : %.4f", cv_rmse)
    logger.info("  İyileşme (Global→CV)      : %.2f%%",
                (baselines["global_mean"] - cv_rmse) / baselines["global_mean"] * 100)
    logger.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
