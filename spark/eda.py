# Takım Üyesi: Arzu (Kıdemli Mühendis)
# Görev: MovieLens 25M verisi üzerinde keşifsel veri analizi (EDA).
#         Tüm grafikler /app/delta/eda/ klasörüne PNG olarak kaydedilir.

import logging
import sys
import os
import matplotlib
matplotlib.use("Agg")  # Sunucusuz (headless) ortam için — ekran gerektirmez
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, avg, round as spark_round,
    year, month, from_unixtime, explode, split,
    desc, asc, when, lit,
)

# ─── Loglama ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
DELTA_RATINGS_PATH  = "/app/delta/ratings"
MOVIES_CSV_PATH     = "/app/data/movies.csv"
EDA_OUTPUT_DIR      = "/app/delta/eda"

# Grafik stili
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
        .appName("MovieLens-EDA")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    logger.info("Spark oturumu hazır. EDA çıktıları → %s", EDA_OUTPUT_DIR)
    return spark


def save_fig(name: str) -> None:
    path = os.path.join(EDA_OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafik kaydedildi → %s", path)


# ─── 1. Rating Dağılımı ──────────────────────────────────────────────────────
def plot_rating_distribution(ratings_pd: pd.DataFrame) -> None:
    counts = ratings_pd["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE[0], edgecolor="#1e1e2e", width=0.6)
    ax.bar_label(bars, labels=[f"{v/1e6:.1f}M" for v in counts.values],
                 padding=4, color="#cdd6f4", fontsize=10)
    ax.set_title("Rating Dağılımı")
    ax.set_xlabel("Rating Değeri")
    ax.set_ylabel("Rating Sayısı")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(axis="y")
    plt.tight_layout()
    save_fig("01_rating_distribution")


# ─── 2. Yıllara Göre Rating Trendi ───────────────────────────────────────────
def plot_rating_trend(spark: SparkSession, ratings_df: DataFrame) -> None:
    trend = (
        ratings_df
        .withColumn("year", year(from_unixtime(col("timestamp"))))
        .groupBy("year").agg(count("*").alias("cnt"))
        .orderBy("year")
        .toPandas()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(trend["year"], trend["cnt"] / 1e6, marker="o", color=PALETTE[1],
            linewidth=2.5, markersize=6)
    ax.fill_between(trend["year"], trend["cnt"] / 1e6, alpha=0.15, color=PALETTE[1])
    ax.set_title("Yıllara Göre Rating Sayısı Trendi")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Rating Sayısı (Milyon)")
    ax.grid(True)
    plt.tight_layout()
    save_fig("02_rating_trend_by_year")


# ─── 3. En Çok Puanlanan 20 Film ─────────────────────────────────────────────
def plot_top_movies(spark: SparkSession, ratings_df: DataFrame, movies_df: DataFrame) -> None:
    top = (
        ratings_df.groupBy("movieId")
        .agg(count("*").alias("cnt"), spark_round(avg("rating"), 2).alias("avg_r"))
        .orderBy(desc("cnt")).limit(20)
        .join(movies_df, "movieId", "left")
        .select("title", "cnt", "avg_r")
        .toPandas()
    )
    # Film başlığını kıs
    top["title"] = top["title"].str[:35]
    top = top.sort_values("cnt")

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(top["title"], top["cnt"] / 1e3, color=PALETTE[2], edgecolor="#1e1e2e")
    for bar, avg_r in zip(bars, top["avg_r"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"★{avg_r}", va="center", fontsize=9, color="#f9e2af")
    ax.set_title("En Çok Puanlanan 20 Film")
    ax.set_xlabel("Rating Sayısı (Bin)")
    ax.grid(axis="x")
    plt.tight_layout()
    save_fig("03_top_20_movies")


# ─── 4. En Yüksek Puanlı 20 Film (min 500 rating) ────────────────────────────
def plot_highest_rated_movies(spark: SparkSession, ratings_df: DataFrame, movies_df: DataFrame) -> None:
    top = (
        ratings_df.groupBy("movieId")
        .agg(count("*").alias("cnt"), spark_round(avg("rating"), 3).alias("avg_r"))
        .filter(col("cnt") >= 500)
        .orderBy(desc("avg_r")).limit(20)
        .join(movies_df, "movieId", "left")
        .select("title", "avg_r", "cnt")
        .toPandas()
    )
    top["title"] = top["title"].str[:35]
    top = top.sort_values("avg_r")

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(top["title"], top["avg_r"], color=PALETTE[3], edgecolor="#1e1e2e")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in top["avg_r"]],
                 padding=3, color="#cdd6f4", fontsize=9)
    ax.set_title("En Yüksek Ortalama Puanlı 20 Film (min 500 rating)")
    ax.set_xlabel("Ortalama Rating")
    ax.set_xlim(3.5, 5.0)
    ax.grid(axis="x")
    plt.tight_layout()
    save_fig("04_highest_rated_movies")


# ─── 5. Kullanıcı Aktivite Dağılımı ──────────────────────────────────────────
def plot_user_activity(spark: SparkSession, ratings_df: DataFrame) -> None:
    user_counts = (
        ratings_df.groupBy("userId")
        .agg(count("*").alias("cnt"))
        .toPandas()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: histogram
    axes[0].hist(user_counts["cnt"], bins=60, color=PALETTE[4], edgecolor="#1e1e2e", log=True)
    axes[0].set_title("Kullanıcı Başına Rating Sayısı (log)")
    axes[0].set_xlabel("Rating Sayısı")
    axes[0].set_ylabel("Kullanıcı Sayısı (log)")
    axes[0].grid(True)

    # Sağ: kutu grafiği (outlier'ları göster)
    axes[1].boxplot(user_counts["cnt"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[4], color="#cdd6f4"),
                    medianprops=dict(color="#f9e2af", linewidth=2))
    axes[1].set_title("Kullanıcı Aktivitesi Kutu Grafiği")
    axes[1].set_ylabel("Rating Sayısı")
    axes[1].grid(axis="y")

    plt.tight_layout()
    save_fig("05_user_activity_distribution")


# ─── 6. Tür Bazında Ortalama Rating ──────────────────────────────────────────
def plot_genre_ratings(spark: SparkSession, ratings_df: DataFrame, movies_df: DataFrame) -> None:
    genre_ratings = (
        ratings_df.join(movies_df, "movieId", "left")
        .withColumn("genre", explode(split(col("genres"), "\\|")))
        .filter(col("genre") != "(no genres listed)")
        .groupBy("genre")
        .agg(
            spark_round(avg("rating"), 3).alias("avg_r"),
            count("*").alias("cnt"),
        )
        .orderBy(desc("avg_r"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(genre_ratings["genre"], genre_ratings["avg_r"],
                  color=PALETTE[:len(genre_ratings)], edgecolor="#1e1e2e")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in genre_ratings["avg_r"]],
                 padding=3, color="#cdd6f4", fontsize=9)
    ax.set_title("Tür Bazında Ortalama Rating")
    ax.set_xlabel("Film Türü")
    ax.set_ylabel("Ortalama Rating")
    ax.set_ylim(3.0, 4.5)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y")
    plt.tight_layout()
    save_fig("06_genre_avg_rating")


# ─── 7. Aylık Rating Yoğunluğu (Heatmap) ────────────────────────────────────
def plot_monthly_heatmap(spark: SparkSession, ratings_df: DataFrame) -> None:
    monthly = (
        ratings_df
        .withColumn("year",  year(from_unixtime(col("timestamp"))))
        .withColumn("month", month(from_unixtime(col("timestamp"))))
        .groupBy("year", "month").agg(count("*").alias("cnt"))
        .filter((col("year") >= 2000) & (col("year") <= 2019))
        .toPandas()
    )
    pivot = monthly.pivot(index="month", columns="year", values="cnt").fillna(0)

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="plasma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(12))
    ax.set_yticklabels(["Oca", "Şub", "Mar", "Nis", "May", "Haz",
                        "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"])
    ax.set_title("Aylık Rating Yoğunluğu (2000–2019)")
    plt.colorbar(im, ax=ax, label="Rating Sayısı")
    plt.tight_layout()
    save_fig("07_monthly_heatmap")


# ─── 8. Rating Değeri Zaman Trendi ───────────────────────────────────────────
def plot_avg_rating_trend(spark: SparkSession, ratings_df: DataFrame) -> None:
    trend = (
        ratings_df
        .withColumn("year", year(from_unixtime(col("timestamp"))))
        .groupBy("year").agg(spark_round(avg("rating"), 3).alias("avg_r"))
        .filter((col("year") >= 1995) & (col("year") <= 2019))
        .orderBy("year")
        .toPandas()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(trend["year"], trend["avg_r"], marker="s", color=PALETTE[5],
            linewidth=2.5, markersize=7)
    ax.fill_between(trend["year"], trend["avg_r"], alpha=0.1, color=PALETTE[5])
    ax.axhline(y=trend["avg_r"].mean(), linestyle="--", color=PALETTE[6],
               linewidth=1.5, label=f"Genel Ort: {trend['avg_r'].mean():.2f}")
    ax.set_title("Yıllar İçinde Ortalama Rating Değeri")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Ortalama Rating")
    ax.set_ylim(3.0, 4.5)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_fig("08_avg_rating_trend")


def main():
    spark = create_spark_session()

    # Ham veriyi yükle
    ratings_df = spark.read.format("delta").load(DELTA_RATINGS_PATH)
    movies_df  = spark.read.csv(MOVIES_CSV_PATH, header=True, inferSchema=True)

    total = ratings_df.count()
    logger.info("Toplam rating sayısı: %d", total)

    # Pandas'a küçük özet çek (rating dağılımı için)
    ratings_sample = ratings_df.select("rating").toPandas()

    logger.info("EDA grafikleri oluşturuluyor...")

    plot_rating_distribution(ratings_sample)
    plot_rating_trend(spark, ratings_df)
    plot_top_movies(spark, ratings_df, movies_df)
    plot_highest_rated_movies(spark, ratings_df, movies_df)
    plot_user_activity(spark, ratings_df)
    plot_genre_ratings(spark, ratings_df, movies_df)
    plot_monthly_heatmap(spark, ratings_df)
    plot_avg_rating_trend(spark, ratings_df)

    logger.info("Tüm EDA grafikleri tamamlandı → %s", EDA_OUTPUT_DIR)
    spark.stop()


if __name__ == "__main__":
    main()
