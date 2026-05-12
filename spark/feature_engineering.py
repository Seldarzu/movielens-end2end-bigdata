# Takım Üyesi: Arzu (Kıdemli Mühendis) & Ayaz (Junior Mühendis)
# Görev: Kapsamlı Feature Engineering Pipeline'ı
#   - Temporal feature'lar (gün, saat, ay, mevsim, hafta sonu)
#   - Kullanıcı bazlı feature'lar (sapma, aktivite, varyans)
#   - Film bazlı feature'lar (yaş, popülerlik, tür sayısı, one-hot)
#   - Etkileşim feature'ları (tür ort, sıra, süre)
#   - Feature importance analizi ve görselleştirme
#   - Delta Lake'e yazma

import logging
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, avg, stddev, lit, when, size, split, explode,
    round as spark_round, from_unixtime, to_timestamp,
    dayofweek, hour, month as spark_month, year as spark_year,
    min as spark_min, max as spark_max, datediff,
    row_number, lag, regexp_extract, collect_set,
    first, variance,
)
from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# ─── Loglama ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
DELTA_RATINGS_PATH  = "/app/delta/ratings_cleaned"
MOVIES_CSV_PATH     = "/app/data/movies.csv"
OUTPUT_PATH         = "/app/delta/ratings_features/full_features"
PLOT_DIR            = "/app/delta/eda"

# Grafik stili
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


def save_fig(name: str) -> None:
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafik kaydedildi → %s", path)


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("MovieLens-FeatureEngineering")
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
    logger.info("Spark oturumu hazır.")
    return spark


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: TEMPORAL (ZAMAN) FEATURE'LAR
# ═══════════════════════════════════════════════════════════════════════════════

def add_temporal_features(df: DataFrame) -> DataFrame:
    """
    Timestamp'ten zaman bazlı feature'lar üretir:
    - rating_datetime: okunabilir tarih
    - day_of_week: 1=Pazar..7=Cumartesi (Spark default), 0-6'ya map'lenir
    - hour_of_day: 0-23 arası saat
    - month: 1-12 arası ay
    - season: 1=İlkbahar, 2=Yaz, 3=Sonbahar, 4=Kış
    - is_weekend: hafta sonu mu (1/0)
    - day_period: sabah/öğle/akşam/gece
    """
    logger.info("=== Temporal Feature'lar Ekleniyor ===")

    df = df.withColumn("rating_datetime",
                       to_timestamp(from_unixtime(col("timestamp"))))

    # day_of_week: Spark'ta 1=Pazar, biz 0=Pazartesi yapıyoruz
    df = df.withColumn("day_of_week_raw", dayofweek(col("rating_datetime")))
    df = df.withColumn("day_of_week",
                       when(col("day_of_week_raw") == 1, 6)  # Pazar → 6
                       .otherwise(col("day_of_week_raw") - 2))
    df = df.drop("day_of_week_raw")

    # hour_of_day
    df = df.withColumn("hour_of_day", hour(col("rating_datetime")))

    # month
    df = df.withColumn("month", spark_month(col("rating_datetime")))

    # season: 3-5=İlkbahar(1), 6-8=Yaz(2), 9-11=Sonbahar(3), 12,1,2=Kış(4)
    df = df.withColumn("season",
                       when(col("month").isin(3, 4, 5), 1)
                       .when(col("month").isin(6, 7, 8), 2)
                       .when(col("month").isin(9, 10, 11), 3)
                       .otherwise(4))

    # is_weekend: Cumartesi(5) veya Pazar(6)
    df = df.withColumn("is_weekend",
                       when(col("day_of_week").isin(5, 6), 1).otherwise(0))

    # day_period: sabah(6-11), öğle(12-17), akşam(18-23), gece(0-5)
    df = df.withColumn("day_period",
                       when((col("hour_of_day") >= 6) & (col("hour_of_day") < 12), 1)   # sabah
                       .when((col("hour_of_day") >= 12) & (col("hour_of_day") < 18), 2)  # öğle
                       .when((col("hour_of_day") >= 18) & (col("hour_of_day") < 24), 3)  # akşam
                       .otherwise(0))  # gece

    logger.info("Temporal feature'lar eklendi: day_of_week, hour_of_day, month, season, is_weekend, day_period")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: KULLANICI BAZLI FEATURE'LAR
# ═══════════════════════════════════════════════════════════════════════════════

def add_user_features(df: DataFrame) -> DataFrame:
    """
    Kullanıcı bazlı feature'lar:
    - user_avg_rating: kullanıcının ortalama puanı
    - user_rating_count: toplam rating sayısı
    - user_rating_deviation: bu puanın kendi ortalamasından sapması
    - user_rating_variance: puanlama varyansı (tutarlılık)
    - user_activity_days: ilk ve son rating arası gün farkı
    """
    logger.info("=== Kullanıcı Bazlı Feature'lar Ekleniyor ===")

    user_window = Window.partitionBy("userId")

    # Kullanıcı ortalaması ve sayısı
    df = df.withColumn("user_avg_rating",
                       spark_round(avg("rating").over(user_window), 3))
    df = df.withColumn("user_rating_count",
                       count("*").over(user_window))

    # Sapma: bu rating - kullanıcının ortalaması
    df = df.withColumn("user_rating_deviation",
                       spark_round(col("rating") - col("user_avg_rating"), 3))

    # Varyans (tutarlılık)
    df = df.withColumn("user_rating_variance",
                       spark_round(variance("rating").over(user_window), 4))

    # Aktivite süresi (ilk ve son rating arası gün)
    df = df.withColumn("user_first_rating",
                       spark_min("rating_datetime").over(user_window))
    df = df.withColumn("user_last_rating",
                       spark_max("rating_datetime").over(user_window))
    df = df.withColumn("user_activity_days",
                       datediff(col("user_last_rating"), col("user_first_rating")))
    df = df.drop("user_first_rating", "user_last_rating")

    logger.info("Kullanıcı feature'ları eklendi: user_avg_rating, user_rating_count, "
                "user_rating_deviation, user_rating_variance, user_activity_days")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: FİLM BAZLI FEATURE'LAR
# ═══════════════════════════════════════════════════════════════════════════════

def add_movie_features(df: DataFrame, movies_df: DataFrame) -> DataFrame:
    """
    Film bazlı feature'lar:
    - movie_avg_rating: filmin ortalama puanı
    - movie_rating_count: filmin toplam rating sayısı
    - movie_age: filmin yaşı (yıl)
    - genre_count: filmin kaç türe ait olduğu
    - primary_genre: filmin ana türü
    - movie_popularity_trend: son dönem popülerlik (basitleştirilmiş)
    """
    logger.info("=== Film Bazlı Feature'lar Ekleniyor ===")

    movie_window = Window.partitionBy("movieId")

    # Film ortalaması ve sayısı
    df = df.withColumn("movie_avg_rating",
                       spark_round(avg("rating").over(movie_window), 3))
    df = df.withColumn("movie_rating_count",
                       count("*").over(movie_window))

    # Movies bilgisini join et
    movies_clean = movies_df.select(
        col("movieId"),
        col("title").alias("movie_title"),
        col("genres"),
    )

    # Film yılını title'dan çıkar: "Toy Story (1995)" → 1995
    movies_clean = movies_clean.withColumn(
        "movie_year",
        regexp_extract(col("movie_title"), r"\((\d{4})\)\s*$", 1).cast(IntegerType())
    )

    # Tür sayısı
    movies_clean = movies_clean.withColumn(
        "genre_count",
        when(col("genres") == "(no genres listed)", 0)
        .otherwise(size(split(col("genres"), "\\|")))
    )

    # Ana tür (ilk tür)
    movies_clean = movies_clean.withColumn(
        "primary_genre",
        split(col("genres"), "\\|").getItem(0)
    )

    df = df.join(
        movies_clean.select("movieId", "movie_year", "genre_count", "primary_genre", "genres"),
        on="movieId", how="left"
    )

    # Film yaşı
    df = df.withColumn("rating_year", spark_year(col("rating_datetime")))
    df = df.withColumn("movie_age",
                       when(col("movie_year").isNotNull(),
                            col("rating_year") - col("movie_year"))
                       .otherwise(0))

    # Popülerlik trendi: son 2 yıldaki rating oranı / toplam rating oranı
    max_year = df.select(spark_max("rating_year")).first()[0]
    if max_year:
        recent_window = Window.partitionBy("movieId")
        df = df.withColumn(
            "is_recent",
            when(col("rating_year") >= (max_year - 1), 1).otherwise(0)
        )
        df = df.withColumn(
            "movie_recent_count",
            count(when(col("is_recent") == 1, True)).over(movie_window)
        )
        df = df.withColumn(
            "movie_popularity_trend",
            spark_round(
                col("movie_recent_count") / col("movie_rating_count"), 4
            )
        )
        df = df.drop("is_recent", "movie_recent_count")

    logger.info("Film feature'ları eklendi: movie_avg_rating, movie_rating_count, "
                "movie_age, genre_count, primary_genre, movie_popularity_trend")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: TÜR ONE-HOT ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def add_genre_encoding(df: DataFrame) -> DataFrame:
    """primary_genre kolonunu StringIndexer + OneHotEncoder ile encode eder."""
    logger.info("=== Tür One-Hot Encoding Ekleniyor ===")

    # Null genre'ları doldur
    df = df.withColumn("primary_genre",
                       when(col("primary_genre").isNull(), "Unknown")
                       .otherwise(col("primary_genre")))

    indexer = StringIndexer(
        inputCol="primary_genre",
        outputCol="genre_index",
        handleInvalid="keep"
    )
    encoder = OneHotEncoder(
        inputCol="genre_index",
        outputCol="genre_encoded"
    )

    pipeline = Pipeline(stages=[indexer, encoder])
    model = pipeline.fit(df)
    df = model.transform(df)

    logger.info("Tür encoding tamamlandı: genre_index, genre_encoded")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: ETKİLEŞİM FEATURE'LARI
# ═══════════════════════════════════════════════════════════════════════════════

def add_interaction_features(df: DataFrame) -> DataFrame:
    """
    Etkileşim feature'ları:
    - user_genre_avg_rating: kullanıcının bu türe verdiği ortalama puan
    - rating_order: kullanıcı için kaçıncı rating
    - time_since_last_rating: önceki rating'den bu yana geçen saniye
    """
    logger.info("=== Etkileşim Feature'ları Ekleniyor ===")

    # Kullanıcının bu türe verdiği ortalama puan
    user_genre_window = Window.partitionBy("userId", "primary_genre")
    df = df.withColumn("user_genre_avg_rating",
                       spark_round(avg("rating").over(user_genre_window), 3))

    # Rating sırası (kronolojik)
    user_time_window = Window.partitionBy("userId").orderBy("timestamp")
    df = df.withColumn("rating_order",
                       row_number().over(user_time_window))

    # Son rating'den bu yana geçen süre (saniye)
    df = df.withColumn("prev_timestamp",
                       lag("timestamp", 1).over(user_time_window))
    df = df.withColumn("time_since_last_rating",
                       when(col("prev_timestamp").isNotNull(),
                            col("timestamp") - col("prev_timestamp"))
                       .otherwise(0))
    df = df.drop("prev_timestamp")

    logger.info("Etkileşim feature'ları eklendi: user_genre_avg_rating, "
                "rating_order, time_since_last_rating")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: FEATURE IMPORTANCE ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_feature_importance(df: DataFrame) -> None:
    """
    RandomForest ile feature importance hesaplar ve görselleştirir.
    Rating'i binary sınıfa çevirip (>=3.5 → 1) sınıflandırma yapar.
    """
    logger.info("=== Feature Importance Analizi ===")

    numeric_features = [
        "day_of_week", "hour_of_day", "month", "season", "is_weekend",
        "day_period", "user_avg_rating", "user_rating_count",
        "user_rating_deviation", "user_rating_variance", "user_activity_days",
        "movie_avg_rating", "movie_rating_count", "movie_age",
        "genre_count", "movie_popularity_trend",
        "user_genre_avg_rating", "rating_order", "time_since_last_rating",
    ]

    # Mevcut kolonları filtrele
    available = [f for f in numeric_features if f in df.columns]
    logger.info("Kullanılacak feature sayısı: %d", len(available))

    # Null'ları 0 ile doldur
    for feat in available:
        df = df.withColumn(feat, when(col(feat).isNull(), 0).otherwise(col(feat)))

    # Binary label
    df = df.withColumn("label",
                       when(col("rating") >= 3.5, 1.0).otherwise(0.0))

    # VectorAssembler
    assembler = VectorAssembler(inputCols=available, outputCol="features",
                                handleInvalid="skip")
    assembled = assembler.transform(df).select("features", "label")

    # Küçük örneklem al (hız için)
    sample = assembled.sample(False, 0.05, seed=42)
    train_data, _ = sample.randomSplit([0.8, 0.2], seed=42)

    logger.info("Random Forest eğitiliyor (feature importance için)...")
    rf = RandomForestClassifier(
        numTrees=50, maxDepth=8, seed=42,
        featuresCol="features", labelCol="label"
    )
    rf_model = rf.fit(train_data)

    importances = rf_model.featureImportances.toArray()

    # Görselleştir
    feat_imp = sorted(zip(available, importances), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in feat_imp]
    values = [f[1] for f in feat_imp]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color=PALETTE[1], edgecolor="#1e1e2e")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Önem Skoru")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#cdd6f4")

    ax.grid(axis="x")
    plt.tight_layout()
    save_fig("15_feature_importance")

    # Konsola yazdır
    logger.info("─── Feature Importance Sıralaması ───")
    for name, imp in feat_imp:
        logger.info("  %-30s : %.4f", name, imp)


# ═══════════════════════════════════════════════════════════════════════════════
# YAZMA & ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════════

def write_features(df: DataFrame, path: str) -> None:
    """Feature tablosunu Delta Lake'e yazar."""
    # genre_encoded (SparseVector) Delta'ya yazılamaz, string'e çevir
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import SparseVector, DenseVector

    def vec_to_str(v):
        if v is None:
            return ""
        return str(v.toArray().tolist())

    vec_udf = udf(vec_to_str, StringType())

    df_write = df
    if "genre_encoded" in df.columns:
        df_write = df_write.withColumn("genre_encoded_str", vec_udf(col("genre_encoded")))
        df_write = df_write.drop("genre_encoded", "genre_index")

    # features kolonu da varsa düşür (VectorAssembler'dan kalan)
    for drop_col in ["features", "label", "rawPrediction", "probability", "prediction"]:
        if drop_col in df_write.columns:
            df_write = df_write.drop(drop_col)

    try:
        (
            df_write.write
            .format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .save(path)
        )
        logger.info("Feature tablosu yazıldı → %s (%d satır)", path, df_write.count())
    except Exception as e:
        logger.error("Feature yazma hatası: %s", e)
        raise


def main():
    spark = create_spark_session()

    # ── Veri Yükleme ──────────────────────────────────────────────────────
    logger.info("Veri yükleniyor...")

    # Temizlenmiş veriyi oku (preprocessing.py çıktısı)
    # Eğer cleaned yoksa ham veriyi oku
    try:
        df = spark.read.format("delta").load(DELTA_RATINGS_PATH)
        logger.info("Temizlenmiş veri yüklendi: %s", DELTA_RATINGS_PATH)
    except Exception:
        logger.warning("Temizlenmiş veri bulunamadı, ham veri kullanılıyor.")
        df = spark.read.format("delta").load("/app/delta/ratings")

    movies_df = spark.read.csv(MOVIES_CSV_PATH, header=True, inferSchema=True)

    total = df.count()
    logger.info("Yüklenen satır: %d | Film sayısı: %d", total, movies_df.count())

    # ── Feature Engineering Adımları ──────────────────────────────────────
    # 1. Temporal feature'lar
    df = add_temporal_features(df)

    # 2. Kullanıcı bazlı feature'lar
    df = add_user_features(df)

    # 3. Film bazlı feature'lar
    df = add_movie_features(df, movies_df)

    # 4. Tür one-hot encoding
    df = add_genre_encoding(df)

    # 5. Etkileşim feature'ları
    df = add_interaction_features(df)

    # Şemayı göster
    logger.info("Son şema:")
    df.printSchema()
    logger.info("Toplam kolon sayısı: %d", len(df.columns))

    # 6. Feature importance analizi
    analyze_feature_importance(df)

    # 7. Delta Lake'e yaz
    write_features(df, OUTPUT_PATH)

    logger.info("=" * 60)
    logger.info("Feature Engineering tamamlandı!")
    logger.info("  Temporal    : day_of_week, hour_of_day, month, season, is_weekend, day_period")
    logger.info("  Kullanıcı   : user_avg_rating, user_rating_count, user_rating_deviation, "
                "user_rating_variance, user_activity_days")
    logger.info("  Film        : movie_avg_rating, movie_rating_count, movie_age, genre_count, "
                "primary_genre, movie_popularity_trend")
    logger.info("  Encoding    : genre_index, genre_encoded")
    logger.info("  Etkileşim   : user_genre_avg_rating, rating_order, time_since_last_rating")
    logger.info("  Çıktı       : %s", OUTPUT_PATH)
    logger.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
