# Takım Üyesi: Arzu (Kıdemli Mühendis)
# Görev: Delta Lake'ten ham ratings verisini okuyup feature tablosu üretmek.
#         - timestamp → okunabilir tarih
#         - Film bazında ortalama rating ve rating sayısı
#         - Kullanıcı bazında rating sayısı ve ortalama rating

import logging
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, from_unixtime, to_timestamp,
    avg, count, round as spark_round,
)

# ─── Loglama Ayarları ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
DELTA_RATINGS_PATH   = "/app/delta/ratings"           # Ham veri kaynağı
DELTA_FEATURES_PATH  = "/app/delta/ratings_features"  # ETL çıktısı

# Film ve kullanıcı istatistiklerinin ayrı çıktı dizinleri
MOVIE_STATS_PATH     = f"{DELTA_FEATURES_PATH}/movie_stats"
USER_STATS_PATH      = f"{DELTA_FEATURES_PATH}/user_stats"
ENRICHED_PATH        = f"{DELTA_FEATURES_PATH}/enriched_ratings"


def create_spark_session() -> SparkSession:
    """Delta Lake destekli Spark oturumu başlatır."""
    try:
        spark = (
            SparkSession.builder
            .appName("MovieLens-ETL")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config(
                "spark.jars.packages",
                "io.delta:delta-core_2.12:2.4.0",
            )
            # Büyük veri setlerinde daha verimli join için
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark oturumu oluşturuldu.")
        return spark
    except Exception as e:
        logger.error("Spark oturumu başlatılamadı: %s", e)
        sys.exit(1)


def read_ratings(spark: SparkSession) -> DataFrame:
    """
    Delta Lake'ten ham ratings tablosunu okur.
    Consumer tarafından yazılan 'ratings' Delta tablosu beklenir.
    """
    try:
        df = spark.read.format("delta").load(DELTA_RATINGS_PATH)
        row_count = df.count()
        logger.info("Ratings yüklendi — satır sayısı: %d", row_count)
        return df
    except Exception as e:
        logger.error("Delta okuma hatası (%s): %s", DELTA_RATINGS_PATH, e)
        sys.exit(1)


def enrich_timestamps(df: DataFrame) -> DataFrame:
    """
    Unix epoch timestamp kolonunu okunabilir datetime'a çevirir.
    Örnek: 964982703 → '2000-07-30 18:45:03'
    """
    return df.withColumn(
        "rating_date",
        to_timestamp(from_unixtime(col("timestamp"))),
    )


def compute_movie_stats(df: DataFrame) -> DataFrame:
    """
    Film bazında özet istatistikler üretir:
    - avg_rating    : ortalama rating (2 ondalık)
    - rating_count  : toplam rating sayısı
    """
    return (
        df.groupBy("movieId")
        .agg(
            spark_round(avg("rating"), 2).alias("avg_rating"),
            count("rating").alias("rating_count"),
        )
        .orderBy(col("rating_count").desc())
    )


def compute_user_stats(df: DataFrame) -> DataFrame:
    """
    Kullanıcı bazında özet istatistikler üretir:
    - rating_count  : kullanıcının toplam rating sayısı
    - avg_rating    : kullanıcının ortalama verdiği rating
    """
    return (
        df.groupBy("userId")
        .agg(
            count("rating").alias("rating_count"),
            spark_round(avg("rating"), 2).alias("avg_rating"),
        )
        .orderBy(col("rating_count").desc())
    )


def write_delta(df: DataFrame, path: str, label: str) -> None:
    """DataFrame'i Delta formatında belirtilen yola yazar."""
    try:
        (
            df.write
            .format("delta")
            .mode("overwrite")
            # Tabloya yeni kolonlar eklenirse hata fırlatma, birleştir
            .option("mergeSchema", "true")
            .save(path)
        )
        logger.info("%s Delta'ya yazıldı → %s", label, path)
    except Exception as e:
        logger.error("%s yazma hatası: %s", label, e)
        raise


def main():
    spark = create_spark_session()

    # 1. Ham veriyi oku
    ratings_df = read_ratings(spark)

    # 2. Timestamp'i tarihe dönüştür
    enriched_df = enrich_timestamps(ratings_df)

    # 3. Film istatistiklerini hesapla
    movie_stats_df = compute_movie_stats(enriched_df)

    # 4. Kullanıcı istatistiklerini hesapla
    user_stats_df = compute_user_stats(enriched_df)

    # 5. Zenginleştirilmiş ratings'i yaz (ML için gerekli tüm kolonlar)
    write_delta(enriched_df,    ENRICHED_PATH,    "Zenginleştirilmiş ratings")
    write_delta(movie_stats_df, MOVIE_STATS_PATH, "Film istatistikleri")
    write_delta(user_stats_df,  USER_STATS_PATH,  "Kullanıcı istatistikleri")

    # Örnek çıktıyı logla
    logger.info("--- En Çok Puanlanan Filmler (ilk 5) ---")
    movie_stats_df.show(5, truncate=False)

    logger.info("--- En Aktif Kullanıcılar (ilk 5) ---")
    user_stats_df.show(5, truncate=False)

    logger.info("ETL tamamlandı.")
    spark.stop()


if __name__ == "__main__":
    main()
