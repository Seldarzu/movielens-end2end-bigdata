# Takım Üyesi: Arzu (Kıdemli Mühendis)
# Görev: Kafka "ratings" topic'inden Spark Structured Streaming ile okuyup
#         gelen veriyi parse edip Delta Lake'e yazan consumer.

import logging
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType

# ─── Loglama Ayarları ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
KAFKA_BROKER        = "kafka:29092"                    # Konteyner içi Kafka adresi
TOPIC_NAME          = "ratings"                        # Okunacak topic
DELTA_OUTPUT_PATH   = "/app/delta/ratings"             # Delta Lake hedef dizini
CHECKPOINT_PATH     = "/app/delta/checkpoints/ratings" # Exactly-once için checkpoint
TRIGGER_INTERVAL    = "10 seconds"                     # Micro-batch işleme aralığı


def create_spark_session() -> SparkSession:
    """
    Delta Lake paketleriyle birlikte Spark oturumu oluşturur.
    Kafka ve Delta Lake JAR'ları Maven'dan otomatik indirilir.
    """
    try:
        spark = (
            SparkSession.builder
            .appName("MovieLens-Kafka-Consumer")
            # Delta Lake catalog ve extension ayarları
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            # Spark Structured Streaming + Kafka entegrasyon paketi
            .config(
                "spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,"
                "io.delta:delta-core_2.12:2.4.0",
            )
            # Delta otomatik şema birleştirme
            .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark oturumu başarıyla oluşturuldu.")
        return spark
    except Exception as e:
        logger.error("Spark oturumu oluşturulamadı: %s", e)
        sys.exit(1)


def get_ratings_schema() -> StructType:
    """Kafka mesajlarındaki JSON payload'ının şemasını tanımlar."""
    return StructType([
        StructField("userId",    IntegerType(), nullable=False),
        StructField("movieId",   IntegerType(), nullable=False),
        StructField("rating",    FloatType(),   nullable=False),
        StructField("timestamp", LongType(),    nullable=False),
    ])


def read_kafka_stream(spark: SparkSession):
    """Kafka topic'inden ham binary stream okur."""
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", TOPIC_NAME)
        # En eski kayıttan başla (geliştirme ortamı için)
        .option("startingOffsets", "earliest")
        # Micro-batch başına en fazla 10.000 mesaj oku (backpressure)
        .option("maxOffsetsPerTrigger", 10_000)
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_and_enrich(raw_stream, schema: StructType):
    """
    Binary Kafka mesajlarını JSON olarak parse eder,
    şema uygular ve işlenme zamanı damgası ekler.
    """
    return (
        raw_stream
        # Kafka'dan gelen 'value' kolonu binary; UTF-8 string'e çevir
        .selectExpr("CAST(value AS STRING) AS json_str", "timestamp AS kafka_timestamp")
        # JSON'ı ayrıştır, şemayı uygula
        .withColumn("data", from_json(col("json_str"), schema))
        # İç içe struct'ı düzleştir
        .select(
            col("data.userId").alias("userId"),
            col("data.movieId").alias("movieId"),
            col("data.rating").alias("rating"),
            col("data.timestamp").alias("timestamp"),
            col("kafka_timestamp"),
            # Pipeline'ın kaydı ne zaman işlediğini izlemek için
            current_timestamp().alias("processed_at"),
        )
        # Parse başarısız olan satırları düşür (userId null olamaz)
        .filter(col("userId").isNotNull())
    )


def write_to_delta(stream) -> None:
    """Parse edilmiş stream'i Delta Lake'e yazar ve sorguyu çalıştırır."""
    query = (
        stream.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        # Her 10 saniyede bir micro-batch işle
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start(DELTA_OUTPUT_PATH)
    )

    logger.info(
        "Structured Streaming sorgusu başladı. Delta hedef: %s",
        DELTA_OUTPUT_PATH,
    )

    try:
        # Sorgu durana ya da hata oluşana kadar bekle
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu. Sorgu kapatılıyor...")
        query.stop()
    except Exception as e:
        logger.error("Streaming hatası: %s", e)
        query.stop()
        raise


def main():
    spark  = create_spark_session()
    schema = get_ratings_schema()

    raw_stream     = read_kafka_stream(spark)
    parsed_stream  = parse_and_enrich(raw_stream, schema)

    write_to_delta(parsed_stream)


if __name__ == "__main__":
    main()
