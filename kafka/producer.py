# Takım Üyesi: Ayaz (Junior Mühendis)
# Görev: MovieLens ratings.csv dosyasını satır satır okuyup
#         Kafka "ratings" topic'ine JSON formatında gönderen producer.

import csv
import json
import time
import logging
import sys
import os
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# ─── Loglama Ayarları ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
KAFKA_BROKER = "kafka:29092"          # Konteyner içi Kafka adresi (pipeline-net)
TOPIC_NAME   = "ratings"             # Mesajların gönderileceği topic
DATA_PATH    = "/app/data/ratings.csv"  # Docker volume mount yolu
DELAY_SEC    = float(os.environ.get("DELAY_SEC", "0.01")) # Mesajlar arası gecikme
LOG_INTERVAL = 10_000                # Her kaç mesajda bir ilerleme logu yazılsın


def create_producer(broker: str, retries: int = 5) -> KafkaProducer:
    """Kafka producer oluşturur; bağlantı kurulamazsa yeniden dener."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=broker,
                # Mesajları JSON string'e çeviren serializer
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Tüm replica'lar yazma onayı versin (güvenilirlik)
                acks="all",
                # Hata durumunda en fazla 3 kez tekrar dene
                retries=3,
                # Toplu gönderim için 16 KB buffer
                batch_size=16384,
                # 10 ms bekle, daha fazla mesaj biriktir
                linger_ms=10,
            )
            logger.info("Kafka bağlantısı kuruldu: %s", broker)
            return producer
        except NoBrokersAvailable:
            logger.warning(
                "Kafka broker bulunamadı (deneme %d/%d). 5 saniye bekleniyor...",
                attempt, retries,
            )
            time.sleep(5)

    logger.error("Kafka broker'a bağlanılamadı. Program sonlandırılıyor.")
    sys.exit(1)


def delivery_callback(err, msg):
    """Gönderim başarısız olduğunda loglayan callback."""
    if err:
        logger.error("Mesaj gönderilemedi | hata: %s", err)


def stream_ratings(producer: KafkaProducer, csv_path: str) -> None:
    """
    ratings.csv dosyasını satır satır okur ve her satırı
    Kafka topic'e JSON mesajı olarak gönderir.
    """
    sent_count   = 0
    error_count  = 0
    skipped_rows = 0

    logger.info("Veri akışı başlıyor: %s → topic '%s'", csv_path, TOPIC_NAME)

    try:
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                try:
                    # CSV kolonlarını doğru tipe dönüştür
                    message = {
                        "userId":    int(row["userId"]),
                        "movieId":   int(row["movieId"]),
                        "rating":    float(row["rating"]),
                        "timestamp": int(row["timestamp"]),
                    }
                except (ValueError, KeyError) as parse_err:
                    # Hatalı satırı atla, sayacı artır
                    skipped_rows += 1
                    logger.debug("Satır atlandı (parse hatası): %s | hata: %s", row, parse_err)
                    continue

                try:
                    # Mesajı Kafka'ya gönder (asenkron)
                    producer.send(
                        TOPIC_NAME,
                        value=message,
                        # userId'yi partition key olarak kullan (aynı kullanıcı aynı partition'a)
                        key=str(message["userId"]).encode("utf-8"),
                    ).add_errback(lambda e: delivery_callback(e, None))

                    sent_count += 1

                    # Her LOG_INTERVAL mesajda bir ilerleme bilgisi yaz
                    if sent_count % LOG_INTERVAL == 0:
                        logger.info(
                            "Gönderilen mesaj: %d | Atlanan satır: %d",
                            sent_count, skipped_rows,
                        )

                    # Gerçekçi streaming simülasyonu için kısa gecikme
                    time.sleep(DELAY_SEC)

                except KafkaError as kafka_err:
                    error_count += 1
                    logger.error("Kafka gönderim hatası: %s", kafka_err)

    except FileNotFoundError:
        logger.error("Dosya bulunamadı: %s", csv_path)
        sys.exit(1)

    # Tüm bekleyen mesajların gönderilmesini garantile
    producer.flush()

    logger.info(
        "Akış tamamlandı — Gönderilen: %d | Atlanan: %d | Hata: %d",
        sent_count, skipped_rows, error_count,
    )


def main():
    producer = create_producer(KAFKA_BROKER)

    try:
        stream_ratings(producer, DATA_PATH)
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu (Ctrl+C).")
    finally:
        producer.close()
        logger.info("Kafka producer kapatıldı.")


if __name__ == "__main__":
    main()
