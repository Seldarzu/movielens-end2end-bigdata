# MovieLens 25M — Uçtan Uca Büyük Veri Pipeline'ı

> **Takım:** Arzu (Kıdemli Mühendis) · Ayaz (Junior Mühendis)
> **Son güncelleme:** _[Ayaz: tarih ekle]_

---

## Proje Özeti

Bu proje, [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) veri setini kullanarak
gerçek zamanlı bir büyük veri pipeline'ı sunar. Ham CSV verisi Kafka üzerinden akar,
Spark Structured Streaming ile Delta Lake'e yazılır, ETL adımından geçerek feature tabloları
üretilir ve son olarak ALS tabanlı bir öneri modeli eğitilip MLflow'a kaydedilir.

---

## Mimari

```
ratings.csv
    │
    ▼
[Kafka Producer]  ──► Kafka Topic: "ratings"
                              │
                              ▼
                    [Spark Structured Streaming]
                    (consumer.py)
                              │
                              ▼
                    Delta Lake: /delta/ratings
                              │
                              ▼
                    [ETL Dönüşümleri]
                    (etl.py)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             movie_stats           user_stats
                    │
                    ▼
            [ALS Modeli Eğitimi]
            (model.py)
                    │
                    ▼
            [MLflow Loglama]
            (log_model.py)
                    │
                    ▼
            MLflow UI: http://localhost:5000
```

---

## Teknoloji Stack

| Katman          | Teknoloji                           | Versiyon  |
|-----------------|-------------------------------------|-----------|
| Konteynerizasyon| Docker + docker-compose             | -         |
| Mesaj Kuyruğu   | Apache Kafka + Zookeeper            | 7.4.0     |
| İşleme Motoru   | Apache Spark (Structured Streaming) | 3.4.1     |
| Depolama Formatı| Delta Lake                          | 2.4.0     |
| ML Takip        | MLflow                              | 2.11.1    |
| Dil             | Python                              | 3.10+     |
| Veri Seti       | MovieLens 25M                       | -         |

---

## Ön Koşullar

- Docker Desktop (en az 6 GB RAM ayır)
- Python 3.10+
- MovieLens 25M veri seti (`ratings.csv`, `movies.csv`) → `./data/` klasörüne koy

```bash
# MovieLens 25M indir
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/
```

---

## Hızlı Başlangıç

### 1. Servisleri Başlat

```bash
docker-compose up -d
```

Servislerin hazır olduğunu doğrula:

```bash
docker-compose ps
```

### 2. Kafka Producer'ı Çalıştır

```bash
cd kafka/
pip install kafka-python
python producer.py
```

### 3. Spark Consumer'ı Başlat (ayrı terminalde)

```bash
docker exec -it spark-master \
  spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 \
  /app/spark/consumer.py
```

### 4. ETL'i Çalıştır

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/etl.py
```

### 5. Modeli Eğit

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/model.py
```

### 6. MLflow'a Logla

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/log_model.py
```

---

## Arayüzler

| Servis       | URL                        |
|--------------|----------------------------|
| Spark Web UI | http://localhost:8080       |
| MLflow UI    | http://localhost:5000       |

---

## Delta Lake Dizin Yapısı

```
delta/
├── ratings/                      # Ham Kafka stream verisi
├── checkpoints/
│   └── ratings/                  # Structured Streaming checkpoint
├── ratings_features/
│   ├── enriched_ratings/         # Zaman damgası zenginleştirilmiş tablo
│   ├── movie_stats/              # Film bazında istatistikler
│   └── user_stats/               # Kullanıcı bazında istatistikler
└── als_model/                    # Eğitilmiş ALS modeli
```

---

## Sorun Giderme

_[Ayaz: yaygın hataları ve çözümlerini buraya ekle]_

---

## Katkıda Bulunanlar

- **Arzu** — Docker kurulumu, Spark Structured Streaming, ETL, ALS modeli
- **Ayaz** — Kafka producer, MLflow loglama, bu README

---

## Lisans

Bu proje eğitim amaçlıdır. MovieLens veri seti GroupLens Research tarafından sağlanmaktadır.
