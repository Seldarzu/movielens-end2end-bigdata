# MovieLens 25M — Uçtan Uca Büyük Veri Pipeline'ı

> **Takım:** Arzu (Kıdemli Mühendis) · Ayaz (Junior Mühendis)  
> **Son güncelleme:** 13 Mayıs 2026  
> **Veri Seti:** [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (~25 milyon rating, ~62.000 film, ~162.000 kullanıcı)

---

## Proje Özeti

Bu proje, MovieLens 25M veri setini kullanarak **gerçek zamanlı bir büyük veri pipeline'ı** sunar.
Ham CSV verisi Kafka üzerinden akar, Spark Structured Streaming ile Delta Lake'e yazılır,
kapsamlı veri ön işleme ve feature engineering adımlarından geçerek zenginleştirilmiş tablolar üretilir.
Ardından **ALS tabanlı öneri modeli**, **4 sınıflandırma** ve **4 regresyon** modeli eğitilip
MLflow'a kaydedilir. Tüm süreç boyunca **25+ görselleştirme** üretilir.

---

## Mimari

```
ratings.csv
    │
    ▼
[Kafka Producer]  ──► Kafka Topic: "ratings"
  (producer.py)              │
                             ▼
                   [Spark Structured Streaming]
                   (consumer.py)
                             │
                             ▼
                   Delta Lake: /delta/ratings
                             │
              ┌──────────────┤
              ▼              ▼
   [Veri Ön İşleme]    [EDA Analizi]
   (preprocessing.py)  (eda.py)
              │         → 8 grafik
              ▼
   Delta Lake: /delta/ratings_cleaned
              │
              ▼
   [Feature Engineering]
   (feature_engineering.py)
              │
              ├── Temporal: day_of_week, hour, season, is_weekend...
              ├── Kullanıcı: deviation, variance, activity_days...
              ├── Film: movie_age, popularity_trend, genre_count...
              └── Etkileşim: user_genre_avg, rating_order...
              │
              ▼
   Delta Lake: /delta/ratings_features/full_features
              │
    ┌─────────┼──────────────────┐
    ▼         ▼                  ▼
[ALS Model] [Sınıflandırma]  [Regresyon]
(model.py)  (classification.py)
    │         │                  │
    │    ┌────┴────┐        ┌───┴────┐
    │    │ LogReg  │        │ Linear │
    │    │ RF      │        │ RF     │
    │    │ GBT     │        │ GBT    │
    │    │ DTree   │        │ DTree  │
    │    └─────────┘        └────────┘
    │
    ▼
[Genişletilmiş Metrikler]
(extended_metrics.py)
    │
    ├── ALS: RMSE, MAE, MSE, R²
    ├── Öneri: Precision@K, Recall@K, NDCG
    └── Sınıflandırma: Confusion Matrix, Macro/Micro
    │
    ▼
[Gelişmiş Görselleştirmeler]
(advanced_visualizations.py)
    │
    ├── ROC Curves, Learning Curve
    └── Residual Plots, Karşılaştırma Grafikleri
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

| Katman           | Teknoloji                           | Versiyon  |
|------------------|-------------------------------------|-----------|
| Konteynerizasyon | Docker + docker-compose             | -         |
| Mesaj Kuyruğu    | Apache Kafka + Zookeeper            | 7.4.0     |
| İşleme Motoru    | Apache Spark (Structured Streaming) | 3.4.1     |
| Depolama Formatı | Delta Lake                          | 2.4.0     |
| ML Framework     | Spark MLlib (ALS, RF, GBT, LR, DT) | 3.4.1     |
| ML Takip         | MLflow                              | 2.11.1    |
| Dil              | Python                              | 3.10+     |
| Görselleştirme   | Matplotlib                          | -         |
| Veri Seti        | MovieLens 25M                       | -         |

---

## Proje Dosya Yapısı

```
movielens-end2end-bigdata/
│
├── docker-compose.yml          # Tüm servisleri ayağa kaldırır
├── run_pipeline.sh             # Tek tuşla tüm pipeline'ı çalıştıran orkestrasyon scripti
├── README.md                   # Bu dosya
│
├── kafka/
│   └── producer.py             # CSV → Kafka (JSON stream)
│
├── spark/
│   ├── consumer.py             # Kafka → Delta Lake (Structured Streaming)
│   ├── preprocessing.py        # Veri ön işleme pipeline'ı
│   ├── eda.py                  # Keşifsel veri analizi (8 grafik)
│   ├── etl.py                  # Temel ETL dönüşümleri
│   ├── feature_engineering.py  # Kapsamlı feature üretimi (19 feature)
│   ├── model.py                # ALS öneri modeli (Grid Search + CV)
│   ├── classification.py       # 4 sınıflandırma + 4 regresyon modeli
│   ├── extended_metrics.py     # Genişletilmiş metrikler + görselleştirme
│   ├── advanced_visualizations.py # Gelişmiş grafikler (ROC, Learning Curve vs.)
│   └── log_model.py            # MLflow'a model loglama
│
├── mlflow/
│   └── log_model.py            # MLflow loglama (yedek kopya)
│
├── data/                       # (git'te yok) MovieLens veri seti buraya konur
│   └── ml-25m/
│       ├── ratings.csv
│       └── movies.csv
│
└── delta/                      # (otomatik oluşur) Delta Lake çıktıları
    ├── ratings/                # Ham Kafka stream verisi
    ├── ratings_cleaned/        # Ön işlemeden geçmiş temiz veri
    ├── checkpoints/            # Structured Streaming checkpoint
    ├── ratings_features/
    │   ├── enriched_ratings/   # Zaman damgası zenginleştirilmiş
    │   ├── movie_stats/        # Film bazında istatistikler
    │   ├── user_stats/         # Kullanıcı bazında istatistikler
    │   └── full_features/      # 19 feature ile zenginleştirilmiş tablo
    ├── als_model/              # Eğitilmiş ALS modeli
    ├── als_model_best/         # En iyi CV modeli
    ├── models/                 # Sınıflandırma + Regresyon modelleri
    ├── eda/                    # Tüm grafik çıktıları (PNG)
    └── preprocessing_report/   # Ön işleme raporları (PNG)
```

---

## Ön Koşullar

- **Docker Desktop** (en az **6 GB RAM** ayır — Ayarlar → Resources → Memory)
- **Python 3.10+** (host makinede, sadece producer için gerekli)
- **MovieLens 25M** veri seti

### Veri Setini İndirme

```bash
# MovieLens 25M indir ve proje klasörüne çıkar
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/
```

> **Not:** İndirilen dosya ~250 MB, açılmış hali ~650 MB.  
> `data/ml-25m/ratings.csv` ve `data/ml-25m/movies.csv` dosyaları oluşmalıdır.

---

## Hızlı Başlangıç (Adım Adım)

### 1. Docker Servislerini Başlat

```bash
# Tüm servisleri arka planda başlat
docker-compose up -d
```

Servislerin hazır olduğunu doğrula:

```bash
docker-compose ps
```

Beklenen çıktı — tüm servisler `Up` durumunda olmalı:

```
NAME              STATUS
zookeeper         Up (healthy)
kafka             Up (healthy)
kafka-producer    Up
spark-master      Up
spark-worker      Up
mlflow            Up (healthy)
```

> **İpucu:** Kafka producer otomatik başlar ve `ratings.csv`'yi Kafka'ya stream etmeye başlar.
> İlk başlangıçta Kafka'nın hazır olması ~30 saniye sürebilir.

### 2. Kafka Producer'ın Çalıştığını Doğrula

```bash
# Producer loglarını izle
docker logs -f kafka-producer
```

Beklenen çıktı:

```
Kafka bağlantısı kuruldu: kafka:29092
Veri akışı başlıyor: /app/data/ratings.csv → topic 'ratings'
Gönderilen mesaj: 10000 | Atlanan satır: 0
Gönderilen mesaj: 20000 | Atlanan satır: 0
...
```

> **Not:** 25M satırın tamamının gönderilmesi ~4-5 saat sürer.
> Demo için birkaç dakika akmasını beklemeniz yeterlidir (50.000-100.000 mesaj).

### 3. Spark Consumer'ı Başlat (ayrı terminal)

```bash
docker exec -it spark-master \
  spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 \
  /app/spark/consumer.py
```

> Consumer, Kafka'dan veriyi 10 saniyelik micro-batch'lerle okuyup Delta Lake'e yazar.
> Yeterli veri aktığında `Ctrl+C` ile durdurabilirsiniz.

### 4. Veri Ön İşleme

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/preprocessing.py
```

Bu adım şunları yapar:
- Missing value analizi ve temizleme
- Duplikasyon kontrolü ve kaldırma
- Outlier detection (IQR yöntemi)
- Aşırı aktif kullanıcı tespiti (10.000+ rating)
- Şüpheli puanlama kalıpları tespiti
- Temiz veriyi `/delta/ratings_cleaned`'e yazar
- Veri kalitesi raporunu MLflow'a loglar

### 5. EDA (Keşifsel Veri Analizi)

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/eda.py
```

8 grafik üretir: rating dağılımı, yıllık trend, top 20 film, en yüksek puanlı filmler,
kullanıcı aktivitesi, tür analizi, aylık heatmap, ortalama rating trendi.

### 6. ETL Dönüşümleri

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/etl.py
```

### 7. Feature Engineering

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/feature_engineering.py
```

19 feature üretir ve Feature Importance grafiği oluşturur:

| Kategori | Feature'lar |
|----------|-------------|
| Temporal | `day_of_week`, `hour_of_day`, `month`, `season`, `is_weekend`, `day_period` |
| Kullanıcı | `user_avg_rating`, `user_rating_count`, `user_rating_deviation`, `user_rating_variance`, `user_activity_days` |
| Film | `movie_avg_rating`, `movie_rating_count`, `movie_age`, `genre_count`, `primary_genre`, `movie_popularity_trend` |
| Etkileşim | `user_genre_avg_rating`, `rating_order`, `time_since_last_rating` |

### 8. ALS Öneri Modeli Eğitimi

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/model.py
```

Bu adım şunları yapar:
- 3 Baseline model (Global Mean, User Mean, Item Mean)
- 27 kombinasyonluk Grid Search (rank × maxIter × regParam)
- Ablation Study (her parametrenin etkisi)
- 3-Fold Cross Validation
- 6 görselleştirme + MLflow loglama

### 9. Sınıflandırma ve Regresyon Modelleri

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/classification.py
```

**4 Sınıflandırma Modeli** (rating ≥ 3.5 → Beğendi / < 3.5 → Beğenmedi):

| Model | Metrikler |
|-------|-----------|
| Logistic Regression | Accuracy, F1, Precision, Recall, AUC-ROC |
| Random Forest Classifier | ↑ |
| GBT Classifier | ↑ |
| Decision Tree Classifier | ↑ |

**4 Regresyon Modeli** (rating tahmini):

| Model | Metrikler |
|-------|-----------|
| Linear Regression | RMSE, MAE, R², MSE |
| Decision Tree Regressor | ↑ |
| Random Forest Regressor | ↑ |
| GBT Regressor | ↑ |

### 10. Genişletilmiş Değerlendirme Metrikleri

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/extended_metrics.py
```

Bu adım şunları hesaplar:
- **ALS Regresyon:** RMSE, MAE, MSE, R²
- **Öneri Sistemi:** Precision@5, Precision@10, Recall@5, Recall@10, NDCG@10
- **Sınıflandırma:** Weighted/Macro/Micro Precision, Recall, F1, AUC-ROC, Confusion Matrix

### 11. MLflow'a Model Loglama

```bash
docker exec -it spark-master \
  spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  /app/spark/log_model.py
```

---

## Arayüzler

| Servis       | URL                        | Açıklama |
|--------------|----------------------------|----------|
| Spark Web UI | http://localhost:8081       | Spark job ve stage izleme |
| MLflow UI    | http://localhost:5000       | Model takip, metrik karşılaştırma, artifact'lar |

---

## Üretilen Görselleştirmeler

Tüm grafikler `./delta/eda/` dizininde PNG olarak oluşur:

| # | Dosya | Açıklama | Kaynak |
|---|-------|----------|--------|
| 01 | `01_rating_distribution.png` | Rating dağılımı (0.5-5.0) | eda.py |
| 02 | `02_rating_trend_by_year.png` | Yıllara göre rating sayısı | eda.py |
| 03 | `03_top_20_movies.png` | En çok puanlanan 20 film | eda.py |
| 04 | `04_highest_rated_movies.png` | En yüksek puanlı 20 film | eda.py |
| 05 | `05_user_activity_distribution.png` | Kullanıcı aktivite dağılımı | eda.py |
| 06 | `06_genre_avg_rating.png` | Tür bazında ortalama rating | eda.py |
| 07 | `07_monthly_heatmap.png` | Aylık rating yoğunluğu | eda.py |
| 08 | `08_avg_rating_trend.png` | Ortalama rating zaman trendi | eda.py |
| 09 | `09_gridsearch_heatmap.png` | Grid Search ısı haritası | model.py |
| 10 | `10_gridsearch_top10.png` | En iyi 10 Grid Search sonucu | model.py |
| 11 | `11_ablation_study.png` | Parametre etkisi analizi | model.py |
| 12 | `12_model_comparison.png` | Baseline vs ALS karşılaştırma | model.py |
| 13 | `13_rmse_vs_train_time.png` | RMSE vs eğitim süresi | model.py |
| 14 | `14_rank_regparam_lines.png` | Rank-regParam ilişkisi | model.py |
| 15 | `15_feature_importance.png` | Feature önem sıralaması | feature_engineering.py |
| 16 | `16_classification_comparison.png` | 4 sınıflandırma modeli karşılaştırma | classification.py |
| 17 | `17_regression_comparison.png` | 4 regresyon modeli karşılaştırma | classification.py |
| 18 | `18_auc_roc_comparison.png` | AUC-ROC karşılaştırması | classification.py |
| 19 | `19_training_time_comparison.png` | Model eğitim süreleri | classification.py |
| 20 | `20_best_models_summary.png` | En iyi modeller özeti | classification.py |
| 21 | `21_confusion_matrix.png` | Confusion Matrix heatmap | extended_metrics.py |
| 22 | `22_als_extended_metrics.png` | ALS RMSE/MAE/MSE/R² | extended_metrics.py |
| 23 | `23_recommendation_metrics.png` | Precision@K, Recall@K, NDCG | extended_metrics.py |
| 24 | `24_classification_extended_metrics.png` | Weighted/Macro/Micro karşılaştırma | extended_metrics.py |
| 25 | `25_all_metrics_summary.png` | Tüm metriklerin özet tablosu | extended_metrics.py |

Ön işleme raporları `./delta/preprocessing_report/` dizinindedir:

| Dosya | Açıklama |
|-------|----------|
| `preprocessing_summary.png` | Veri kalitesi sorunları + önce/sonra karşılaştırma |
| `preprocessing_iqr_stats.png` | Rating IQR istatistikleri |

---

## Sorun Giderme

### Docker Servisleri Başlamıyor

**Belirti:** `docker-compose up -d` sonrası servisler `Exited` durumunda.

```bash
# Logları kontrol et
docker-compose logs zookeeper
docker-compose logs kafka
```

**Çözüm:**
- Docker Desktop'ta en az **6 GB RAM** ayırdığınızdan emin olun
- Port çakışması varsa (`8081`, `9092`, `5000`) ilgili programları kapatın
- `docker-compose down && docker-compose up -d` ile yeniden başlatın

### Kafka Producer Bağlanamıyor

**Belirti:** `NoBrokersAvailable` hatası.

**Çözüm:**
- Kafka'nın tamamen başlamasını bekleyin (~30 saniye)
- `docker-compose ps` ile Kafka'nın `healthy` durumda olduğunu doğrulayın
- Producer otomatik 5 kez yeniden deneyecektir

### Spark Consumer "Package Not Found" Hatası

**Belirti:** `ClassNotFoundException` veya `Package not found`.

**Çözüm:** `--packages` parametresini kontrol edin:
```bash
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0
```

### Delta Lake Okuma Hatası

**Belirti:** `AnalysisException: Path does not exist`.

**Çözüm:**
- Consumer'ın çalışıp veri yazmasını bekleyin
- `docker exec -it spark-master ls /app/delta/ratings/` ile dizini kontrol edin
- İlk micro-batch'in tamamlanması ~10 saniye sürer

### MLflow UI Açılmıyor

**Belirti:** `http://localhost:5000` erişilemiyor.

**Çözüm:**
```bash
docker-compose logs mlflow
# Pip kurulumu devam ediyorsa bekleyin (~1 dakika)
```

### Bellek Hatası (OutOfMemoryError)

**Belirti:** Spark job `java.lang.OutOfMemoryError` ile çöküyor.

**Çözüm:**
- Docker Desktop RAM'ini artırın (8 GB önerilir)
- `classification.py` ve `feature_engineering.py` varsayılan olarak örneklem alır
- `docker-compose.yml`'de `SPARK_WORKER_MEMORY` değerini artırın

### Windows'ta Path Hatası

**Belirti:** `FileNotFoundError` veya yanlış path.

**Çözüm:**
- Proje klasörünü kısa bir yola taşıyın (`C:\projects\movielens\`)
- Windows path'lerinde Türkçe karakter olmamasına dikkat edin

---

## Pipeline Çalıştırma Sırası (Özet)

**Tüm süreci tek seferde otomatik başlatmak için:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**Adım adım manuel çalıştırmak için:**
```
1. docker-compose up -d              # Servisleri başlat
2. (Kafka producer otomatik başlar)   # ratings.csv → Kafka
3. spark-submit consumer.py           # Kafka → Delta Lake
4. spark-submit preprocessing.py      # Veri temizleme
5. spark-submit eda.py                # Keşifsel analiz
6. spark-submit etl.py                # ETL dönüşümleri
7. spark-submit feature_engineering.py # Feature üretimi
8. spark-submit model.py              # ALS model eğitimi
9. spark-submit classification.py     # Sınıflandırma + Regresyon
10. spark-submit extended_metrics.py  # Genişletilmiş metrikler
11. spark-submit advanced_visualizations.py # Gelişmiş görselleştirmeler
12. spark-submit log_model.py         # MLflow loglama
```

---

## Katkıda Bulunanlar

- **Arzu** — Docker kurulumu, Spark Structured Streaming, ETL, ALS modeli, EDA, Feature Engineering
- **Ayaz** — Kafka producer, MLflow loglama, Veri Ön İşleme, Sınıflandırma/Regresyon, README

---

## Lisans

Bu proje eğitim amaçlıdır. MovieLens veri seti [GroupLens Research](https://grouplens.org/) tarafından sağlanmaktadır.
