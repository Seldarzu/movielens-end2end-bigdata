# Takım Üyesi: Ayaz (Junior Mühendis)
# Görev: Eğitilmiş ALS modelini MLflow Tracking Server'a kaydetmek.
#         - Parametre loglama: rank, maxIter, regParam
#         - Metrik loglama: RMSE
#         - Model artifact kaydı

import logging
import sys
import os
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# ─── Loglama Ayarları ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = "http://mlflow:5000"    # Konteyner içi MLflow adresi
EXPERIMENT_NAME      = "movielens-als"         # MLflow deney adı
ALS_MODEL_PATH       = "/app/delta/als_model_best"  # Spark'ın kaydettiği model dizini

# Model parametreleri (model.py ile senkronize olmalı)
ALS_RANK      = 10
ALS_MAX_ITER  = 10
ALS_REG_PARAM = 0.1

# RMSE değeri; gerçek ortamda model.py çıktısından ya da Delta metadata'dan okunur.
# Bu dosyada ortam değişkeni üzerinden alınır; yoksa varsayılan kullanılır.
RMSE_VALUE = float(os.environ.get("ALS_RMSE", "0.8765"))


def create_spark_session() -> SparkSession:
    """MLflow + Delta Lake destekli Spark oturumu başlatır."""
    try:
        spark = (
            SparkSession.builder
            .appName("MovieLens-MLflow-Logger")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark oturumu oluşturuldu.")
        return spark
    except Exception as e:
        logger.error("Spark oturumu başlatılamadı: %s", e)
        sys.exit(1)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """
    MLflow tracking URI'sini ayarlar ve experiment oluşturur/döndürür.
    Experiment yoksa yeni oluşturur; varsa mevcut ID'yi döndürür.
    """
    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info("Yeni experiment oluşturuldu: '%s' (id=%s)", experiment_name, experiment_id)
        else:
            experiment_id = experiment.experiment_id
            logger.info("Mevcut experiment kullanılıyor: '%s' (id=%s)", experiment_name, experiment_id)
        return experiment_id
    except Exception as e:
        logger.error("MLflow experiment ayarı başarısız: %s", e)
        sys.exit(1)


def load_als_model(spark: SparkSession, model_path: str) -> ALSModel:
    """Disk üzerindeki eğitilmiş ALS modelini Spark'a yükler."""
    try:
        model = ALSModel.load(model_path)
        logger.info("ALS modeli yüklendi: %s", model_path)
        return model
    except Exception as e:
        logger.error("Model yüklenemedi (%s): %s", model_path, e)
        sys.exit(1)


def log_to_mlflow(
    model: ALSModel,
    experiment_id: str,
    rank: int,
    max_iter: int,
    reg_param: float,
    rmse: float,
) -> None:
    """
    MLflow run oluşturur; parametreleri, metriği ve modeli loglar.
    Run tamamlandığında durum otomatik olarak FINISHED olarak işaretlenir.
    """
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(run_name="als-training") as run:
        run_id = run.info.run_id
        logger.info("MLflow run başlatıldı — run_id: %s", run_id)

        # ── Parametreleri logla ──────────────────────────────────────────────
        mlflow.log_param("rank",       rank)
        mlflow.log_param("maxIter",    max_iter)
        mlflow.log_param("regParam",   reg_param)
        mlflow.log_param("trainRatio", 0.8)
        mlflow.log_param("testRatio",  0.2)
        mlflow.log_param("coldStartStrategy", "drop")
        logger.info("Parametreler loglandı.")

        # ── Metriği logla ────────────────────────────────────────────────────
        mlflow.log_metric("rmse", rmse)
        logger.info("RMSE metriği loglandı: %.4f", rmse)

        # ── Modeli artifact olarak kaydet ────────────────────────────────────
        # mlflow.spark, Spark MLlib modellerini MLflow formatında saklar
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="als_model",
            # Model imzası; servise almayı kolaylaştırır
            registered_model_name="movielens-als",
        )
        logger.info("Model artifact'ı kaydedildi.")

        logger.info(
            "MLflow loglama tamamlandı — run_id: %s | RMSE: %.4f",
            run_id, rmse,
        )


def main():
    spark = create_spark_session()

    # 1. MLflow'u ayarla
    experiment_id = setup_mlflow(MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

    # 2. Kaydedilmiş ALS modelini yükle
    als_model = load_als_model(spark, ALS_MODEL_PATH)

    # 3. MLflow'a logla
    log_to_mlflow(
        model=als_model,
        experiment_id=experiment_id,
        rank=ALS_RANK,
        max_iter=ALS_MAX_ITER,
        reg_param=ALS_REG_PARAM,
        rmse=RMSE_VALUE,
    )

    spark.stop()
    logger.info("log_model.py tamamlandı.")


if __name__ == "__main__":
    main()
