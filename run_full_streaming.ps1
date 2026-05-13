Write-Host "======================================================"
Write-Host "🚀 STARTING FULL DATA STREAMING PIPELINE"
Write-Host "======================================================"

Write-Host "Stopping and cleaning up containers..."
docker-compose down
docker rm -f zookeeper kafka spark-master spark-worker mlflow streamlit-dashboard kafka-producer 2>$null

Write-Host "Starting containers with MAX SPEED producer..."
# Producer'ın gecikmesiz (sıfır bekleme) çalışması için ortam değişkeni veriyoruz
$env:DELAY_SEC="0"
docker-compose up --build -d

Write-Host "Waiting 30 seconds for Kafka to initialize..."
Start-Sleep -Seconds 30

Write-Host "Fixing permissions for Spark..."
docker exec -u root spark-master bash -c "mkdir -p /home/spark/.ivy2/cache /mlflow/artifacts && chmod -R 777 /home/spark /mlflow"

Write-Host "Starting Spark Consumer (Background Streaming)..."
# Consumer arka planda SÜREKLİ çalışacak, asla durdurulmayacak!
docker exec -d spark-master /opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 /app/spark/consumer.py

Write-Host "Waiting 30 seconds for initial data to populate..."
Start-Sleep -Seconds 30

$deltaPackages = "io.delta:delta-core_2.12:2.4.0"

$scripts = @(
    "preprocessing.py",
    "eda.py",
    "etl.py",
    "feature_engineering.py",
    "model.py",
    "classification.py",
    "extended_metrics.py",
    "advanced_visualizations.py",
    "log_model.py"
)

foreach ($script in $scripts) {
    Write-Host "Running Batch Job: $script ..."
    # Batch job'lar, o an Delta Lake'te ne kadar veri varsa onunla çalışır
    docker exec spark-master /opt/spark/bin/spark-submit --packages $deltaPackages /app/spark/$script
}

Write-Host "======================================================"
Write-Host "PIPELINE STARTED SUCCESSFULLY!"
Write-Host "Kafka is currently streaming the FULL 25 Million dataset."
Write-Host "Watch the Live Stream Dashboard: http://localhost:8501"
Write-Host "======================================================"
