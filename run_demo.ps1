Write-Host "======================================================"
Write-Host "  DEMO MODU - Delta verileri korunuyor"
Write-Host "  Onceki pipeline sonuclari (MLflow, grafikler) sakli"
Write-Host "======================================================"

Write-Host "Containerlar baslatiliyor (temizleme YOK)..."
docker-compose up -d

Write-Host "60 saniye bekleniyor (Kafka ve servisler hazir olsun)..."
Start-Sleep -Seconds 60

Write-Host "Izinler duzenleniyor..."
docker exec -u root spark-master bash -c "mkdir -p /home/spark/.ivy2/cache /mlflow/artifacts && chmod -R 777 /home/spark /mlflow"

Write-Host "Eski checkpoint temizleniyor (cakisma onlenir)..."
docker exec spark-master bash -c "rm -rf /app/delta/checkpoints/ratings"

Write-Host ""
Write-Host "======================================================"
Write-Host "  Hazir! Tarayicide ac:"
Write-Host "  Streamlit  --> http://localhost:8501  (Canli Mod ac)"
Write-Host "  MLflow     --> http://localhost:5000"
Write-Host "  Spark UI   --> http://localhost:8081"
Write-Host "======================================================"
Write-Host ""
Write-Host "Kafka -> Delta Lake streaming basliyor..."
Write-Host "(Durdurmak icin Ctrl+C)"
Write-Host ""

docker exec spark-master /opt/spark/bin/spark-submit `
    --master spark://spark-master:7077 `
    --total-executor-cores 1 `
    --executor-memory 512m `
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 `
    /app/spark/consumer.py
