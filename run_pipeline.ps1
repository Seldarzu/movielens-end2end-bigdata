Write-Host "Stopping and cleaning up containers..."
docker-compose down
docker rm -f zookeeper kafka spark-master spark-worker mlflow streamlit-dashboard kafka-producer 2>$null

Write-Host "Starting containers..."
docker-compose up --build -d

Write-Host "Waiting 60 seconds for Kafka and Producer to initialize and start streaming..."
Start-Sleep -Seconds 60

Write-Host "Fixing permissions for Spark Ivy cache and MLflow artifacts..."
docker exec -u root spark-master bash -c "mkdir -p /home/spark/.ivy2/cache /mlflow/artifacts && chmod -R 777 /home/spark /mlflow"

Write-Host "Starting Spark Consumer (collecting data for 60 seconds)..."
docker exec -d spark-master /opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 /app/spark/consumer.py
Start-Sleep -Seconds 65
docker exec spark-master bash -c "pkill -f consumer.py"
Start-Sleep -Seconds 5

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
    Write-Host "Running $script ..."
    docker exec spark-master /opt/spark/bin/spark-submit --packages $deltaPackages /app/spark/$script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: $script encountered an error or exited with non-zero code."
    }
}

Write-Host "======================================================"
Write-Host "Pipeline execution finished!"
Write-Host "Streamlit Dashboard: http://localhost:8501"
Write-Host "MLFlow Tracking: http://localhost:5000"
Write-Host "======================================================"
