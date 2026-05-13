Write-Host "Delta Lake'te veri bekleniyor (maks 5 dakika)..."
$maxWait = 300
$waited = 0
while ($waited -lt $maxWait) {
    $result = docker exec spark-master bash -c "test -d /app/delta/ratings/_delta_log && echo exists" 2>$null
    if ($result -eq "exists") {
        Write-Host "Veri bulundu! ($waited saniye beklendi) Devam ediliyor..."
        Start-Sleep -Seconds 15
        break
    }
    Write-Host "Veri bekleniyor... ($waited / $maxWait saniye)"
    Start-Sleep -Seconds 15
    $waited += 15
}
if ($waited -ge $maxWait) {
    Write-Host "HATA: $maxWait saniyede veri gelmedi. Consumer calisiyor mu?"
    exit 1
}

$deltaPackages = "io.delta:delta-core_2.12:2.4.0"

$scripts = @(
    "eda.py",
    "preprocessing.py",
    "etl.py",
    "feature_engineering.py",
    "model.py",
    "classification.py",
    "scale_analysis.py",
    "extended_metrics.py",
    "advanced_visualizations.py",
    "log_model.py"
)

foreach ($script in $scripts) {
    Write-Host "Running $script ..."
    docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --total-executor-cores 2 --executor-memory 1g --packages $deltaPackages /app/spark/$script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: $script encountered an error or exited with non-zero code."
    }
}

Write-Host "======================================================"
Write-Host "Pipeline execution finished!"
Write-Host "Streamlit Dashboard: http://localhost:8501"
Write-Host "MLFlow Tracking:     http://localhost:5000"
Write-Host "======================================================"
