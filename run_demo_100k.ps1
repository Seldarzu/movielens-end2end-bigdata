Write-Host "======================================================"
Write-Host "  DEMO MODU - 100K Satir ile Hizli Pipeline"
Write-Host "  Consumer 100K satira ulasinca otomatik durur"
Write-Host "  Pipeline toplam ~10-15 dakikada tamamlanir"
Write-Host "======================================================"

# ── 1. Eski veriyi temizle ────────────────────────────────────────────────
Write-Host ""
Write-Host "[1/5] Eski veriler temizleniyor..."
docker exec spark-master bash -c "rm -rf /app/delta/ratings /app/delta/ratings_cleaned /app/delta/ratings_features /app/delta/checkpoints /app/delta/eda /app/delta/models /app/delta/scale_results /app/delta/preprocessing_report /app/delta/als_model_best"
docker exec spark-master bash -c "echo 'CREATE TABLE IF NOT EXISTS runs (run_uuid TEXT);' | sqlite3 /mlflow/mlflow.db 2>/dev/null; python3 -c \"import sqlite3; c=sqlite3.connect('/mlflow/mlflow.db'); c.execute('DELETE FROM runs'); c.execute('DELETE FROM metrics'); c.execute('DELETE FROM params'); c.execute('DELETE FROM tags'); c.execute('DELETE FROM experiments WHERE name != chr(68)||chr(101)||chr(102)||chr(97)||chr(117)||chr(108)||chr(116)'); c.execute('DELETE FROM registered_models'); c.execute('DELETE FROM model_versions'); c.commit(); print('MLflow temizlendi')\""
Write-Host "   Temizleme tamamlandi."

# ── 2. Containerlar ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/5] Servisler kontrol ediliyor..."
$running = docker ps --format "{{.Names}}" | Select-String "spark-master"
if (-not $running) {
    docker-compose up -d
    Write-Host "   60 saniye bekleniyor..."
    Start-Sleep -Seconds 60
} else {
    Write-Host "   Servisler zaten calisiyor."
}

docker exec -u root spark-master bash -c "mkdir -p /home/spark/.ivy2/cache /mlflow/artifacts && chmod -R 777 /home/spark /mlflow"
docker exec spark-master bash -c "rm -rf /app/delta/checkpoints/ratings"

# ── 3. Consumer'i arkaplanda baslat ──────────────────────────────────────
Write-Host ""
Write-Host "[3/5] Kafka -> Delta Lake streaming basliyor..."
Write-Host "   Tarayicida izle --> http://localhost:8501 (Canli Mod ac)"
Write-Host ""

$consumerJob = Start-Job -ScriptBlock {
    docker exec spark-master /opt/spark/bin/spark-submit `
        --master spark://spark-master:7077 `
        --total-executor-cores 1 `
        --executor-memory 512m `
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 `
        /app/spark/consumer.py
}

# ── 4. 100K satira ulasinca durdur ────────────────────────────────────────
Write-Host "[4/5] 100.000 satir bekleniyor (her 10 saniyede kontrol)..."
$target = 100000
$count  = 0
$dots   = 0

while ($count -lt $target) {
    Start-Sleep -Seconds 10

    $result = docker exec spark-master python3 -c @"
try:
    from deltalake import DeltaTable
    import os
    if os.path.exists('/app/delta/ratings'):
        dt = DeltaTable('/app/delta/ratings')
        aa = dt.get_add_actions(flatten=True)
        print(int(aa['num_records'].sum()) if 'num_records' in aa.columns else 0)
    else:
        print(0)
except:
    print(0)
"@

    $count = [int]($result.Trim())
    $pct   = [math]::Min([math]::Round($count / $target * 100, 1), 100)
    $dots++
    Write-Host ("   {0,7:N0} / {1:N0} satir  ({2}%)" -f $count, $target, $pct)

    # Consumer caktiysa cik
    if ($consumerJob.State -eq "Completed" -or $consumerJob.State -eq "Failed") {
        Write-Host "   Consumer beklenmedik sekilde durdu!" -ForegroundColor Yellow
        break
    }
}

# Consumer'i durdur
Write-Host ""
Write-Host "   100K satirina ulasildi! Consumer durduruluyor..."
docker exec spark-master bash -c "pkill -f consumer.py 2>/dev/null; pkill -f SparkSubmit 2>/dev/null; echo done"
Stop-Job $consumerJob -ErrorAction SilentlyContinue
Remove-Job $consumerJob -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5

$finalCount = docker exec spark-master python3 -c @"
try:
    from deltalake import DeltaTable
    dt = DeltaTable('/app/delta/ratings')
    aa = dt.get_add_actions(flatten=True)
    print(int(aa['num_records'].sum()))
except:
    print('?')
"@
Write-Host "   Delta Lake'teki toplam satir: $($finalCount.Trim())"

# ── 5. Pipeline calistir ──────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================================"
Write-Host "[5/5] Pipeline basliyor..."
Write-Host "   MLflow  --> http://localhost:5000"
Write-Host "   Spark   --> http://localhost:8081"
Write-Host "======================================================"
Write-Host ""

.\run_scripts.ps1
