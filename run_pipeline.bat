@echo off
setlocal

set SPARK=docker exec spark-master /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/ivy2
set DELTA=io.delta:delta-core_2.12:2.4.0
set KAFKA_PKG=org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0

echo.
echo ================================================================
echo  MovieLens 25M -- Uc Tan Uce Buyuk Veri Pipeline
echo ================================================================
echo.

:: ── ADIM 1: Consumer - Kafka'dan Delta Lake'e veri yaz (30 sn) ──
echo [1/9] Consumer baslatiliyor (30 saniye veri toplanacak)...
docker exec -d spark-master /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/ivy2 --packages %KAFKA_PKG% /app/spark/consumer.py
echo      Veri akisi icin 30 saniye bekleniyor...
timeout /t 30 /nobreak >nul
docker exec spark-master bash -c "pkill -f consumer.py" 2>nul
echo      Consumer durduruldu - veri Delta Lake'e yazildi.
echo.

:: ── ADIM 2: EDA - Ham veri uzerinde kesifsel analiz ─────────────
echo [2/9] EDA calistiriliyor (ham veri uzerinde)...
%SPARK% --packages %DELTA% /app/spark/eda.py
if %errorlevel% neq 0 ( echo HATA: EDA basarisiz. & exit /b 1 )
echo      EDA tamamlandi.
echo.

:: ── ADIM 3: Preprocessing - EDA bulgularina gore temizle ────────
echo [3/9] Veri on isleme calistiriliyor...
%SPARK% --packages %DELTA% /app/spark/preprocessing.py
if %errorlevel% neq 0 ( echo HATA: Preprocessing basarisiz. & exit /b 1 )
echo      On isleme tamamlandi.
echo.

:: ── ADIM 4: ETL ─────────────────────────────────────────────────
echo [4/9] ETL calistiriliyor...
%SPARK% --packages %DELTA% /app/spark/etl.py
if %errorlevel% neq 0 ( echo HATA: ETL basarisiz. & exit /b 1 )
echo      ETL tamamlandi.
echo.

:: ── ADIM 5: Feature Engineering ─────────────────────────────────
echo [5/9] Feature Engineering calistiriliyor (19 feature)...
%SPARK% --packages %DELTA% /app/spark/feature_engineering.py
if %errorlevel% neq 0 ( echo HATA: Feature Engineering basarisiz. & exit /b 1 )
echo      Feature Engineering tamamlandi.
echo.

:: ── ADIM 6: ALS Model ───────────────────────────────────────────
echo [6/9] ALS model egitiliyor (Grid Search + Ablation + CV)...
%SPARK% --packages %DELTA% /app/spark/model.py
if %errorlevel% neq 0 ( echo HATA: ALS model basarisiz. & exit /b 1 )
echo      ALS model tamamlandi.
echo.

:: ── ADIM 7: Siniflandirma + Regresyon ───────────────────────────
echo [7/9] Siniflandirma ve Regresyon modelleri egitiliyor (8 model)...
%SPARK% --packages %DELTA% /app/spark/classification.py
if %errorlevel% neq 0 ( echo HATA: Classification basarisiz. & exit /b 1 )
echo      Siniflandirma + Regresyon tamamlandi.
echo.

:: ── ADIM 8: Genisletilmis metrikler ─────────────────────────────
echo [8/9] Genisletilmis metrikler hesaplaniyor...
%SPARK% --packages %DELTA% /app/spark/extended_metrics.py
if %errorlevel% neq 0 ( echo HATA: Extended metrics basarisiz. & exit /b 1 )
echo      Metrikler tamamlandi.
echo.

:: ── ADIM 9: Gelismis gorsellestirmeler + MLflow loglama ─────────
echo [9/9] Gelismis gorsellestirmeler ve MLflow loglama...
%SPARK% --packages %DELTA% /app/spark/advanced_visualizations.py
if %errorlevel% neq 0 ( echo HATA: Advanced viz basarisiz. & exit /b 1 )
%SPARK% --packages %DELTA% /app/spark/log_model.py
if %errorlevel% neq 0 ( echo HATA: MLflow loglama basarisiz. & exit /b 1 )
echo      Tamamlandi.
echo.

echo ================================================================
echo  PIPELINE TAMAMLANDI!
echo.
echo  Dashboard  : http://localhost:8501
echo  Spark UI   : http://localhost:8081
echo  MLflow UI  : http://localhost:5000
echo ================================================================
echo.

endlocal
