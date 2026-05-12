#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# MovieLens 25M — Uçtan Uca Pipeline Orkestrasyon Scripti
# Takım: Arzu (Kıdemli Mühendis) & Ayaz (Junior Mühendis)
#
# Bu script tüm pipeline adımlarını sırayla çalıştırır:
#   1. Docker servislerini başlat
#   2. Kafka producer'ın veri göndermesini bekle
#   3. Spark consumer ile Delta Lake'e yaz
#   4. Veri ön işleme
#   5. EDA
#   6. ETL
#   7. Feature Engineering
#   8. ALS Model eğitimi
#   9. Sınıflandırma + Regresyon modelleri
#  10. Genişletilmiş metrikler
#  11. Gelişmiş görselleştirmeler
#  12. MLflow loglama
#
# Kullanım:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Not: Docker Desktop'un çalışıyor olması gerekir.
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Hata olursa dur

# ─── Renkli Çıktı ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'  # Renk sıfırlama

# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

print_header() {
    echo ""
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "${CYAN}[ADIM $1/$TOTAL_STEPS]${NC} ${GREEN}$2${NC}"
}

print_success() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}  ✗ HATA: $1${NC}"
    echo -e "${RED}  Pipeline durduruldu.${NC}"
    exit 1
}

wait_seconds() {
    echo -e "${YELLOW}  ⏳ $1 saniye bekleniyor...${NC}"
    sleep $1
}

TOTAL_STEPS=12
SPARK_PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0"
DELTA_PACKAGES="io.delta:delta-core_2.12:2.4.0"

# ─── Spark Submit Wrapper ────────────────────────────────────────────────────

run_spark_job() {
    local script=$1
    local packages=$2
    local description=$3

    echo -e "${BLUE}  → spark-submit $script${NC}"

    docker exec spark-master \
        spark-submit \
        --packages "$packages" \
        "/app/spark/$script" 2>&1 | while IFS= read -r line; do
            # Sadece INFO ve ERROR loglarını göster
            if echo "$line" | grep -q "\[INFO\]\|ERROR\|tamamlandı\|başlıyor\|SONUÇ\|==="; then
                echo "    $line"
            fi
        done

    if [ $? -eq 0 ]; then
        print_success "$description tamamlandı"
    else
        print_error "$description başarısız oldu"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE BAŞLANGIÇ
# ═══════════════════════════════════════════════════════════════════════════════

print_header "MovieLens 25M — Uçtan Uca Büyük Veri Pipeline'ı"
echo -e "${CYAN}Başlangıç zamanı: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""

# ─── ADIM 1: Docker Servislerini Başlat ──────────────────────────────────────

print_step 1 "Docker servisleri başlatılıyor..."

docker-compose up -d 2>&1

if [ $? -ne 0 ]; then
    print_error "Docker servisleri başlatılamadı. Docker Desktop çalışıyor mu?"
fi

print_success "Docker servisleri başlatıldı"
wait_seconds 10

# Servislerin durumunu kontrol et
echo -e "${BLUE}  Servis durumları:${NC}"
docker-compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || docker-compose ps

# ─── ADIM 2: Kafka'nın Hazır Olmasını Bekle ──────────────────────────────────

print_step 2 "Kafka broker hazır olması bekleniyor..."

MAX_RETRIES=12
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; then
        print_success "Kafka broker hazır"
        break
    fi
    RETRY=$((RETRY + 1))
    echo -e "${YELLOW}  Deneme $RETRY/$MAX_RETRIES — 10 saniye bekleniyor...${NC}"
    sleep 10
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    print_error "Kafka broker $MAX_RETRIES denemede hazır olmadı"
fi

# ─── ADIM 3: Kafka Producer Durumunu Kontrol Et ──────────────────────────────

print_step 3 "Kafka producer kontrol ediliyor..."

PRODUCER_STATUS=$(docker inspect -f '{{.State.Running}}' kafka-producer 2>/dev/null || echo "false")
if [ "$PRODUCER_STATUS" = "true" ]; then
    print_success "Kafka producer çalışıyor (otomatik başlatıldı)"
else
    print_warning "Producer çalışmıyor, yeniden başlatılıyor..."
    docker-compose restart kafka-producer
    wait_seconds 5
fi

# Veri akışının başlamasını bekle
echo -e "${YELLOW}  ⏳ Veri akışının başlaması için 30 saniye bekleniyor...${NC}"
echo -e "${YELLOW}  (Producer ratings.csv'yi Kafka'ya stream ediyor)${NC}"
sleep 30

# ─── ADIM 4: Spark Consumer ─────────────────────────────────────────────────

print_step 4 "Spark consumer başlatılıyor (60 saniye veri toplanacak)..."

# Consumer'ı arka planda başlat, belirli süre sonra durdur
docker exec -d spark-master \
    spark-submit \
    --packages "$SPARK_PACKAGES" \
    /app/spark/consumer.py

print_success "Consumer başlatıldı"
echo -e "${YELLOW}  ⏳ Delta Lake'e veri yazılması için 60 saniye bekleniyor...${NC}"
sleep 60

# Consumer'ı durdur (Ctrl+C simülasyonu)
docker exec spark-master bash -c "pkill -f consumer.py" 2>/dev/null || true
print_success "Consumer durduruldu, veri Delta Lake'e yazıldı"
wait_seconds 5

# ─── ADIM 5: Veri Ön İşleme ─────────────────────────────────────────────────

print_step 5 "Veri ön işleme başlatılıyor..."
run_spark_job "preprocessing.py" "$DELTA_PACKAGES" "Veri ön işleme"
wait_seconds 3

# ─── ADIM 6: EDA ────────────────────────────────────────────────────────────

print_step 6 "Keşifsel veri analizi (EDA) başlatılıyor..."
run_spark_job "eda.py" "$DELTA_PACKAGES" "EDA (8 grafik)"
wait_seconds 3

# ─── ADIM 7: ETL + Feature Engineering ──────────────────────────────────────

print_step 7 "ETL ve Feature Engineering başlatılıyor..."
run_spark_job "etl.py" "$DELTA_PACKAGES" "ETL dönüşümleri"
wait_seconds 2
run_spark_job "feature_engineering.py" "$DELTA_PACKAGES" "Feature Engineering (19 feature)"
wait_seconds 3

# ─── ADIM 8: ALS Model Eğitimi ──────────────────────────────────────────────

print_step 8 "ALS öneri modeli eğitiliyor (Grid Search + CV)..."
run_spark_job "model.py" "$DELTA_PACKAGES" "ALS model eğitimi"
wait_seconds 3

# ─── ADIM 9: Sınıflandırma + Regresyon ──────────────────────────────────────

print_step 9 "Sınıflandırma ve regresyon modelleri eğitiliyor..."
run_spark_job "classification.py" "$DELTA_PACKAGES" "Sınıflandırma + Regresyon (8 model)"
wait_seconds 3

# ─── ADIM 10: Genişletilmiş Metrikler ───────────────────────────────────────

print_step 10 "Genişletilmiş değerlendirme metrikleri hesaplanıyor..."
run_spark_job "extended_metrics.py" "$DELTA_PACKAGES" "Genişletilmiş metrikler"
wait_seconds 3

# ─── ADIM 11: Gelişmiş Görselleştirmeler ────────────────────────────────────

print_step 11 "Gelişmiş görselleştirmeler oluşturuluyor..."
run_spark_job "advanced_visualizations.py" "$DELTA_PACKAGES" "Gelişmiş görselleştirmeler (5 grafik)"
wait_seconds 3

# ─── ADIM 12: MLflow Loglama ────────────────────────────────────────────────

print_step 12 "MLflow'a model loglama yapılıyor..."
run_spark_job "log_model.py" "$DELTA_PACKAGES" "MLflow loglama"

# ═══════════════════════════════════════════════════════════════════════════════
# SONUÇ
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
print_header "PIPELINE TAMAMLANDI"
echo ""
echo -e "${GREEN}  Bitiş zamanı: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""
echo -e "${CYAN}  Arayüzler:${NC}"
echo -e "    Spark Web UI : ${BLUE}http://localhost:8081${NC}"
echo -e "    MLflow UI    : ${BLUE}http://localhost:5000${NC}"
echo ""
echo -e "${CYAN}  Çıktılar:${NC}"
echo -e "    Delta Lake   : ./delta/"
echo -e "    Grafikler    : ./delta/eda/ (30 PNG)"
echo -e "    Ön İşleme    : ./delta/preprocessing_report/"
echo -e "    Modeller     : ./delta/models/"
echo ""
echo -e "${CYAN}  Servisleri durdurmak için:${NC}"
echo -e "    docker-compose down"
echo ""
