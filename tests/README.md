# Test Kılavuzu

Bu dizin, MovieLens pipeline'ının **Spark bağımsız** birim testlerini içerir.
Testler Docker veya Spark gerektirmez; host makinede doğrudan çalışır.

## Kapsam

| Test Dosyası | Kapsam |
|---|---|
| `test_config.py` | Path sabitleri, `require_path()`, `check_mlflow()` |
| `test_eda_utils.py` | Adaptif birim formatı (`_fmt`) — K / M etiketleri |
| `test_preprocessing_logic.py` | Rating aralığı, duplikat, null, hyperactive kullanıcı |
| `test_classification_logic.py` | Binary etiket, overfitting eşiği, confusion matrix, F1 |
| `test_pipeline_integration.py` | Dosya varlığı, docker-compose servisleri, path tutarlılığı |

**Toplam: 93 test**

## Kurulum

```bash
pip install pytest pytest-mock pandas
```

## Çalıştırma

```bash
# Tüm testler
pytest tests/ -v

# Sadece bir dosya
pytest tests/test_config.py -v

# Kısa çıktı
pytest tests/ -q

# Belirli bir test
pytest tests/test_classification_logic.py::test_dynamic_overfit_threshold -v
```

## Notlar

- Spark MLlib testleri (model eğitimi, feature engineering) Docker içinde çalıştırılmalıdır.
- Bu testler iş mantığını (business logic) doğrular; Spark API'sini doğrulamaz.
- CI ortamında `pytest tests/ --tb=short -q` komutu önerilir.
