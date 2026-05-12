import streamlit as st
import pandas as pd
import time
import os
import json
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from deltalake import DeltaTable

# ─── Sayfa Ayarları ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovieLens BigData Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema renkleri (Dracula vari)
st.markdown("""
<style>
    .reportview-container {
        background: #1e1e2e;
    }
    .main {
        background-color: #1e1e2e;
        color: #cdd6f4;
    }
    h1, h2, h3 {
        color: #cba6f7;
    }
    .stMetric-value {
        color: #a6e3a1 !important;
    }
    .stMetric-label {
        color: #89b4fa !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sabitler ───────────────────────────────────────────────────────────────
DELTA_RATINGS_PATH = "/app/delta/ratings"
MLFLOW_DB_PATH     = "/mlflow/mlflow.db"
EDA_DIR            = "/app/delta/eda"

# ─── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────

@st.cache_data(ttl=5) # 5 saniyede bir cache temizle
def get_total_ratings():
    try:
        if os.path.exists(DELTA_RATINGS_PATH):
            dt = DeltaTable(DELTA_RATINGS_PATH)
            # Daha hızlı count için add dosyalarındaki istatistikleri toplayabiliriz ama
            # pandas'a çevirip saymak veya sadece version kontrolü yapmak daha basit
            # df = dt.to_pandas()
            # return len(df)
            
            # Delta file metadata history sayma:
            add_actions = dt.get_add_actions(flatten=True)
            if not add_actions.empty and 'num_records' in add_actions.columns:
                return int(add_actions['num_records'].sum())
            else:
                return len(dt.to_pandas())
        return 0
    except Exception as e:
        return 0

@st.cache_data(ttl=30)
def get_mlflow_metrics():
    """MLflow SQLite DB'sinden model metriklerini çek."""
    try:
        if not os.path.exists(MLFLOW_DB_PATH):
            return pd.DataFrame()
            
        conn = sqlite3.connect(MLFLOW_DB_PATH)
        query = """
            SELECT r.run_uuid, r.name, m.key, m.value
            FROM runs r
            JOIN metrics m ON r.run_uuid = m.run_uuid
            WHERE r.status = 'FINISHED'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
            
        # Pivot table yap
        pivot_df = df.pivot_table(index='name', columns='key', values='value', aggfunc='first').reset_index()
        return pivot_df
    except Exception as e:
        return pd.DataFrame()

def load_image(filename):
    path = os.path.join(EDA_DIR, filename)
    if os.path.exists(path):
        return path
    return None

# ─── Ana Düzen ──────────────────────────────────────────────────────────────

st.title("🎬 MovieLens 25M — Gerçek Zamanlı Pipeline Dashboard")
st.markdown("Büyük Veri Mimarisi: **Kafka → Spark Streaming → Delta Lake → MLflow**")

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Apache_Kafka_logo.svg/1200px-Apache_Kafka_logo.svg.png", width=100)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/1200px-Apache_Spark_logo.svg.png", width=100)
st.sidebar.title("Navigasyon")
page = st.sidebar.radio("Sayfalar:", [
    "📡 Canlı Veri Akışı", 
    "🏆 Model Metrikleri (MLflow)", 
    "📊 Görselleştirmeler (EDA)"
])

# ─── SAYFA 1: Canlı Veri Akışı ──────────────────────────────────────────────
if page == "📡 Canlı Veri Akışı":
    st.header("Gerçek Zamanlı Veri İzleme")
    
    col1, col2, col3 = st.columns(3)
    
    # Canlı Metric Component
    metric_placeholder = col1.empty()
    speed_placeholder = col2.empty()
    status_placeholder = col3.empty()
    
    chart_placeholder = st.empty()
    
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.timestamps = []
        st.session_state.last_count = 0
    
    # Oto Yenileme Mekanizması (Gerçek zamanlı simülasyonu)
    count = get_total_ratings()
    
    # Hız hesapla
    diff = count - st.session_state.last_count
    st.session_state.last_count = count
    
    if count > 0:
        st.session_state.history.append(count)
        st.session_state.timestamps.append(pd.Timestamp.now())
        # Sadece son 50 noktayı tut
        if len(st.session_state.history) > 50:
            st.session_state.history.pop(0)
            st.session_state.timestamps.pop(0)
    
    metric_placeholder.metric(label="Toplam İşlenen Rating", value=f"{count:,}")
    speed_placeholder.metric(label="Akış Hızı (rating/sn)", value=f"{diff:,}" if count > 0 else "0", delta=diff)
    
    if diff > 0:
        status_placeholder.success("Kafka'dan veri akıyor 🟢")
    elif count > 0:
        status_placeholder.warning("Akış durdu / Bekliyor 🟡")
    else:
        status_placeholder.error("Delta Lake boş / Kafka çalışmıyor 🔴")
        
    if len(st.session_state.history) > 1:
        df_chart = pd.DataFrame({
            "Zaman": st.session_state.timestamps,
            "Rating Sayısı": st.session_state.history
        })
        fig = px.area(df_chart, x="Zaman", y="Rating Sayısı", 
                      title="Gerçek Zamanlı Veri Büyümesi",
                      color_discrete_sequence=["#cba6f7"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4")
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Dashboard yenileme butonu veya loop
    st.markdown("---")
    st.markdown("*Tablo saniyede bir otomatik güncellenir. Otomatik güncellemeyi açmak için aşağıdaki butona basın:*")
    if st.button("🔄 Anlık Akışı İzle (Live Mode)"):
        while True:
            time.sleep(2)
            st.rerun()

# ─── SAYFA 2: Model Metrikleri ──────────────────────────────────────────────
elif page == "🏆 Model Metrikleri (MLflow)":
    st.header("MLflow Model Başarı Metrikleri")
    
    df_metrics = get_mlflow_metrics()
    
    if df_metrics.empty:
        st.info("Henüz MLflow veritabanında metrik bulunamadı. Lütfen modelleri eğitin.")
    else:
        st.markdown("### Tüm Modellerin Karşılaştırması")
        # Kolonları filtrele ve düzenle
        display_df = df_metrics.copy()
        display_df = display_df.set_index('name')
        st.dataframe(display_df.style.highlight_max(axis=0, color='#89b4fa').highlight_min(axis=0, color='#f38ba8'))
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sınıflandırma (Accuracy)")
            if 'accuracy' in display_df.columns:
                fig = px.bar(display_df.dropna(subset=['accuracy']).reset_index(), 
                             x='name', y='accuracy', color='name',
                             title="Modellerin Doğruluk Oranları")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4")
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            st.markdown("#### Regresyon (RMSE)")
            if 'rmse' in display_df.columns:
                fig = px.bar(display_df.dropna(subset=['rmse']).reset_index(), 
                             x='name', y='rmse', color='name',
                             title="Modellerin Hata Oranları (Düşük daha iyi)")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4")
                st.plotly_chart(fig, use_container_width=True)

# ─── SAYFA 3: Görselleştirmeler ─────────────────────────────────────────────
elif page == "📊 Görselleştirmeler (EDA)":
    st.header("Keşifsel Veri Analizi ve Model Grafikleri")
    
    tab1, tab2, tab3 = st.tabs(["Ön İşleme & EDA", "Model Karşılaştırmaları", "Gelişmiş Grafikler"])
    
    with tab1:
        st.markdown("### Rating ve Trend Analizleri")
        col1, col2 = st.columns(2)
        with col1:
            if load_image("01_rating_distribution.png"): st.image(load_image("01_rating_distribution.png"))
            if load_image("07_monthly_heatmap.png"): st.image(load_image("07_monthly_heatmap.png"))
        with col2:
            if load_image("08_avg_rating_trend.png"): st.image(load_image("08_avg_rating_trend.png"))
            if load_image("15_feature_importance.png"): st.image(load_image("15_feature_importance.png"))
            
    with tab2:
        st.markdown("### Modellerin Kıyaslanması")
        col1, col2 = st.columns(2)
        with col1:
            if load_image("16_classification_comparison.png"): st.image(load_image("16_classification_comparison.png"))
            if load_image("18_auc_roc_comparison.png"): st.image(load_image("18_auc_roc_comparison.png"))
        with col2:
            if load_image("17_regression_comparison.png"): st.image(load_image("17_regression_comparison.png"))
            if load_image("20_best_models_summary.png"): st.image(load_image("20_best_models_summary.png"))
            
    with tab3:
        st.markdown("### Karmaşıklık, Hata ve Matrisler")
        if load_image("30_unified_model_comparison.png"): st.image(load_image("30_unified_model_comparison.png"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if load_image("29_all_confusion_matrices.png"): st.image(load_image("29_all_confusion_matrices.png"))
            if load_image("27_learning_curve.png"): st.image(load_image("27_learning_curve.png"))
        with col2:
            if load_image("28_residual_plots.png"): st.image(load_image("28_residual_plots.png"))
            if load_image("26_roc_curves_all_models.png"): st.image(load_image("26_roc_curves_all_models.png"))

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6c7086;'>MovieLens 25M Big Data Pipeline &copy; 2026 — Arzu & Ayaz</div>", unsafe_allow_html=True)
