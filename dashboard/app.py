import streamlit as st
import pandas as pd
import time
import os
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

st.markdown("""
<style>
    .main { background-color: #1e1e2e; color: #cdd6f4; }
    h1, h2, h3 { color: #cba6f7; }
    .stMetric-value { color: #a6e3a1 !important; }
    .stMetric-label { color: #89b4fa !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Sabitler ───────────────────────────────────────────────────────────────
DELTA_RATINGS_PATH = "/app/delta/ratings"
MLFLOW_DB_PATH     = "/mlflow/mlflow.db"
EDA_DIR            = "/app/delta/eda"

# ─── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────

@st.cache_data(ttl=3)
def get_total_ratings():
    try:
        if not os.path.exists(DELTA_RATINGS_PATH):
            return 0
        dt = DeltaTable(DELTA_RATINGS_PATH)
        add_actions = dt.get_add_actions(flatten=True)
        if not add_actions.empty and "num_records" in add_actions.columns:
            return int(add_actions["num_records"].sum())
        return len(dt.to_pandas())
    except Exception:
        return 0


@st.cache_data(ttl=5)
def get_recent_records(n=8):
    """Delta Lake'ten ilk n kaydı döndür (örnekleme)."""
    try:
        if not os.path.exists(DELTA_RATINGS_PATH):
            return pd.DataFrame()
        dt = DeltaTable(DELTA_RATINGS_PATH)
        tbl = dt.to_pyarrow_dataset()
        sample = tbl.head(n)
        df = sample.to_pandas() if hasattr(sample, "to_pandas") else pd.DataFrame(sample.to_pydict())
        cols = [c for c in ["userId", "movieId", "rating", "timestamp"] if c in df.columns]
        return df[cols] if cols else df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def get_mlflow_metrics():
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
        pivot_df = df.pivot_table(index="name", columns="key", values="value", aggfunc="first").reset_index()
        return pivot_df
    except Exception:
        return pd.DataFrame()


def load_image(filename):
    path = os.path.join(EDA_DIR, filename)
    return path if os.path.exists(path) else None


# ─── Ana Düzen ──────────────────────────────────────────────────────────────
st.title("🎬 MovieLens 25M — Gerçek Zamanlı Pipeline Dashboard")
st.markdown("Büyük Veri Mimarisi: **Kafka → Spark Streaming → Delta Lake → MLflow**")

st.sidebar.markdown("### Kafka · Spark · Delta Lake")
auto_refresh = st.sidebar.toggle("🔄 Canlı Mod (3 sn)", value=False)
st.sidebar.caption("Açıkken sayfa her 3 saniyede bir güncellenir.")
st.sidebar.title("Navigasyon")
page = st.sidebar.radio("Sayfalar:", [
    "📡 Canlı Veri Akışı",
    "🏆 Model Metrikleri (MLflow)",
    "📊 Görselleştirmeler (EDA)",
])

# ─── SAYFA 1: Canlı Veri Akışı ──────────────────────────────────────────────
if page == "📡 Canlı Veri Akışı":
    TOTAL_TARGET = 25_000_000

    st.header("📡 Gerçek Zamanlı Kafka → Delta Lake Akışı")

    # Session state başlat
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.timestamps = []
        st.session_state.speed_history = []
        st.session_state.last_count = 0
        st.session_state.last_time = time.time()

    # Anlık veri al
    count = get_total_ratings()
    now = time.time()
    elapsed_sec = max(now - st.session_state.last_time, 0.001)
    diff = max(count - st.session_state.last_count, 0)
    speed = int(diff / elapsed_sec) if diff > 0 else 0

    st.session_state.last_count = count
    st.session_state.last_time = now

    if count > 0:
        st.session_state.history.append(count)
        st.session_state.timestamps.append(pd.Timestamp.now())
        st.session_state.speed_history.append(speed)
        if len(st.session_state.history) > 60:
            st.session_state.history.pop(0)
            st.session_state.timestamps.pop(0)
            st.session_state.speed_history.pop(0)

    progress = min(count / TOTAL_TARGET, 1.0)

    # Durum bandı
    if diff > 0:
        st.success(f"🟢  Kafka akışı aktif  —  son güncelleme: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    elif count > 0:
        st.warning(f"🟡  Akış durdu / bekleniyor  —  toplam {count:,} kayıt Delta Lake'te")
    else:
        st.error("🔴  Delta Lake boş  —  Kafka producer çalışıyor mu?")

    # İlerleme çubuğu
    st.markdown(f"#### Yükleme İlerlemesi: **{progress*100:.2f}%**")
    st.progress(progress)
    st.caption(f"{count:,} / {TOTAL_TARGET:,} rating  ({TOTAL_TARGET - count:,} kaldı)")

    # 4 metrik kutusu
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Yüklenen Rating", f"{count:,}", delta=f"+{diff:,}" if diff > 0 else None)
    c2.metric("🎯 Hedef", f"{TOTAL_TARGET:,}")
    c3.metric("⚡ Akış Hızı", f"{speed:,} /sn")
    if speed > 0:
        eta_sec = int((TOTAL_TARGET - count) / speed)
        eta_str = f"{eta_sec // 3600}s {(eta_sec % 3600) // 60}d" if eta_sec > 3600 else f"{eta_sec // 60}d {eta_sec % 60}sn"
    else:
        eta_str = "—"
    c4.metric("⏱ Tahmini Süre", eta_str)

    st.markdown("---")

    # İki grafik yan yana
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### Kümülatif Yükleme Grafiği")
        if len(st.session_state.history) > 1:
            df_cum = pd.DataFrame({
                "Zaman": st.session_state.timestamps,
                "Rating Sayısı": st.session_state.history,
            })
            fig_cum = px.area(df_cum, x="Zaman", y="Rating Sayısı",
                              color_discrete_sequence=["#cba6f7"])
            fig_cum.add_hline(y=TOTAL_TARGET, line_dash="dash", line_color="#f38ba8",
                              annotation_text="Hedef 25M", annotation_position="top left")
            fig_cum.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="", yaxis_title="Rating",
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        else:
            st.info("Veri akışı başladığında grafik oluşacak...")

    with col_right:
        st.markdown("##### Anlık Akış Hızı (rating/sn)")
        if len(st.session_state.speed_history) > 1:
            df_spd = pd.DataFrame({
                "Zaman": st.session_state.timestamps[-len(st.session_state.speed_history):],
                "Hız": st.session_state.speed_history,
            })
            fig_spd = px.bar(df_spd, x="Zaman", y="Hız",
                             color_discrete_sequence=["#89b4fa"])
            fig_spd.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="", yaxis_title="rating/sn",
            )
            st.plotly_chart(fig_spd, use_container_width=True)
        else:
            st.info("Akış hızı verisi birikmesini bekliyor...")

    # Son kayıtlar tablosu
    st.markdown("---")
    st.markdown("##### 🗄️ Delta Lake'ten Örnek Kayıtlar")
    df_recent = get_recent_records(8)
    if not df_recent.empty:
        if "timestamp" in df_recent.columns:
            df_recent["timestamp"] = pd.to_datetime(df_recent["timestamp"], unit="s", errors="coerce")
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    else:
        st.caption("Henüz Delta Lake'te veri yok.")

    # Otomatik yenileme
    if auto_refresh:
        time.sleep(3)
        st.rerun()

# ─── SAYFA 2: Model Metrikleri ──────────────────────────────────────────────
elif page == "🏆 Model Metrikleri (MLflow)":
    st.header("MLflow Model Başarı Metrikleri")

    df_metrics = get_mlflow_metrics()

    if df_metrics.empty:
        st.info("Henüz MLflow veritabanında metrik bulunamadı. Lütfen modelleri eğitin.")
    else:
        st.markdown("### Tüm Modellerin Karşılaştırması")
        display_df = df_metrics.set_index("name")
        st.dataframe(
            display_df.style.format(na_rep="-")
                .highlight_max(axis=0, color="#89b4fa")
                .highlight_min(axis=0, color="#f38ba8"),
            use_container_width=True,
        )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Sınıflandırma (Accuracy)")
            if "accuracy" in display_df.columns:
                fig = px.bar(
                    display_df.dropna(subset=["accuracy"]).reset_index(),
                    x="name", y="accuracy", color="name",
                    title="Modellerin Doğruluk Oranları",
                )
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Regresyon (RMSE)")
            if "rmse" in display_df.columns:
                fig = px.bar(
                    display_df.dropna(subset=["rmse"]).reset_index(),
                    x="name", y="rmse", color="name",
                    title="Modellerin Hata Oranları (Düşük daha iyi)",
                )
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
            if load_image("31_overfit_check.png"): st.image(load_image("31_overfit_check.png"))
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
