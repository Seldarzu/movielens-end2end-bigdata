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
    st.header("🏆 MLflow Model Sonuçları")
    st.markdown("**MovieLens 2.19M rating** üzerinde eğitilen tüm modellerin karşılaştırmalı analizi.")

    tab_als, tab_cls, tab_reg, tab_scale = st.tabs([
        "🤖 ALS Öneri Modeli",
        "🏷️ Sınıflandırma",
        "📈 Regresyon",
        "📏 Veri Ölçeği Analizi",
    ])

    # ── ALS TAB ──────────────────────────────────────────────────────────────
    with tab_als:
        st.markdown("### ALS Grid Search — 8 Kombinasyon (rank × maxIter × regParam)")

        # Özet metrik kutuları
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 En İyi RMSE", "0.8121", delta="-0.2505 vs baseline", delta_color="normal")
        c2.metric("📐 En İyi rank", "20")
        c3.metric("🔁 En İyi maxIter", "20")
        c4.metric("🎛️ En İyi regParam", "0.1")

        st.markdown("---")

        # Baseline karşılaştırması
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Baseline Karşılaştırması")
            baseline_df = pd.DataFrame({
                "Model": ["Global Mean", "Item Mean", "User Mean", "ALS (En İyi)"],
                "RMSE": [1.0626, 0.9669, 0.9562, 0.8121],
                "Tür": ["Baseline", "Baseline", "Baseline", "ALS"],
            })
            colors = {"Baseline": "#f38ba8", "ALS": "#a6e3a1"}
            fig_base = px.bar(
                baseline_df, x="Model", y="RMSE", color="Tür",
                color_discrete_map=colors,
                title="ALS vs Baseline RMSE Karşılaştırması",
                text="RMSE",
            )
            fig_base.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_base.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", showlegend=True,
                yaxis=dict(range=[0, 1.2]),
            )
            st.plotly_chart(fig_base, use_container_width=True)

        with col2:
            st.markdown("#### Grid Search Sonuçları")
            gs_df = pd.DataFrame({
                "Kombinasyon": [
                    "r10_i10_reg0.01","r10_i10_reg0.1","r10_i20_reg0.01","r10_i20_reg0.1",
                    "r20_i10_reg0.01","r20_i10_reg0.1","r20_i20_reg0.01","r20_i20_reg0.1",
                ],
                "RMSE": [0.8570, 0.8206, 0.8542, 0.8156, 0.8925, 0.8182, 0.8920, 0.8120],
                "Süre (sn)": [23.99, 17.93, 28.14, 34.59, 31.28, 31.55, 58.88, 51.10],
            })
            fig_gs = px.scatter(
                gs_df, x="Süre (sn)", y="RMSE", text="Kombinasyon",
                color="RMSE", color_continuous_scale="RdYlGn_r",
                title="Grid Search: RMSE vs Eğitim Süresi",
                size=[10]*8,
            )
            fig_gs.update_traces(textposition="top center", textfont_size=9)
            fig_gs.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4",
            )
            st.plotly_chart(fig_gs, use_container_width=True)

        st.markdown("#### Cross Validation Sonuçları")
        cv_cols = st.columns(3)
        cv_cols[0].metric("CV Ortalama RMSE", "0.8372")
        cv_cols[1].metric("Test RMSE", "0.8121")
        cv_cols[2].metric("Eğitim Süresi", "133.3 sn")

        st.markdown("#### Genişletilmiş ALS Metrikleri")
        ext_cols = st.columns(4)
        ext_cols[0].metric("RMSE", "0.8121")
        ext_cols[1].metric("MAE", "0.6280")
        ext_cols[2].metric("MSE", "0.6594")
        ext_cols[3].metric("R²", "0.4144")

    # ── SINIFLANDIRMA TAB ─────────────────────────────────────────────────────
    with tab_cls:
        st.markdown("### Sınıflandırma — Rating ≥ 4.0 → 'Beğendi' (4 Model + 3-Fold CV)")

        cls_data = {
            "Model": ["Logistic Regression", "Random Forest", "GBT", "Decision Tree", "Majority Baseline"],
            "F1 Score": [0.7287, 0.7224, 0.7181, 0.7157, 0.3389],
            "Accuracy": [0.7287, 0.7224, 0.7185, 0.7159, None],
            "AUC-ROC": [0.8067, 0.7950, 0.7973, 0.6874, None],
            "CV F1 (Ort.)": [0.7254, 0.7201, 0.7205, 0.7169, None],
            "Eğitim (sn)": [42.6, 147.4, 130.0, 18.4, None],
        }
        cls_df = pd.DataFrame(cls_data)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### F1 Score Karşılaştırması")
            fig_f1 = px.bar(
                cls_df[cls_df["Model"] != "Majority Baseline"],
                x="Model", y="F1 Score",
                color="F1 Score", color_continuous_scale="Viridis",
                text="F1 Score", title="F1 Score (Yüksek = İyi)",
            )
            fig_f1.add_hline(y=0.3389, line_dash="dash", line_color="#f38ba8",
                             annotation_text="Baseline F1: 0.3389")
            fig_f1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_f1.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", yaxis=dict(range=[0, 0.82]),
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with col2:
            st.markdown("#### AUC-ROC Karşılaştırması")
            fig_auc = px.bar(
                cls_df[cls_df["Model"] != "Majority Baseline"],
                x="Model", y="AUC-ROC",
                color="AUC-ROC", color_continuous_scale="Blues",
                text="AUC-ROC", title="AUC-ROC Skoru",
            )
            fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_auc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", yaxis=dict(range=[0, 0.9]),
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig_auc, use_container_width=True)

        st.markdown("#### Test vs CV F1 Karşılaştırması (Overfitting Kontrolü)")
        fig_cv = go.Figure()
        models_cls = cls_df[cls_df["CV F1 (Ort.)"].notna()]["Model"].tolist()
        test_f1 = [0.7287, 0.7224, 0.7181, 0.7157]
        cv_f1   = [0.7254, 0.7201, 0.7205, 0.7169]
        fig_cv.add_trace(go.Bar(name="Test F1", x=models_cls, y=test_f1,
                                marker_color="#89b4fa", text=[f"{v:.4f}" for v in test_f1],
                                textposition="outside"))
        fig_cv.add_trace(go.Bar(name="CV F1 (3-Fold)", x=models_cls, y=cv_f1,
                                marker_color="#cba6f7", text=[f"{v:.4f}" for v in cv_f1],
                                textposition="outside"))
        fig_cv.update_layout(
            barmode="group", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd6f4", yaxis=dict(range=[0.70, 0.74]),
            title="Test F1 vs 3-Fold CV F1 — Overfitting Yok",
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        st.markdown("#### Detaylı Metrik Tablosu")
        st.dataframe(
            cls_df.style.format(na_rep="—", precision=4)
                .highlight_max(subset=["F1 Score", "Accuracy", "AUC-ROC"], color="#1e4620")
                .highlight_min(subset=["F1 Score", "Accuracy", "AUC-ROC"], color="#4a1010"),
            use_container_width=True, hide_index=True,
        )

    # ── REGRESYON TAB ─────────────────────────────────────────────────────────
    with tab_reg:
        st.markdown("### Regresyon — Rating Tahmin (4 Model + 3-Fold CV)")

        reg_data = {
            "Model": ["Linear Regression", "Random Forest", "GBT", "Decision Tree", "Mean Baseline"],
            "RMSE": [0.8266, 0.8440, 0.8605, 0.8707, 1.0597],
            "MAE":  [0.6326, 0.6490, 0.6691, 0.6716, 0.8401],
            "R²":   [0.3916, 0.3656, 0.3406, 0.3249, None],
            "CV RMSE": [0.8318, 0.8498, 0.8655, 0.8768, None],
            "Eğitim (sn)": [11.6, 146.0, 126.6, 17.2, None],
        }
        reg_df = pd.DataFrame(reg_data)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### RMSE Karşılaştırması")
            fig_rmse = px.bar(
                reg_df, x="Model", y="RMSE",
                color="RMSE", color_continuous_scale="RdYlGn_r",
                text="RMSE", title="RMSE (Düşük = İyi)",
            )
            fig_rmse.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_rmse.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", yaxis=dict(range=[0, 1.2]),
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col2:
            st.markdown("#### R² Skoru")
            fig_r2 = px.bar(
                reg_df[reg_df["R²"].notna()], x="Model", y="R²",
                color="R²", color_continuous_scale="Greens",
                text="R²", title="R² (Yüksek = İyi)",
            )
            fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_r2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", yaxis=dict(range=[0, 0.45]),
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("#### Baseline İyileştirme Analizi")
        improv_cols = st.columns(4)
        improvements = [
            ("Linear Regression", 0.8266, 22.0),
            ("Random Forest", 0.8440, 20.4),
            ("GBT", 0.8605, 18.8),
            ("Decision Tree", 0.8707, 17.8),
        ]
        for i, (name, rmse, pct) in enumerate(improvements):
            improv_cols[i].metric(name, f"RMSE: {rmse}", delta=f"-{pct:.1f}% vs baseline", delta_color="normal")

        st.markdown("#### Detaylı Metrik Tablosu")
        st.dataframe(
            reg_df.style.format(na_rep="—", precision=4)
                .highlight_min(subset=["RMSE", "MAE", "CV RMSE"], color="#1e4620")
                .highlight_max(subset=["RMSE", "MAE", "CV RMSE"], color="#4a1010"),
            use_container_width=True, hide_index=True,
        )

    # ── ÖLÇEK ANALİZİ TAB ────────────────────────────────────────────────────
    with tab_scale:
        st.markdown("### Veri Ölçeği Analizi — 100K → 500K → 1M → 2.2M")
        st.caption("Aynı model farklı veri boyutlarında eğitilerek ölçeklenebilirlik test edildi.")

        scale_df = pd.DataFrame({
            "Veri Boyutu": ["100K", "500K", "1M", "2.2M (Tümü)"],
            "n_rows": [100_000, 500_000, 1_000_000, 2_194_029],
            "LR RMSE": [0.8334, 0.8299, 0.8293, 0.8266],
            "RF RMSE": [0.8512, 0.8462, 0.8468, 0.8440],
            "LR Süre (sn)": [8.31, 7.28, 10.21, 11.37],
            "RF Süre (sn)": [13.25, 42.30, 79.89, 144.58],
            "LR R²": [0.3885, 0.3891, 0.3911, 0.3916],
            "RF R²": [0.3621, 0.3647, 0.3651, 0.3656],
        })

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### RMSE vs Veri Boyutu")
            fig_sc_rmse = go.Figure()
            fig_sc_rmse.add_trace(go.Scatter(
                x=scale_df["Veri Boyutu"], y=scale_df["LR RMSE"],
                name="Linear Regression", mode="lines+markers+text",
                marker=dict(size=10, color="#89b4fa"),
                text=[f"{v:.4f}" for v in scale_df["LR RMSE"]],
                textposition="top center",
            ))
            fig_sc_rmse.add_trace(go.Scatter(
                x=scale_df["Veri Boyutu"], y=scale_df["RF RMSE"],
                name="Random Forest", mode="lines+markers+text",
                marker=dict(size=10, color="#cba6f7"),
                text=[f"{v:.4f}" for v in scale_df["RF RMSE"]],
                textposition="top center",
            ))
            fig_sc_rmse.update_layout(
                title="Veri Ölçeği Arttıkça RMSE Düşüyor",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", yaxis=dict(range=[0.80, 0.87]),
            )
            st.plotly_chart(fig_sc_rmse, use_container_width=True)

        with col2:
            st.markdown("#### Eğitim Süresi vs Veri Boyutu")
            fig_sc_time = go.Figure()
            fig_sc_time.add_trace(go.Scatter(
                x=scale_df["Veri Boyutu"], y=scale_df["LR Süre (sn)"],
                name="Linear Regression", mode="lines+markers",
                marker=dict(size=10, color="#89b4fa"),
            ))
            fig_sc_time.add_trace(go.Scatter(
                x=scale_df["Veri Boyutu"], y=scale_df["RF Süre (sn)"],
                name="Random Forest", mode="lines+markers",
                marker=dict(size=10, color="#cba6f7"),
            ))
            fig_sc_time.update_layout(
                title="Veri Boyutu → Eğitim Süresi Ölçeklenebilirliği",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cdd6f4", yaxis_title="Saniye",
            )
            st.plotly_chart(fig_sc_time, use_container_width=True)

        st.markdown("#### Ölçek Analizi Özet Tablosu")
        st.dataframe(scale_df.drop(columns=["n_rows"]).style.format(precision=4),
                     use_container_width=True, hide_index=True)

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
