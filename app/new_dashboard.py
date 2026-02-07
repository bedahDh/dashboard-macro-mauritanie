import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ========== Configuration de la page ==========
st.set_page_config(
    page_title="Dashboard macroéconomique – Mauritanie",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ========== CSS personnalisé ==========
st.markdown(
    """
    <style>
    /* Masquer l'en-tête par défaut */
    header {visibility: hidden;}

    /* Styles généraux */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Barre latérale */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #dee2e6;
    }

    /* Cartes de métriques */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #2E8B57;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }

    /* En-tête principal */
    .main-header {
        color: #2E8B57;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }

    .sub-header {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Boutons */
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        background-color: #1e6b47;
    }

    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #dee2e6;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white !important;
    }

    /* Pied de page */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        background-color: white;
        border-radius: 10px;
    }

    /* Espacement amélioré */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== En-tête principal ==========
st.markdown('<h1 class="main-header" style="text-align: center;">📊 Dashboard macroéconomique de la Mauritanie</h1>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Analyse interactive des principaux indicateurs (2000–2023) | Données : Banque Mondiale & Banque Centrale de Mauritanie</p>',
    unsafe_allow_html=True)


# ========== Chargement des données ==========
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "mauritania_macro_clean.csv"
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Fichier non trouvé : {DATA_PATH}")
        # Retourner un DataFrame vide comme fallback
        return pd.DataFrame()


df = load_data()

# Vérifier si les données sont chargées
if df.empty:
    st.warning("⚠️ Les données n'ont pas pu être chargées. Vérifiez le chemin du fichier.")
    st.stop()

# ========== Barre latérale ==========
with st.sidebar:
    st.markdown("### ⚙️ Paramètres de visualisation")
    st.markdown("---")

    # Sélection d'indicateur
    indicators = [
        "GDP",
        "GDP_growth",
        "Inflation",
        "Exchange_rate",
        "Unemployment",
        "FDI",
        "GDP_per_capita"
    ]

    indicator = st.selectbox(
        "**Choisir un indicateur principal**",
        indicators,
        help="Sélectionnez l'indicateur à analyser"
    )

    st.markdown("---")

    # Sélection de période
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())

    start_year, end_year = st.slider(
        "**Période d'analyse**",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        help="Sélectionnez la période à visualiser"
    )

    st.markdown("---")

    # Options supplémentaires
    col1, col2 = st.columns(2)
    with col1:
        normalize = st.checkbox("Normaliser", help="Normaliser les données pour comparaison")
    with col2:
        show_trend = st.checkbox("Tendance", help="Afficher la ligne de tendance")

    st.markdown("---")

    # Bouton de rafraîchissement
    if st.button("🔄 Rafraîchir les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ========== Cartes de KPI principales ==========
st.markdown("### 📈 Indicateurs Clés")

# Créer 4 colonnes pour les KPI
col1, col2, col3, col4 = st.columns(4)

# Calculer les dernières valeurs disponibles
latest_year = df["Year"].max()

with col1:
    if "GDP" in df.columns:
        latest_gdp = df[df["Year"] == latest_year]["GDP"].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6c757d;">PIB (dernière valeur)</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #2E8B57;">{latest_gdp:,.1f}</div>
            <div style="font-size: 0.8rem; color: #adb5bd;">Année {latest_year}</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if "Inflation" in df.columns:
        latest_inflation = df[df["Year"] == latest_year]["Inflation"].values[0]
        inflation_color = "#dc3545" if latest_inflation > 5 else "#28a745"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6c757d;">Inflation</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {inflation_color};">{latest_inflation:.1f}%</div>
            <div style="font-size: 0.8rem; color: #adb5bd;">Année {latest_year}</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    if "Unemployment" in df.columns:
        latest_unemp = df[df["Year"] == latest_year]["Unemployment"].values[0]
        unemp_color = "#dc3545" if latest_unemp > 10 else "#28a745"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6c757d;">Chômage</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {unemp_color};">{latest_unemp:.1f}%</div>
            <div style="font-size: 0.8rem; color: #adb5bd;">Année {latest_year}</div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    if "FDI" in df.columns:
        latest_fdi = df[df["Year"] == latest_year]["FDI"].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6c757d;">Investissement Direct Étranger</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #2E8B57;">{latest_fdi:,.1f}</div>
            <div style="font-size: 0.8rem; color: #adb5bd;">Année {latest_year}</div>
        </div>
        """, unsafe_allow_html=True)

# ========== Onglets principaux ==========
tab1, tab2, tab3 = st.tabs(["📈 Analyse d'indicateur", "🔍 Comparaison multi-indicateurs", "📋 Données et statistiques"])

with tab1:
    st.subheader(f"Évolution de {indicator} ({start_year}–{end_year})")

    # Filtrer les données
    df_filtered = df[
        (df["Year"] >= start_year) &
        (df["Year"] <= end_year)
        ][["Year", indicator]].copy()

    # Normaliser si demandé
    if normalize and indicator in df_filtered.columns:
        scaler = StandardScaler()
        df_filtered[indicator] = scaler.fit_transform(df_filtered[[indicator]])

    # Créer la visualisation avec Plotly
    if not df_filtered.empty:
        fig = go.Figure()

        # Ajouter la ligne principale
        fig.add_trace(go.Scatter(
            x=df_filtered["Year"],
            y=df_filtered[indicator],
            mode='lines+markers',
            name=indicator,
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=8, color='#FFD700')
        ))

        # Ajouter la ligne de tendance si demandée
        if show_trend and len(df_filtered) > 2:
            z = np.polyfit(df_filtered["Year"], df_filtered[indicator], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df_filtered["Year"],
                y=p(df_filtered["Year"]),
                mode='lines',
                name='Tendance linéaire',
                line=dict(color='#dc3545', width=2, dash='dash')
            ))

        # Personnaliser le layout
        fig.update_layout(
            template='plotly_white',
            height=450,
            hovermode='x unified',
            xaxis_title="Année",
            yaxis_title="Valeur normalisée" if normalize else "Valeur",
            title=f"Évolution de {indicator}",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)

        # Afficher des statistiques rapides
        if indicator in df_filtered.columns:
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                max_val = df_filtered[indicator].max()
                max_year = df_filtered[df_filtered[indicator] == max_val]['Year'].iloc[0]
                st.metric("Valeur maximale", f"{max_val:.2f}", f"en {max_year}")

            with col_stats2:
                min_val = df_filtered[indicator].min()
                min_year = df_filtered[df_filtered[indicator] == min_val]['Year'].iloc[0]
                st.metric("Valeur minimale", f"{min_val:.2f}", f"en {min_year}")

            with col_stats3:
                avg_val = df_filtered[indicator].mean()
                st.metric("Moyenne", f"{avg_val:.2f}", "sur la période")
    else:
        st.warning("Aucune donnée disponible pour la période sélectionnée.")

with tab2:
    st.subheader("Comparaison entre indicateurs")

    # Sélection multiple d'indicateurs
    selected_indicators = st.multiselect(
        "Choisir les indicateurs à comparer :",
        indicators,
        default=["GDP", "Inflation", "Unemployment"],
        help="Sélectionnez au moins deux indicateurs"
    )

    if len(selected_indicators) >= 2:
        # Filtrer les données
        df_comp = df[
            (df["Year"] >= start_year) &
            (df["Year"] <= end_year)
            ][["Year"] + selected_indicators].copy()

        # Normaliser si demandé
        if normalize:
            scaler = StandardScaler()
            df_comp[selected_indicators] = scaler.fit_transform(df_comp[selected_indicators])

        # Créer le graphique de comparaison
        fig2 = go.Figure()

        # Couleurs pour les différentes séries
        colors = ['#2E8B57', '#FFD700', '#dc3545', '#007bff', '#6f42c1']

        for i, col in enumerate(selected_indicators):
            fig2.add_trace(go.Scatter(
                x=df_comp["Year"],
                y=df_comp[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=3)
            ))

        # Personnaliser le layout
        fig2.update_layout(
            template='plotly_white',
            height=450,
            hovermode='x unified',
            xaxis_title="Année",
            yaxis_title="Valeurs normalisées" if normalize else "Valeurs",
            title="Comparaison des indicateurs sélectionnés"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Matrice de corrélation
        st.subheader("Matrice de corrélation")
        corr_matrix = df_comp[selected_indicators].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            title="Corrélations entre indicateurs"
        )

        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.info("Veuillez sélectionner au moins deux indicateurs pour la comparaison.")

with tab3:
    st.subheader("Aperçu des données")

    # Filtrer les données pour la période sélectionnée
    display_df = df[
        (df["Year"] >= start_year) &
        (df["Year"] <= end_year)
        ]

    # Afficher le tableau de données
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Year": st.column_config.NumberColumn("Année", format="%d"),
            "GDP": st.column_config.NumberColumn("PIB", format="%.2f"),
            "Inflation": st.column_config.NumberColumn("Inflation", format="%.2f%%"),
            "Unemployment": st.column_config.NumberColumn("Chômage", format="%.2f%%")
        }
    )

    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.dataframe(display_df.describe(), use_container_width=True)

    # Bouton de téléchargement
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données (CSV)",
        data=csv,
        file_name=f"donnees_mauritanie_{start_year}_{end_year}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ========== Pied de page ==========
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p><strong>📚 Sources de données :</strong> Banque Mondiale (World Bank)</p>
        <p><strong>🕐 Période couverte :</strong> 2000–2023 | <strong>Dernière mise à jour :</strong> 01-01-2023</p>
        <p><strong>📊 Projet :</strong> Visualisation de données macroéconomiques - Université de Nouakchott</p>
    </div>
    """,
    unsafe_allow_html=True
)