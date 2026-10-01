import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# Configuração da página
st.set_page_config(
    page_title="NexusAudit AI Enterprise",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CSS CUSTOMIZADO: DESIGN SYSTEM EXECUTIVO / CYBERSECURITY
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Estilização Geral do Fundo e Tipografia */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Ocultar elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header do Projeto */
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 32px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* Cartões Métrica / KPI customizados */
    .kpi-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background-color: #3b82f6;
    }
    .kpi-card.danger::before { background-color: #ef4444; }
    .kpi-card.warning::before { background-color: #f59e0b; }
    .kpi-card.success::before { background-color: #10b981; }
    
    .kpi-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 8px;
    }
    
    /* Sidebar customizada */
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid #1e293b;
    }

    /* Footer / Assinatura do Autor */
    .custom-footer {
        margin-top: 40px;
        padding: 16px 0;
        border-top: 1px solid #1e293b;
        text-align: center;
        color: #64748b;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    .author-name {
        color: #3b82f6;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER PERSONALIZADO
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">NexusAudit AI Enterprise</div>
    <div class="main-subtitle">Plataforma Preditiva de Auditoria Contínua & Inteligência de Riscos Financeiros</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# BARRA LATERAL: Configurações de Auditoria
# ---------------------------------------------------------
st.sidebar.markdown("### Parâmetros do Modelo")

taxa_contaminacao = st.sidebar.slider(
    "Sensibilidade de Anomalia (%)",
    min_value=1.0,
    max_value=15.0,
    value=5.0,
    step=0.5
) / 100.0

n_estimadores = st.sidebar.slider(
    "Árvores de Decisão (Isolation Forest)",
    min_value=50,
    max_value=300,
    value=100,
    step=25
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Fonte de Dados")
usar_dados_sinteticos = st.sidebar.checkbox("Usar massa de dados sintéticos", value=True)

# ---------------------------------------------------------
# PROCESSAMENTO E MODELAGEM DE DADOS
# ---------------------------------------------------------
@st.cache_data
def gerar_dados_transacoes(n_amostras=1200):
    np.random.seed(42)
    valor_normal = np.random.normal(loc=250.0, scale=80.0, size=int(n_amostras * 0.95))
    frequencia_normal = np.random.poisson(lam=3, size=int(n_amostras * 0.95))
    distancia_ip = np.random.exponential(scale=15.0, size=int(n_amostras * 0.95))
    
    valor_anomalo = np.random.uniform(high=15000.0, low=3000.0, size=int(n_amostras * 0.05))
    frequencia_anomala = np.random.poisson(lam=18, size=int(n_amostras * 0.05))
    distancia_ip_anomala = np.random.uniform(high=5000.0, low=800.0, size=int(n_amostras * 0.05))
    
    valores = np.concatenate([valor_normal, valor_anomalo])
    frequencias = np.concatenate([frequencia_normal, frequencia_anomala])
    distancias = np.concatenate([distancia_ip, distancia_ip_anomala])
    
    df_temp = pd.DataFrame({
        'ID_Transacao': [f'TXN-{10000 + i}' for i in range(len(valores))],
        'Valor_R$': np.round(np.abs(valores), 2),
        'Frequencia_24h': frequencias,
        'Distancia_Origem_KM': np.round(np.abs(distancias), 1),
        'Canal': np.random.choice(['Web Banking', 'Mobile App', 'API Core', 'POS Terminal'], size=len(valores))
    })
    return df_temp.sample(frac=1, random_state=42).reset_index(drop=True)

if usar_dados_sinteticos:
    df = gerar_dados_transacoes(1200)
else:
    uploaded_file = st.sidebar.file_uploader("Envie o log CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Aguardando upload de arquivo CSV...")
        st.stop()

# Execução do Machine Learning
features = ['Valor_R$', 'Frequencia_24h', 'Distancia_Origem_KM']
X = df[features]

modelo = IsolationForest(
    n_estimators=n_estimadores,
    contamination=taxa_contaminacao,
    random_state=42
)

df['Predicao'] = modelo.fit_predict(X)
df['Score_Anomalia'] = modelo.decision_function(X)
df['Classificacao'] = df['Predicao'].map({1: 'Aprovada', -1: 'Suspeita de Fraude'})

# Métricas Principais
total_transacoes = len(df)
total_anomalias = (df['Classificacao'] == 'Suspeita de Fraude').sum()
volume_suspeito = df[df['Classificacao'] == 'Suspeita de Fraude']['Valor_R$'].sum()
taxa_risco = (total_anomalias / total_transacoes) * 100

# ---------------------------------------------------------
# PAINEL DE KPIs CUSTOMIZADOS (CARDS HTML/CSS)
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class="kpi-card success">
    <div class="kpi-label">Volume Processado</div>
    <div class="kpi-value">{total_transacoes:,}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="kpi-card danger">
    <div class="kpi-label">Transações Suspeitas</div>
    <div class="kpi-value">{total_anomalias}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="kpi-card warning">
    <div class="kpi-label">Capital sob Risco</div>
    <div class="kpi-value">R$ {volume_suspeito:,.2f}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="kpi-card danger">
    <div class="kpi-label">Taxa de Fraude</div>
    <div class="kpi-value">{taxa_risco:.2f}%</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# GRÁFICOS COM PALETA CYBERSECURITY (PLOTLY CUSTOMIZADO)
# ---------------------------------------------------------
g1, g2 = st.columns([6, 4])

plot_layout = dict(
    paper_bgcolor='#0f172a',
    plot_bgcolor='#0f172a',
    font=dict(color='#94a3b8', family="Inter"),
    margin=dict(l=20, r=20, t=40, b=20)
)

with g1:
    st.markdown("##### Dispersão de Anomalias (Valor vs. Distância Geográfica)")
    fig_scatter = px.scatter(
        df,
        x='Valor_R$',
        y='Distancia_Origem_KM',
        color='Classificacao',
        size='Frequencia_24h',
        color_discrete_map={'Aprovada': '#10b981', 'Suspeita de Fraude': '#ef4444'},
        hover_data=['ID_Transacao', 'Canal']
    )
    fig_scatter.update_layout(**plot_layout, height=380)
    fig_scatter.update_xaxes(gridcolor='#1e293b', title="Valor da Operação (R$)")
    fig_scatter.update_yaxes(gridcolor='#1e293b', title="Distância do IP (KM)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with g2:
    st.markdown("##### Distribuição de Incidentes por Canal")
    df_canal = df[df['Classificacao'] == 'Suspeita de Fraude'].groupby('Canal').size().reset_index(name='Anomalias')
    fig_bar = px.bar(
        df_canal,
        x='Canal',
        y='Anomalias',
        color_discrete_sequence=['#3b82f6']
    )
    fig_bar.update_layout(**plot_layout, height=380, showlegend=False)
    fig_bar.update_xaxes(gridcolor='#1e293b', title="")
    fig_bar.update_yaxes(gridcolor='#1e293b', title="Qtd. de Anomalias")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# TABELA ESTILIZADA DE AUDITORIA
# ---------------------------------------------------------
st.markdown("##### Matriz Detalhada de Transações Alvo de Auditoria")

df_suspeitas = df[df['Classificacao'] == 'Suspeita de Fraude'].sort_values(by='Score_Anomalia', ascending=True)

st.dataframe(
    df_suspeitas[[
        'ID_Transacao', 'Valor_R$', 'Frequencia_24h', 
        'Distancia_Origem_KM', 'Canal', 'Score_Anomalia'
    ]].style.format({
        'Valor_R$': 'R$ {:,.2f}',
        'Distancia_Origem_KM': '{:,.1f} km',
        'Score_Anomalia': '{:.4f}'
    }),
    use_container_width=True,
    height=300
)

# ---------------------------------------------------------
# RODAPÉ COM ASSINATURA DE AUTORIA
# ---------------------------------------------------------
st.markdown("""
<div class="custom-footer">
    Desenvolvido por <span class="author-name">Thales Câmara</span> | Engenharia de Software & Financial Data Science
</div>
""", unsafe_allow_html=True)