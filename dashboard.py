"""
Dashboard Antifraude Sicredi — Frontend em Streamlit.

Pré-requisitos:
    pip install streamlit requests

Como usar:
    1. Suba o backend:   uvicorn main:app --reload
    2. Abra o dashboard: streamlit run dashboard.py
"""

import time
import requests
import streamlit as st

from agente_antifraude import AgenteAntifraude

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Antifraude Sicredi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Estilo visual (Sicredi: verde + branco)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* ─── Variáveis de tema (modo claro como base) ─────────────────────── */
        :root {
            --af-surface:        #f6f8fa;
            --af-surface-2:      #eef2f7;
            --af-border:         #d0d7de;
            --af-border-accent:  #006633;
            --af-text:           #1f2328;
            --af-text-muted:     #57606a;
            --af-green-bg:       rgba(0, 102, 51, 0.08);
            --af-green-border:   rgba(0, 102, 51, 0.35);
            --af-risk-alto-bg:   rgba(220, 38, 38, 0.12);
            --af-risk-alto-fg:   #b91c1c;
            --af-risk-medio-bg:  rgba(217, 119, 6, 0.12);
            --af-risk-medio-fg:  #92400e;
            --af-risk-baixo-bg:  rgba(22, 101, 52, 0.10);
            --af-risk-baixo-fg:  #166534;
            --af-track-bg:       rgba(0,0,0,0.10);
            --af-code-bg:        rgba(0,0,0,0.06);
            --af-hist-alto-bg:   rgba(220, 38, 38, 0.08);
            --af-hist-medio-bg:  rgba(217, 119, 6, 0.08);
            --af-hist-baixo-bg:  rgba(22, 101, 52, 0.06);
            --af-hist-neu-bg:    rgba(0,0,0,0.04);
            --af-shap-pos:       #b91c1c;
            --af-shap-neg:       #166534;
            --af-chat-ai-bg:     rgba(0, 102, 51, 0.07);
            --af-chat-user-bg:   rgba(0,0,0,0.05);
        }

        /* ─── Modo escuro (Streamlit aplica [data-theme="dark"] no body) ───── */
        @media (prefers-color-scheme: dark) {
            :root {
                --af-surface:        rgba(255,255,255,0.05);
                --af-surface-2:      rgba(255,255,255,0.08);
                --af-border:         rgba(255,255,255,0.14);
                --af-text:           #e6edf3;
                --af-text-muted:     #8b949e;
                --af-green-bg:       rgba(0, 185, 90, 0.10);
                --af-green-border:   rgba(0, 185, 90, 0.30);
                --af-risk-alto-bg:   rgba(248, 113, 113, 0.14);
                --af-risk-alto-fg:   #fca5a5;
                --af-risk-medio-bg:  rgba(251, 191, 36, 0.14);
                --af-risk-medio-fg:  #fcd34d;
                --af-risk-baixo-bg:  rgba(74, 222, 128, 0.12);
                --af-risk-baixo-fg:  #86efac;
                --af-track-bg:       rgba(255,255,255,0.12);
                --af-code-bg:        rgba(255,255,255,0.08);
                --af-hist-alto-bg:   rgba(248, 113, 113, 0.10);
                --af-hist-medio-bg:  rgba(251, 191, 36, 0.10);
                --af-hist-baixo-bg:  rgba(74, 222, 128, 0.08);
                --af-hist-neu-bg:    rgba(255,255,255,0.04);
                --af-shap-pos:       #fca5a5;
                --af-shap-neg:       #86efac;
                --af-chat-ai-bg:     rgba(0, 185, 90, 0.10);
                --af-chat-user-bg:   rgba(255,255,255,0.06);
            }
        }

        /* Streamlit dark theme também adiciona data-theme no body */
        body[data-theme="dark"],
        [data-theme="dark"] {
            --af-surface:        rgba(255,255,255,0.05);
            --af-surface-2:      rgba(255,255,255,0.08);
            --af-border:         rgba(255,255,255,0.14);
            --af-text:           #e6edf3;
            --af-text-muted:     #8b949e;
            --af-green-bg:       rgba(0, 185, 90, 0.10);
            --af-green-border:   rgba(0, 185, 90, 0.30);
            --af-risk-alto-bg:   rgba(248, 113, 113, 0.14);
            --af-risk-alto-fg:   #fca5a5;
            --af-risk-medio-bg:  rgba(251, 191, 36, 0.14);
            --af-risk-medio-fg:  #fcd34d;
            --af-risk-baixo-bg:  rgba(74, 222, 128, 0.12);
            --af-risk-baixo-fg:  #86efac;
            --af-track-bg:       rgba(255,255,255,0.12);
            --af-code-bg:        rgba(255,255,255,0.08);
            --af-hist-alto-bg:   rgba(248, 113, 113, 0.10);
            --af-hist-medio-bg:  rgba(251, 191, 36, 0.10);
            --af-hist-baixo-bg:  rgba(74, 222, 128, 0.08);
            --af-hist-neu-bg:    rgba(255,255,255,0.04);
            --af-shap-pos:       #fca5a5;
            --af-shap-neg:       #86efac;
            --af-chat-ai-bg:     rgba(0, 185, 90, 0.10);
            --af-chat-user-bg:   rgba(255,255,255,0.06);
        }

        /* ─── Cabeçalho principal ───────────────────────────────────────────── */
        .header-bar {
            background: linear-gradient(135deg, #005229 0%, #007a3d 100%);
            color: #ffffff;
            padding: 16px 24px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow: 0 2px 8px rgba(0, 82, 41, 0.30);
        }
        .header-bar h1 {
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        .header-bar .header-sub {
            font-size: 0.82rem;
            opacity: 0.82;
            margin-top: 2px;
        }
        .header-icon { font-size: 2.2rem; line-height: 1; }

        /* ─── Badges de risco ───────────────────────────────────────────────── */
        .badge-alto {
            background: var(--af-risk-alto-bg);
            color: var(--af-risk-alto-fg);
            border: 1px solid var(--af-risk-alto-fg);
            padding: 2px 10px;
            border-radius: 12px;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .03em;
            text-transform: uppercase;
        }
        .badge-medio {
            background: var(--af-risk-medio-bg);
            color: var(--af-risk-medio-fg);
            border: 1px solid var(--af-risk-medio-fg);
            padding: 2px 10px;
            border-radius: 12px;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .03em;
            text-transform: uppercase;
        }
        .badge-baixo {
            background: var(--af-risk-baixo-bg);
            color: var(--af-risk-baixo-fg);
            border: 1px solid var(--af-risk-baixo-fg);
            padding: 2px 10px;
            border-radius: 12px;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .03em;
            text-transform: uppercase;
        }

        /* ─── Card de detalhe da transação ──────────────────────────────────── */
        .detail-card {
            background: var(--af-green-bg);
            border: 1px solid var(--af-green-border);
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 14px;
        }
        .detail-card h3 {
            margin: 0 0 14px 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--af-text);
        }
        .detail-card table { width: 100%; border-collapse: collapse; }
        .detail-card td {
            padding: 5px 10px 5px 0;
            font-size: .875rem;
            vertical-align: top;
        }
        .detail-card .td-label {
            color: var(--af-text-muted);
            width: 90px;
            white-space: nowrap;
            font-weight: 500;
        }
        .detail-card .td-value { color: var(--af-text); }
        .detail-card .td-value strong { font-weight: 600; }
        .detail-card code {
            background: var(--af-code-bg);
            border-radius: 4px;
            padding: 1px 5px;
            font-size: .82rem;
        }
        .val-destaque { color: #007a3d; font-weight: 700; font-size: 1.05rem; }
        [data-theme="dark"] .val-destaque,
        @media (prefers-color-scheme: dark) { .val-destaque { color: #4ade80; } }

        /* ─── Gauge de escore ───────────────────────────────────────────────── */
        .score-gauge {
            background: var(--af-surface);
            border: 1px solid var(--af-border);
            border-radius: 10px;
            padding: 14px 18px;
            margin: 10px 0 16px 0;
        }
        .score-gauge-header {
            display: flex;
            align-items: baseline;
            gap: 10px;
            margin-bottom: 10px;
        }
        .score-val {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.02em;
        }
        .score-nivel {
            font-size: .9rem;
            font-weight: 700;
            letter-spacing: .04em;
            text-transform: uppercase;
        }
        .score-bar-track {
            height: 12px;
            background: var(--af-track-bg);
            border-radius: 6px;
            margin-bottom: 10px;
            position: relative;
            overflow: hidden;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width .5s cubic-bezier(.4,0,.2,1);
        }
        .score-meta {
            font-size: .75rem;
            color: var(--af-text-muted);
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }
        .score-meta code {
            background: var(--af-code-bg);
            border-radius: 3px;
            padding: 1px 5px;
            font-size: .73rem;
        }

        /* ─── Seção de features de velocidade ──────────────────────────────── */
        .section-title {
            font-size: .95rem;
            font-weight: 700;
            color: var(--af-text);
            margin: 20px 0 8px 0;
            padding-bottom: 6px;
            border-bottom: 2px solid var(--af-border);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* ─── Tabela SHAP ───────────────────────────────────────────────────── */
        .shap-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
        .shap-table tr:hover td { background: var(--af-surface-2); }
        .shap-table td {
            padding: 4px 8px 4px 0;
            font-size: .82rem;
            border-bottom: 1px solid var(--af-border);
            color: var(--af-text);
        }
        .shap-table td:last-child { text-align: right; font-weight: 700; }
        .shap-table code {
            background: var(--af-code-bg);
            border-radius: 4px;
            padding: 1px 5px;
            font-size: .78rem;
            color: var(--af-text);
        }
        .shap-pos { color: var(--af-shap-pos); }
        .shap-neg { color: var(--af-shap-neg); }

        /* ─── Cards do histórico ────────────────────────────────────────────── */
        .hist-card {
            display: flex;
            align-items: center;
            gap: 8px;
            border-radius: 7px;
            padding: 8px 12px;
            margin-bottom: 6px;
            font-size: .84rem;
            border: 1px solid var(--af-border);
            color: var(--af-text);
            transition: background .15s;
        }
        .hist-card.alto  { background: var(--af-hist-alto-bg);  border-color: var(--af-risk-alto-fg); }
        .hist-card.medio { background: var(--af-hist-medio-bg); border-color: var(--af-risk-medio-fg); }
        .hist-card.baixo { background: var(--af-hist-baixo-bg); border-color: var(--af-risk-baixo-fg); }
        .hist-card.neu   { background: var(--af-hist-neu-bg);   }
        .hist-card.atual { outline: 2px solid #ef4444; outline-offset: 1px; }
        .hist-card .hist-id { font-weight: 700; white-space: nowrap; }
        .hist-card .hist-val { font-weight: 700; white-space: nowrap; }
        .hist-card .hist-ts { color: var(--af-text-muted); font-size: .78rem; white-space: nowrap; }
        .hist-card .hist-sep { color: var(--af-border); margin: 0 2px; }

        /* ─── Mensagens do chat ─────────────────────────────────────────────── */
        .chat-msg-ai {
            background: var(--af-chat-ai-bg);
            border: 1px solid var(--af-green-border);
            border-left: 3px solid #007a3d;
            padding: 10px 14px;
            border-radius: 2px 10px 10px 10px;
            margin-bottom: 10px;
            font-size: .875rem;
            color: var(--af-text);
        }
        .chat-msg-ai .chat-sender {
            font-size: .75rem;
            font-weight: 700;
            color: #007a3d;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: .05em;
        }
        .chat-msg-user {
            background: var(--af-chat-user-bg);
            border: 1px solid var(--af-border);
            border-right: 3px solid var(--af-text-muted);
            padding: 10px 14px;
            border-radius: 10px 2px 10px 10px;
            margin-bottom: 10px;
            font-size: .875rem;
            text-align: right;
            color: var(--af-text);
        }
        .chat-msg-user .chat-sender {
            font-size: .75rem;
            font-weight: 700;
            color: var(--af-text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: .05em;
        }

        /* ─── Estado vazio ──────────────────────────────────────────────────── */
        .empty-state {
            text-align: center;
            padding: 80px 20px;
            color: var(--af-text-muted);
        }
        .empty-state .empty-icon { font-size: 3.5rem; margin-bottom: 12px; }
        .empty-state h3 { font-size: 1.1rem; font-weight: 600; margin: 0 0 8px 0; color: var(--af-text); }
        .empty-state p  { font-size: .9rem; margin: 0; }

        /* ─── Sidebar ───────────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] > div { padding-top: 1rem; }

        /* ─── Botões ────────────────────────────────────────────────────────── */
        div[data-testid="stHorizontalBlock"] button { font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="header-bar">
        <span class="header-icon">🛡️</span>
        <div>
            <h1>Central de Análise Antifraude</h1>
            <div class="header-sub">Sicredi &nbsp;·&nbsp; Monitoramento em tempo real de transações suspeitas</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers para a API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_transacoes_suspeitas() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/transacoes-suspeitas", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"❌ Não foi possível conectar ao backend: {exc}")
        return []


def fetch_historico(client_id: str) -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/cliente/{client_id}/historico", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def executar_acao_api(transacao_id: int, tipo_acao: str, justificativa: str) -> dict | None:
    payload = {"tipo_acao": tipo_acao, "justificativa": justificativa}
    try:
        r = requests.post(
            f"{API_BASE}/transacao/{transacao_id}/executar-acao",
            json=payload,
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Erro ao executar ação: {exc}")
        return None


# ---------------------------------------------------------------------------
# Gauge de escore de fraude
# ---------------------------------------------------------------------------

def _cor_escore(escore: float) -> str:
    if escore >= 0.40:
        return "#ef4444"   # vermelho
    if escore >= 0.15:
        return "#f59e0b"   # âmbar
    return "#22c55e"       # verde


def render_score_gauge(escore: float, nivel: str) -> None:
    """Renderiza a barra de probabilidade de fraude com cor proporcional ao risco."""
    pct     = round(escore * 100, 1)
    cor     = _cor_escore(escore)
    largura = round(escore * 100, 2)
    rotulo  = {"alto": "ALTO", "medio": "MÉDIO", "baixo": "BAIXO"}.get(nivel, nivel.upper())

    # Ícone e label descritivo por nível
    icone_nivel = {"alto": "🔴", "medio": "🟡", "baixo": "🟢"}.get(nivel, "⚪")

    st.markdown(
        f"""
        <div class="score-gauge">
            <div class="score-gauge-header">
                <span class="score-val" style="color:{cor}">{pct}%</span>
                <span class="score-nivel" style="color:{cor}">{icone_nivel} Probabilidade de Fraude — {rotulo}</span>
            </div>
            <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{largura}%;background:{cor}"></div>
            </div>
            <div class="score-meta">
                <span>Escore bruto: <code>{escore:.6f}</code></span>
                <span>Threshold médio: <code>0.15</code></span>
                <span>Threshold alto: <code>0.40</code></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Integração com AgenteAntifraude
# ---------------------------------------------------------------------------

def gerar_analise_ia(transacao: dict, historico: list[dict]) -> dict:
    """
    Instancia AgenteAntifraude com a transação e o histórico reais
    e retorna o ResultadoAnalise completo.
    """
    return AgenteAntifraude(transacao, historico).analisar()


def _texto_inicial_ia(transacao: dict, historico: list[dict]) -> str:
    """Gera a mensagem inicial do agente ao selecionar uma transação."""
    resultado = gerar_analise_ia(transacao, historico)
    return resultado["justificativa"]


# ---------------------------------------------------------------------------
# Respostas contextuais do chat
# ---------------------------------------------------------------------------

def _resposta_chat(pergunta: str, tx: dict, historico: list[dict]) -> str:
    """
    Gera respostas contextuais usando o AgenteAntifraude e os dados reais
    da transação selecionada.
    """
    p = pergunta.lower()
    escore   = tx.get("escore_risco", 0.0)
    nivel    = tx.get("risco", "baixo")
    count_10 = tx.get("card_txn_count_10min")
    count_1h = tx.get("card_txn_count_1h")
    seg      = tx.get("seconds_since_last_txn")
    hvf      = tx.get("high_value_flag")

    # ── Escore / probabilidade / modelo ────────────────────────────────────
    if any(k in p for k in ["escore", "score", "probabilidade", "modelo", "lgbm", "lightgbm"]):
        threshold = 0.40 if nivel == "alto" else 0.15
        return (
            f"O modelo LightGBM calculou um **escore de probabilidade de fraude de "
            f"`{escore:.4f}` ({escore * 100:.1f}%)**.\n\n"
            f"Esse valor {'supera' if escore >= 0.40 else 'está acima do' if escore >= 0.15 else 'está abaixo do'} "
            f"threshold de `{threshold}` definido para risco **{nivel.upper()}**.\n\n"
            "O modelo foi treinado com divisão temporal estrita (80/20) e usa "
            "197 features numéricas e categóricas da transação."
        )

    # ── Velocidade / frequência ─────────────────────────────────────────────
    if any(k in p for k in ["velocidade", "frequência", "frequencia", "transações", "10 min", "10min", "1h", "hora"]):
        partes = []
        if count_10 is not None:
            partes.append(
                f"**{int(count_10)} transação(ões) do cartão nos últimos 10 minutos** "
                f"({'⚠️ acima do limiar de 2' if count_10 >= 2 else '✅ dentro do esperado'})"
            )
        if count_1h is not None:
            partes.append(
                f"**{int(count_1h)} transação(ões) do cartão na última hora** "
                f"({'⚠️ acima do limiar de 4' if count_1h >= 4 else '✅ dentro do esperado'})"
            )
        if not partes:
            return "As features de velocidade não estão disponíveis para esta transação."
        return (
            "Dados de velocidade calculados pelo pipeline de engenharia de features:\n\n"
            + "\n".join(f"• {p}" for p in partes)
            + "\n\nAltas frequências em janelas curtas são um dos principais sinais de "
            "uso indevido de cartão por terceiros."
        )

    # ── Intervalo temporal ──────────────────────────────────────────────────
    if any(k in p for k in ["intervalo", "tempo", "último", "ultima", "anterior", "segundos"]):
        if seg is None:
            return (
                "Esta é a **primeira transação registrada** para o cliente — "
                "não há intervalo temporal anterior para comparar."
            )
        minutos = seg / 60
        alerta = seg < 60
        return (
            f"O intervalo desde a última transação do cliente é de "
            f"**{seg:.0f}s ({minutos:.1f} min)**.\n\n"
            + ("⚠️ Intervalo abaixo de 60s — possível uso simultâneo do cartão ou automação suspeita."
               if alerta else
               f"✅ Intervalo dentro dos parâmetros normais.")
        )

    # ── Flag de valor alto ──────────────────────────────────────────────────
    if any(k in p for k in ["valor", "quantia", "montante", "high_value", "atípico", "atipico"]):
        media_hist = (
            sum(h["valor"] for h in historico) / len(historico) if historico else None
        )
        base = (
            f"O valor da transação é **R$ {tx['valor']:,.2f}**"
            + (f", contra uma média histórica de **R$ {media_hist:,.2f}**" if media_hist else "")
            + ".\n\n"
        )
        if hvf and hvf >= 1:
            base += (
                "⚠️ O pipeline de features sinalizou `high_value_flag = 1`: o valor supera "
                "o dobro da média histórica expandida do cliente."
            )
        else:
            base += "✅ O pipeline não sinalizou anomalia de valor (`high_value_flag = 0`)."
        return base

    # ── Recomendação / ação ─────────────────────────────────────────────────
    if any(k in p for k in ["bloqu", "recomend", "ação", "acao", "devo", "fazer"]):
        resultado = gerar_analise_ia(tx, historico)
        acao_label = {
            "bloquear_cartao":  "🚫 Bloquear Cartão",
            "bloquear_compra":  "🔒 Bloquear Compra",
            "enviar_mensagem":  "📩 Enviar Mensagem ao Cliente",
            "monitorar":        "👁️ Monitorar",
        }.get(resultado["acao_sugerida"], resultado["acao_sugerida"])
        return (
            f"Com base no escore `{escore:.4f}` e nos fatores identificados, "
            f"a ação recomendada é **{acao_label}**.\n\n"
            + resultado["justificativa"]
        )

    # ── Perfil / histórico do cliente ───────────────────────────────────────
    if any(k in p for k in ["cliente", "histórico", "histor", "perfil"]):
        if not historico:
            return (
                f"Não há histórico de transações anteriores para o cliente "
                f"**{tx['client_id']}**."
            )
        media = sum(h["valor"] for h in historico) / len(historico)
        suspeitas = sum(1 for h in historico if h.get("suspeita"))
        return (
            f"O cliente **{tx['client_id']}** possui **{len(historico)} transação(ões)** registradas.\n\n"
            f"• Valor médio: **R$ {media:,.2f}**\n"
            f"• Transações suspeitas no histórico: **{suspeitas}**\n"
            f"• Esta transação (R$ {tx['valor']:,.2f}) representa "
            f"**{tx['valor'] / media:.1f}×** o valor médio histórico."
        )

    # ── Local / geografia ───────────────────────────────────────────────────
    if any(k in p for k in ["local", "cidade", "país", "localiz", "geo"]):
        local = tx.get("local") or "não informado"
        return (
            f"A transação foi originada em **{local}**.\n\n"
            "O modelo considera coordenadas geográficas (latitude/longitude do "
            "estabelecimento e do terminal) como features, além de calcular a "
            "velocidade geográfica entre transações consecutivas."
        )

    # ── Fallback completo com análise do agente ─────────────────────────────
    resultado = gerar_analise_ia(tx, historico)
    fatores_txt = (
        "\n".join(f"• {f}" for f in resultado["fatores"])
        if resultado["fatores"]
        else "Nenhum fator adicional identificado."
    )
    return (
        "Posso detalhar: **escore do modelo**, **velocidade do cartão**, "
        "**intervalo entre transações**, **flag de valor alto**, "
        "**perfil do cliente**, **localização** ou **recomendação de ação**.\n\n"
        f"**Resumo atual:**\n{fatores_txt}"
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "transacao_selecionada" not in st.session_state:
    st.session_state.transacao_selecionada = None
if "historico_cache" not in st.session_state:
    st.session_state.historico_cache = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "acao_feedback" not in st.session_state:
    st.session_state.acao_feedback = None
if "acoes_realizadas" not in st.session_state:
    st.session_state.acoes_realizadas = set()

# ---------------------------------------------------------------------------
# Layout: sidebar esquerda (lista) | centro (detalhes) | direita (chat IA)
# ---------------------------------------------------------------------------

# ── Sidebar: lista de transações suspeitas ──────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Transações Suspeitas")

    if st.button("🔄 Atualizar lista", use_container_width=True):
        st.cache_data.clear()
        st.session_state.acao_feedback = None

    transacoes = fetch_transacoes_suspeitas()

    if not transacoes:
        st.info("Nenhuma transação suspeita no momento.")
    else:
        st.markdown(f"**{len(transacoes)} transação(ões) aguardando análise**")
        st.markdown("---")

        for t in transacoes:
            tid = t["id"]
            ja_resolvida = tid in st.session_state.acoes_realizadas
            risco_cor = {"alto": "🔴", "medio": "🟡", "baixo": "🟢"}.get(t["risco"], "⚪")
            escore_pct = f"{t.get('escore_risco', 0) * 100:.1f}%"
            cat = (t.get("categoria") or "—").replace("_", " ").title()
            label = (
                f"{risco_cor} #{tid} — {t['client_id']}\n"
                f"R$ {t['valor']:,.2f} · {escore_pct} · {cat}"
            )
            btn_type = "secondary" if ja_resolvida else "primary"
            if st.button(
                label + (" ✅" if ja_resolvida else ""),
                key=f"btn_tx_{tid}",
                use_container_width=True,
                type=btn_type,
            ):
                historico = fetch_historico(t["client_id"])
                st.session_state.transacao_selecionada = t
                st.session_state.historico_cache = historico
                st.session_state.chat_history = [
                    {
                        "role": "ai",
                        "text": _texto_inicial_ia(t, historico),
                    }
                ]
                st.session_state.acao_feedback = None

    st.markdown("---")
    st.caption("🏦 Sicredi · Antifraude v2.0")

# ── Área principal ──────────────────────────────────────────────────────────

tx = st.session_state.transacao_selecionada

if tx is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🛡️</div>
            <h3>Selecione uma transação suspeita na barra lateral</h3>
            <p>O painel de detalhes e o Agente de IA serão exibidos aqui.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

historico = st.session_state.historico_cache

# ── Colunas: detalhes (60 %) | chat IA (40 %) ──────────────────────────────

col_detalhes, col_chat = st.columns([6, 4], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# COLUNA ESQUERDA — Detalhes da transação
# ════════════════════════════════════════════════════════════════════════════

with col_detalhes:

    risco_badge = {
        "alto":  '<span class="badge-alto">ALTO</span>',
        "medio": '<span class="badge-medio">MÉDIO</span>',
        "baixo": '<span class="badge-baixo">BAIXO</span>',
    }.get(tx["risco"], tx["risco"])

    cat_display   = (tx.get("categoria") or "—").replace("_", " ").title()
    local_display = tx.get("local") or "—"
    ts_display    = tx.get("timestamp", "—").replace("T", " ")

    st.markdown(
        f"""
        <div class="detail-card">
            <h3>Transação #{tx['id']} &nbsp; {risco_badge}</h3>
            <table>
                <tr>
                    <td class="td-label">Cliente</td>
                    <td class="td-value"><strong>{tx['client_id']}</strong></td>
                </tr>
                <tr>
                    <td class="td-label">Cartão</td>
                    <td class="td-value"><code>{tx['card_id']}</code></td>
                </tr>
                <tr>
                    <td class="td-label">Valor</td>
                    <td class="td-value"><span class="val-destaque">R$ {tx['valor']:,.2f}</span></td>
                </tr>
                <tr>
                    <td class="td-label">Categoria</td>
                    <td class="td-value">{cat_display}</td>
                </tr>
                <tr>
                    <td class="td-label">Local</td>
                    <td class="td-value">{local_display}</td>
                </tr>
                <tr>
                    <td class="td-label">Data/Hora</td>
                    <td class="td-value">{ts_display}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Gauge de escore de fraude ───────────────────────────────────────────

    render_score_gauge(tx.get("escore_risco", 0.0), tx["risco"])

    # ── Métricas de velocidade do pipeline ──────────────────────────────────

    count_10 = tx.get("card_txn_count_10min")
    count_1h = tx.get("card_txn_count_1h")
    seg_     = tx.get("seconds_since_last_txn")
    hvf_     = tx.get("high_value_flag")

    if any(v is not None for v in [count_10, count_1h, seg_, hvf_]):
        st.markdown('<div class="section-title">⚡ Features de Velocidade (Pipeline)</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)

        def _delta_color(val, limiar):
            return "inverse" if val is not None and val >= limiar else "normal"

        with mc1:
            st.metric(
                "Txn/10 min",
                f"{int(count_10)}" if count_10 is not None else "—",
                delta="⚠️ alto" if count_10 is not None and count_10 >= 2 else None,
                delta_color=_delta_color(count_10, 2),
            )
        with mc2:
            st.metric(
                "Txn/1 hora",
                f"{int(count_1h)}" if count_1h is not None else "—",
                delta="⚠️ alto" if count_1h is not None and count_1h >= 4 else None,
                delta_color=_delta_color(count_1h, 4),
            )
        with mc3:
            if seg_ is not None:
                minutos = seg_ / 60
                st.metric(
                    "Intervalo",
                    f"{minutos:.1f} min",
                    delta="⚠️ < 1 min" if seg_ < 60 else None,
                    delta_color="inverse" if seg_ < 60 else "normal",
                )
            else:
                st.metric("Intervalo", "1ª txn")
        with mc4:
            st.metric(
                "Valor atípico",
                "Sim ⚠️" if hvf_ else "Não ✅",
                delta=None,
            )

    # ── SHAP — Explicabilidade do escore ───────────────────────────────────

    shap_vals: dict = tx.get("shap_values") or {}
    if shap_vals:
        st.markdown('<div class="section-title">🔍 Explicabilidade do Escore (SHAP)</div>', unsafe_allow_html=True)
        st.caption(
            "Contribuição SHAP de cada feature para o escore de risco desta transação (log-odds). "
            "Valores **positivos** aumentam P(fraude); **negativos** reduzem. "
            "Ordenados por magnitude absoluta — top 15."
        )

        # Monta DataFrame para o gráfico nativo do Streamlit
        import pandas as _pd_shap
        df_shap = _pd_shap.DataFrame(
            {"Contribuição SHAP": list(shap_vals.values())},
            index=list(shap_vals.keys()),
        ).sort_values("Contribuição SHAP")

        st.bar_chart(df_shap, use_container_width=True)

        # Tabela compacta com sinal, cor semântica e valor numérico
        linhas_html = ""
        for feat, val in sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True):
            cls   = "shap-pos" if val > 0 else "shap-neg"
            sinal = "▲" if val > 0 else "▼"
            linhas_html += (
                f"<tr>"
                f"<td><code>{feat}</code></td>"
                f"<td class='{cls}'>{sinal} {val:+.6f}</td>"
                f"</tr>"
            )

        st.markdown(
            f"<table class='shap-table'>{linhas_html}</table>",
            unsafe_allow_html=True,
        )

    # ── Histórico do cliente ────────────────────────────────────────────────

    st.markdown('<div class="section-title">📋 Histórico Recente do Cliente</div>', unsafe_allow_html=True)

    if not historico:
        st.info("Sem histórico disponível.")
    else:
        for h in sorted(historico, key=lambda x: x["timestamp"], reverse=True):
            suspeita_icon = "🚨" if h["suspeita"] else "✅"
            risco_cls     = h.get("risco", "neu")
            atual_cls     = " atual" if h["id"] == tx["id"] else ""
            escore_h      = h.get("escore_risco", 0.0)
            cor_escore_h  = _cor_escore(escore_h)
            cat_h         = (h.get("categoria") or "—").replace("_", " ").title()
            local_h       = h.get("local") or "—"
            ts_h          = h.get("timestamp", "—").replace("T", " ")

            st.markdown(
                f"""
                <div class="hist-card {risco_cls}{atual_cls}">
                    <span>{suspeita_icon}</span>
                    <span class="hist-id">#{h['id']}</span>
                    <span class="hist-sep">·</span>
                    <span>{cat_h}</span>
                    <span class="hist-sep">·</span>
                    <span class="hist-val">R$ {h['valor']:,.2f}</span>
                    <span class="hist-sep">·</span>
                    <span style="color:{cor_escore_h};font-weight:700">{escore_h * 100:.1f}%</span>
                    <span class="hist-sep">·</span>
                    <span>{local_h}</span>
                    <span class="hist-sep">·</span>
                    <span class="hist-ts">{ts_h}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Feedback visual de ação ─────────────────────────────────────────────

    if st.session_state.acao_feedback:
        fb = st.session_state.acao_feedback
        st.success(f"✅ **Ação executada com sucesso!**\n\n{fb['resultado']}")

    # ── Botões de ação ──────────────────────────────────────────────────────

    st.markdown('<div class="section-title">⚡ Executar Ação</div>', unsafe_allow_html=True)

    ja_resolvida = tx["id"] in st.session_state.acoes_realizadas
    if ja_resolvida:
        st.info("✅ Esta transação já foi tratada nesta sessão.")
    else:
        bcol1, bcol2, bcol3 = st.columns(3)

        justificativas_map = {
            "bloquear_compra":  "Transação bloqueada pelo analista via dashboard antifraude.",
            "bloquear_cartao":  "Cartão bloqueado pelo analista via dashboard antifraude.",
            "enviar_mensagem":  "Mensagem de alerta enviada ao cliente via dashboard antifraude.",
        }

        def _handle_acao(tipo: str):
            resultado = executar_acao_api(tx["id"], tipo, justificativas_map[tipo])
            if resultado:
                st.session_state.acao_feedback = resultado
                st.session_state.acoes_realizadas.add(tx["id"])
                rotulos = {
                    "bloquear_compra": "🔒 Bloquear Compra",
                    "bloquear_cartao": "🚫 Bloquear Cartão",
                    "enviar_mensagem": "📩 Enviar Mensagem",
                }
                st.session_state.chat_history.append(
                    {"role": "user", "text": f"Ação executada: **{rotulos[tipo]}**"}
                )
                st.session_state.chat_history.append(
                    {
                        "role": "ai",
                        "text": (
                            f"✅ Ação registrada com sucesso.\n\n"
                            f"_{resultado['resultado']}_\n\n"
                            "O caso será monitorado até a resolução completa."
                        ),
                    }
                )
                st.cache_data.clear()

        with bcol1:
            if st.button("🔒 Bloquear Compra", use_container_width=True, type="primary"):
                _handle_acao("bloquear_compra")
                time.sleep(0.1)
                st.rerun()

        with bcol2:
            if st.button("🚫 Bloquear Cartão", use_container_width=True, type="primary"):
                _handle_acao("bloquear_cartao")
                time.sleep(0.1)
                st.rerun()

        with bcol3:
            if st.button("📩 Enviar Mensagem", use_container_width=True):
                _handle_acao("enviar_mensagem")
                time.sleep(0.1)
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# COLUNA DIREITA — Chat com Agente de IA
# ════════════════════════════════════════════════════════════════════════════

with col_chat:

    st.markdown('<div class="section-title">🤖 Agente de IA Antifraude</div>', unsafe_allow_html=True)
    st.caption(
        "Análise baseada no escore LightGBM + features de velocidade e comportamento temporal."
    )

    chat_container = st.container(height=480, border=True)

    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "ai":
                st.markdown(
                    f'<div class="chat-msg-ai">'
                    f'<div class="chat-sender">🤖 Agente IA</div>'
                    f'{msg["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-msg-user">'
                    f'<div class="chat-sender">👤 Analista</div>'
                    f'{msg["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Caixa de mensagem livre ─────────────────────────────────────────────

    with st.form("form_chat", clear_on_submit=True):
        user_input = st.text_input(
            "Pergunte ao Agente de IA…",
            placeholder="Ex: Por que o escore é alto? Qual a velocidade do cartão?",
            label_visibility="collapsed",
        )
        enviado = st.form_submit_button("Enviar ▶", use_container_width=True)

    if enviado and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        resp = _resposta_chat(user_input, tx, historico)
        st.session_state.chat_history.append({"role": "ai", "text": resp})
        st.rerun()
