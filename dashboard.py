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
        /* Cabeçalho principal */
        .header-bar {
            background: #006633;
            color: #ffffff;
            padding: 14px 24px;
            border-radius: 8px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-bar h1 { margin: 0; font-size: 1.4rem; font-weight: 700; }
        .header-bar span { font-size: 0.85rem; opacity: .75; }

        /* Badge de risco */
        .badge-alto  { background:#fee2e2; color:#b91c1c; padding:2px 10px; border-radius:12px; font-size:.78rem; font-weight:600; }
        .badge-medio { background:#fef9c3; color:#92400e; padding:2px 10px; border-radius:12px; font-size:.78rem; font-weight:600; }
        .badge-baixo { background:#dcfce7; color:#166534; padding:2px 10px; border-radius:12px; font-size:.78rem; font-weight:600; }

        /* Card de detalhe */
        .detail-card {
            background: #f8fdf9;
            border: 1px solid #d1fae5;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }

        /* Mensagens do chat */
        .chat-msg-ai {
            background: #f0fdf4;
            border-left: 3px solid #006633;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 8px;
            font-size: .88rem;
        }
        .chat-msg-user {
            background: #f1f5f9;
            border-left: 3px solid #94a3b8;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 8px;
            font-size: .88rem;
            text-align: right;
        }

        /* Botões de ação */
        div[data-testid="stHorizontalBlock"] button { font-size: .82rem; }

        /* Remove padding excessivo do sidebar */
        section[data-testid="stSidebar"] > div { padding-top: 1rem; }
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
        <span style="font-size:2rem">🛡️</span>
        <div>
            <h1>Central de Análise Antifraude</h1>
            <span>Sicredi — Monitoramento em tempo real de transações suspeitas</span>
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


def fetch_historico(cliente_id: str) -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/cliente/{cliente_id}/historico", timeout=5)
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
# Agente de IA simulado
# ---------------------------------------------------------------------------

_PERFIS_RISCO = {
    "alto": (
        "🔴 **Risco ALTO detectado.**\n\n"
        "Os algoritmos de detecção identificaram padrões altamente atípicos nesta transação:\n"
        "- Horário incomum (madrugada) combinado com valor elevado\n"
        "- Localização geográfica divergente do perfil histórico do cliente\n"
        "- Categoria de gasto nunca registrada anteriormente\n\n"
        "**Recomendação:** Bloquear o cartão imediatamente e acionar o cliente "
        "para validação de identidade antes de liberar novas transações."
    ),
    "medio": (
        "🟡 **Risco MÉDIO detectado.**\n\n"
        "Existem anomalias moderadas que justificam atenção:\n"
        "- Valor acima da média habitual do cliente\n"
        "- Local diferente do padrão de consumo recente\n\n"
        "**Recomendação:** Enviar mensagem de confirmação ao cliente. "
        "Caso não haja resposta em 30 minutos, bloquear a compra preventivamente."
    ),
    "baixo": (
        "🟢 **Risco BAIXO detectado.**\n\n"
        "Os indicadores estão dentro dos padrões esperados para este cliente. "
        "A transação foi sinalizada por monitoramento preventivo, mas não apresenta "
        "evidências concretas de fraude no momento.\n\n"
        "**Recomendação:** Monitorar. Nenhuma ação imediata necessária."
    ),
}

def gerar_justificativa_ia(transacao: dict) -> str:
    return _PERFIS_RISCO.get(transacao["risco"], _PERFIS_RISCO["medio"])


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "transacao_selecionada" not in st.session_state:
    st.session_state.transacao_selecionada = None
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
            label = (
                f"{risco_cor} #{tid} — {t['cliente_id']}\n"
                f"R$ {t['valor']:,.2f} | {t['categoria']}"
            )
            btn_type = "secondary" if ja_resolvida else "primary"
            if st.button(
                label + (" ✅" if ja_resolvida else ""),
                key=f"btn_tx_{tid}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state.transacao_selecionada = t
                st.session_state.chat_history = [
                    {
                        "role": "ai",
                        "text": gerar_justificativa_ia(t),
                    }
                ]
                st.session_state.acao_feedback = None

    st.markdown("---")
    st.caption("🏦 Sicredi · Antifraude v1.0")

# ── Área principal ──────────────────────────────────────────────────────────

tx = st.session_state.transacao_selecionada

if tx is None:
    st.markdown(
        """
        <div style="text-align:center; padding: 80px 0; color: #64748b;">
            <div style="font-size:3rem">🛡️</div>
            <h3>Selecione uma transação suspeita na barra lateral</h3>
            <p>O painel de detalhes e o Agente de IA serão exibidos aqui.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

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

    st.markdown(
        f"""
        <div class="detail-card">
            <h3 style="margin:0 0 12px 0">Transação #{tx['id']} &nbsp; {risco_badge}</h3>
            <table style="width:100%; font-size:.9rem; border-collapse:collapse;">
                <tr><td style="color:#57606a;padding:4px 8px 4px 0">Cliente</td>
                    <td><strong>{tx['cliente_id']}</strong></td></tr>
                <tr><td style="color:#57606a;padding:4px 8px 4px 0">Valor</td>
                    <td><strong style="color:#006633">R$ {tx['valor']:,.2f}</strong></td></tr>
                <tr><td style="color:#57606a;padding:4px 8px 4px 0">Categoria</td>
                    <td>{tx['categoria'].replace('_', ' ').title()}</td></tr>
                <tr><td style="color:#57606a;padding:4px 8px 4px 0">Local</td>
                    <td>{tx['local']}</td></tr>
                <tr><td style="color:#57606a;padding:4px 8px 4px 0">Data/Hora</td>
                    <td>{tx['data_hora'].replace('T', ' ')}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Histórico do cliente ────────────────────────────────────────────────

    st.markdown("#### 📋 Histórico Recente do Cliente")
    historico = fetch_historico(tx["cliente_id"])

    if not historico:
        st.info("Sem histórico disponível.")
    else:
        for h in sorted(historico, key=lambda x: x["data_hora"], reverse=True):
            suspeita_icon = "🚨" if h["suspeita"] else "✅"
            risco_color = {"alto": "#fee2e2", "medio": "#fef9c3", "baixo": "#f0fdf4"}.get(
                h["risco"], "#f8fafc"
            )
            destaque = "border:2px solid #ef4444;" if h["id"] == tx["id"] else ""
            st.markdown(
                f"""
                <div style="background:{risco_color};{destaque}border-radius:6px;
                            padding:8px 12px;margin-bottom:6px;font-size:.84rem;">
                    {suspeita_icon} &nbsp;
                    <strong>#{h['id']}</strong> —
                    {h['categoria'].replace('_',' ').title()} &nbsp;|&nbsp;
                    <strong>R$ {h['valor']:,.2f}</strong> &nbsp;|&nbsp;
                    {h['local']} &nbsp;|&nbsp;
                    <span style="color:#57606a">{h['data_hora'].replace('T',' ')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Feedback visual de ação ─────────────────────────────────────────────

    if st.session_state.acao_feedback:
        fb = st.session_state.acao_feedback
        st.success(f"✅ **Ação executada com sucesso!**\n\n{fb['resultado']}")

    # ── Botões de ação ──────────────────────────────────────────────────────

    st.markdown("#### ⚡ Executar Ação")

    ja_resolvida = tx["id"] in st.session_state.acoes_realizadas
    if ja_resolvida:
        st.info("✅ Esta transação já foi tratada nesta sessão.")
    else:
        bcol1, bcol2, bcol3 = st.columns(3)

        acoes_map = {
            "🔒 Bloquear Compra":  "bloquear_compra",
            "🚫 Bloquear Cartão":  "bloquear_cartao",
            "📩 Enviar Mensagem":  "enviar_mensagem",
        }

        justificativas_map = {
            "bloquear_compra":  "Transação bloqueada pelo analista via dashboard antifraude.",
            "bloquear_cartao":  "Cartão bloqueado pelo analista via dashboard antifraude.",
            "enviar_mensagem":  "Mensagem de alerta enviada ao cliente via dashboard antifraude.",
        }

        def _handle_acao(tipo: str):
            just = justificativas_map[tipo]
            resultado = executar_acao_api(tx["id"], tipo, just)
            if resultado:
                st.session_state.acao_feedback = resultado
                st.session_state.acoes_realizadas.add(tx["id"])
                # Adiciona mensagem do analista ao chat
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

    st.markdown("#### 🤖 Agente de IA Antifraude")
    st.caption("Análise automática baseada em modelos de detecção de anomalias.")

    chat_container = st.container(height=480, border=True)

    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "ai":
                st.markdown(
                    f'<div class="chat-msg-ai"><strong>🤖 Agente IA</strong><br>{msg["text"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-msg-user"><strong>👤 Analista</strong><br>{msg["text"]}</div>',
                    unsafe_allow_html=True,
                )

    # ── Caixa de mensagem livre ─────────────────────────────────────────────

    with st.form("form_chat", clear_on_submit=True):
        user_input = st.text_input(
            "Pergunte ao Agente de IA…",
            placeholder="Ex: Por que essa transação é suspeita?",
            label_visibility="collapsed",
        )
        enviado = st.form_submit_button("Enviar ▶", use_container_width=True)

    if enviado and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        # Respostas contextuais simuladas
        pergunta = user_input.lower()
        if any(p in pergunta for p in ["bloqu", "recomend", "ação", "acao", "devo"]):
            resp = (
                f"Com base no nível de risco **{tx['risco'].upper()}** desta transação, "
                + _PERFIS_RISCO.get(tx["risco"], "").split("**Recomendação:**")[-1].strip()
            )
        elif any(p in pergunta for p in ["cliente", "histórico", "histor", "perfil"]):
            resp = (
                f"O cliente **{tx['cliente_id']}** possui {len(historico)} transação(ões) registradas. "
                f"O valor médio histórico é R$ "
                f"{(sum(h['valor'] for h in historico) / len(historico)):,.2f}. "
                f"Esta transação de R$ {tx['valor']:,.2f} representa um desvio significativo."
                if historico else
                "Não foi possível recuperar o histórico do cliente no momento."
            )
        elif any(p in pergunta for p in ["local", "cidade", "país", "localiz"]):
            resp = (
                f"A transação foi originada em **{tx['local']}**. "
                "Localização incompatível com o padrão geográfico habitual do cliente, "
                "o que aumenta o score de risco no modelo de detecção."
            )
        elif any(p in pergunta for p in ["valor", "quantia", "montante"]):
            resp = (
                f"O valor de **R$ {tx['valor']:,.2f}** foi classificado como atípico "
                f"para a categoria **{tx['categoria'].replace('_',' ')}** deste cliente, "
                "ultrapassando o limiar estatístico de desvio-padrão definido pelo modelo."
            )
        else:
            resp = (
                "Posso ajudar com análise de risco, histórico do cliente, localização "
                "e recomendações de ação. Pode reformular sua pergunta?"
            )

        st.session_state.chat_history.append({"role": "ai", "text": resp})
        st.rerun()
