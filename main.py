"""
Sistema Antifraude Bancário — Backend simulado com FastAPI.
Execute com: uvicorn main:app --reload
Documentação interativa disponível em: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sistema Antifraude Bancário",
    description="API para análise e gestão de transações suspeitas.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class Transacao(BaseModel):
    """Representa uma transação financeira no sistema."""

    id: int
    cliente_id: str
    valor: float = Field(..., gt=0, description="Valor em R$ (deve ser positivo)")
    local: str
    categoria: str
    risco: str = Field(..., pattern="^(baixo|medio|alto)$", description="Nível de risco: baixo | medio | alto")
    suspeita: bool = False
    data_hora: str


class AcaoAnalista(BaseModel):
    """Representa a ação executada por um analista sobre uma transação."""

    tipo_acao: str = Field(
        ...,
        pattern="^(bloquear_compra|bloquear_cartao|enviar_mensagem)$",
        description="Ação: bloquear_compra | bloquear_cartao | enviar_mensagem",
    )
    justificativa: str = Field(..., min_length=10, description="Justificativa com ao menos 10 caracteres")


# ---------------------------------------------------------------------------
# Base de dados em memória
# ---------------------------------------------------------------------------

# Simulação de transações — algumas marcadas como suspeitas (suspeita=True)
transacoes_db: list[dict] = [
    {
        "id": 1,
        "cliente_id": "CLI-001",
        "valor": 12500.00,
        "local": "São Paulo, SP",
        "categoria": "transferencia",
        "risco": "alto",
        "suspeita": True,
        "data_hora": "2024-07-15T02:34:00",
    },
    {
        "id": 2,
        "cliente_id": "CLI-002",
        "valor": 89.90,
        "local": "Rio de Janeiro, RJ",
        "categoria": "alimentacao",
        "risco": "baixo",
        "suspeita": False,
        "data_hora": "2024-07-15T12:10:00",
    },
    {
        "id": 3,
        "cliente_id": "CLI-001",
        "valor": 3200.00,
        "local": "Frankfurt, Alemanha",
        "categoria": "compra_internacional",
        "risco": "alto",
        "suspeita": True,
        "data_hora": "2024-07-15T03:05:00",
    },
    {
        "id": 4,
        "cliente_id": "CLI-003",
        "valor": 450.00,
        "local": "Curitiba, PR",
        "categoria": "eletronicos",
        "risco": "medio",
        "suspeita": True,
        "data_hora": "2024-07-14T23:58:00",
    },
    {
        "id": 5,
        "cliente_id": "CLI-002",
        "valor": 1200.00,
        "local": "Belo Horizonte, MG",
        "categoria": "transferencia",
        "risco": "medio",
        "suspeita": False,
        "data_hora": "2024-07-14T18:22:00",
    },
    {
        "id": 6,
        "cliente_id": "CLI-003",
        "valor": 7800.00,
        "local": "Miami, EUA",
        "categoria": "compra_internacional",
        "risco": "alto",
        "suspeita": True,
        "data_hora": "2024-07-13T04:17:00",
    },
]

# Registro de ações executadas pelos analistas (auditoria)
acoes_executadas: list[dict] = []

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def buscar_transacao(transacao_id: int) -> dict:
    """Localiza uma transação pelo ID ou lança 404."""
    for t in transacoes_db:
        if t["id"] == transacao_id:
            return t
    raise HTTPException(status_code=404, detail=f"Transação {transacao_id} não encontrada.")


def simular_acao(tipo_acao: str, transacao: dict) -> str:
    """
    Simula o efeito de cada ação sobre a transação.
    Retorna uma mensagem descrevendo o resultado da operação.
    """
    cliente = transacao["cliente_id"]
    tid = transacao["id"]

    if tipo_acao == "bloquear_compra":
        # Marca a transação como não-suspeita (bloqueada/resolvida)
        transacao["suspeita"] = False
        return f"Compra #{tid} do cliente {cliente} bloqueada com sucesso."

    if tipo_acao == "bloquear_cartao":
        # Bloqueia todas as transações futuras simulando bloqueio de cartão
        transacao["suspeita"] = False
        return (
            f"Cartão do cliente {cliente} bloqueado. "
            f"Transação #{tid} encerrada e conta sinalizada para revisão."
        )

    if tipo_acao == "enviar_mensagem":
        # Simula envio de notificação ao cliente
        return (
            f"Mensagem de alerta enviada ao cliente {cliente} "
            f"referente à transação #{tid}. Aguardando confirmação."
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/transacoes-suspeitas",
    response_model=list[Transacao],
    summary="Lista todas as transações suspeitas",
    tags=["Transações"],
)
def listar_transacoes_suspeitas():
    """
    Retorna todas as transações sinalizadas como suspeitas
    (campo `suspeita = true`) presentes na base de dados.
    """
    suspeitas = [t for t in transacoes_db if t["suspeita"]]

    if not suspeitas:
        raise HTTPException(status_code=404, detail="Nenhuma transação suspeita encontrada.")

    return suspeitas


@app.get(
    "/cliente/{cliente_id}/historico",
    response_model=list[Transacao],
    summary="Histórico de transações de um cliente",
    tags=["Clientes"],
)
def historico_cliente(cliente_id: str):
    """
    Retorna todas as transações (suspeitas ou não) associadas
    ao `cliente_id` informado.
    """
    historico = [t for t in transacoes_db if t["cliente_id"] == cliente_id]

    if not historico:
        raise HTTPException(
            status_code=404,
            detail=f"Nenhuma transação encontrada para o cliente '{cliente_id}'.",
        )

    return historico


@app.post(
    "/transacao/{id}/executar-acao",
    summary="Executa uma ação de analista sobre uma transação",
    tags=["Ações"],
)
def executar_acao(id: int, acao: AcaoAnalista):
    """
    Permite que um analista execute uma ação sobre uma transação específica.

    **Ações disponíveis:**
    - `bloquear_compra` — Bloqueia e resolve a transação suspeita.
    - `bloquear_cartao` — Bloqueia o cartão do cliente e encerra a transação.
    - `enviar_mensagem` — Envia uma notificação de alerta ao cliente.
    """
    transacao = buscar_transacao(id)

    # Executa a ação simulada e obtém o resultado
    resultado = simular_acao(acao.tipo_acao, transacao)

    # Registra a ação para auditoria
    registro = {
        "transacao_id": id,
        "cliente_id": transacao["cliente_id"],
        "tipo_acao": acao.tipo_acao,
        "justificativa": acao.justificativa,
        "resultado": resultado,
        "executado_em": datetime.now().isoformat(),
    }
    acoes_executadas.append(registro)

    return {
        "status": "sucesso",
        "acao_executada": acao.tipo_acao,
        "resultado": resultado,
        "auditoria": registro,
    }


@app.get(
    "/acoes",
    summary="Lista todas as ações executadas pelos analistas (auditoria)",
    tags=["Ações"],
)
def listar_acoes():
    """
    Retorna o log completo de todas as ações executadas durante a sessão,
    útil para fins de auditoria e rastreabilidade.
    """
    if not acoes_executadas:
        return {"mensagem": "Nenhuma ação registrada nesta sessão."}
    return acoes_executadas
