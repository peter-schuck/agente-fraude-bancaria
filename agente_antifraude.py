"""
Agente de análise antifraude baseado em regras heurísticas.

Uso:
    from agente_antifraude import AgenteAntifraude

    resultado = AgenteAntifraude(transacao, historico).analisar()
    # {"nivel_risco": "alto", "justificativa": "...", "acao_sugerida": "bloquear_cartao"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import TypedDict


# ---------------------------------------------------------------------------
# Tipos de dados
# ---------------------------------------------------------------------------

class Transacao(TypedDict):
    id: int
    cliente_id: str
    valor: float
    local: str
    categoria: str
    risco: str
    suspeita: bool
    data_hora: str  # ISO-8601, ex: "2024-07-15T02:34:00"


class ResultadoAnalise(TypedDict):
    nivel_risco: str          # "baixo" | "medio" | "alto"
    justificativa: str        # texto explicativo para o analista
    acao_sugerida: str        # "monitorar" | "enviar_mensagem" | "bloquear_compra" | "bloquear_cartao"


# ---------------------------------------------------------------------------
# Constantes de calibração
# ---------------------------------------------------------------------------

# Quantas vezes o valor pode superar a média histórica antes de ser atípico
_FATOR_VALOR_MEDIO  = 2.0   # acima de 2× a média → gatilho médio
_FATOR_VALOR_ALTO   = 4.0   # acima de 4× a média → gatilho alto

# Desvios-padrão acima da média para considerar atípico (quando há dados suficientes)
_DESVIOS_ATIPICO    = 2.0

# Janela de tempo (horas) para detectar mudança rápida de localização
_JANELA_MUDANCA_LOCAL_H = 2.0

# Hora da madrugada (0h–5h) eleva o risco
_HORA_INICIO_MADRUGADA = 0
_HORA_FIM_MADRUGADA    = 5

# Categorias que historicamente concentram mais fraudes
_CATEGORIAS_ALTO_RISCO = {"compra_internacional", "transferencia", "saque"}


# ---------------------------------------------------------------------------
# Agente principal
# ---------------------------------------------------------------------------

@dataclass
class AgenteAntifraude:
    """
    Analisa uma transação frente ao histórico do cliente e retorna
    nivel_risco, justificativa e acao_sugerida.

    Args:
        transacao: dicionário com os campos de uma Transacao.
        historico:  lista de transações anteriores do mesmo cliente
                    (pode incluir a transação atual — ela é ignorada pelo ID).
    """

    transacao: Transacao
    historico: list[Transacao]

    # Acumuladores internos (preenchidos durante a análise)
    _fatores: list[str] = field(default_factory=list, init=False, repr=False)
    _pontos:  int       = field(default=0,            init=False, repr=False)

    # ------------------------------------------------------------------
    # Ponto de entrada público
    # ------------------------------------------------------------------

    def analisar(self) -> ResultadoAnalise:
        """Executa todas as regras heurísticas e devolve o resultado consolidado."""
        historico_anterior = [
            h for h in self.historico if h["id"] != self.transacao["id"]
        ]

        self._regra_valor_atipico(historico_anterior)
        self._regra_mudanca_local(historico_anterior)
        self._regra_horario_madrugada()
        self._regra_categoria_alto_risco()
        self._regra_primeira_transacao(historico_anterior)

        nivel     = self._calcular_nivel()
        justifica = self._montar_justificativa(nivel)
        acao      = self._definir_acao(nivel)

        return ResultadoAnalise(
            nivel_risco=nivel,
            justificativa=justifica,
            acao_sugerida=acao,
        )

    # ------------------------------------------------------------------
    # Regras heurísticas (cada uma acumula pontos e registra fatores)
    # ------------------------------------------------------------------

    def _regra_valor_atipico(self, historico: list[Transacao]) -> None:
        """Compara o valor da transação com a média e o desvio do histórico."""
        if not historico:
            return

        valores = [h["valor"] for h in historico]
        media   = mean(valores)

        if media == 0:
            return

        fator = self.transacao["valor"] / media

        # Verificação por desvio-padrão (mais precisa com ≥ 3 amostras)
        if len(valores) >= 3:
            dp = stdev(valores)
            if dp > 0 and (self.transacao["valor"] - media) / dp >= _DESVIOS_ATIPICO:
                self._fatores.append(
                    f"Valor R$ {self.transacao['valor']:,.2f} está {(self.transacao['valor'] - media) / dp:.1f}σ "
                    f"acima da média histórica de R$ {media:,.2f}."
                )
                self._pontos += 2 if fator >= _FATOR_VALOR_ALTO else 1
                return  # evita contagem dupla com a regra de fator abaixo

        # Verificação por múltiplo da média (fallback para histórico curto)
        if fator >= _FATOR_VALOR_ALTO:
            self._fatores.append(
                f"Valor R$ {self.transacao['valor']:,.2f} é {fator:.1f}× a média histórica "
                f"de R$ {media:,.2f} — muito acima do padrão."
            )
            self._pontos += 2
        elif fator >= _FATOR_VALOR_MEDIO:
            self._fatores.append(
                f"Valor R$ {self.transacao['valor']:,.2f} é {fator:.1f}× a média histórica "
                f"de R$ {media:,.2f} — acima do padrão."
            )
            self._pontos += 1

    def _regra_mudanca_local(self, historico: list[Transacao]) -> None:
        """
        Detecta se houve mudança de localização em menos de `_JANELA_MUDANCA_LOCAL_H` horas.
        Considera apenas a transação imediatamente anterior.
        """
        if not historico:
            return

        tx_atual = self.transacao
        dt_atual = _parse_dt(tx_atual["data_hora"])

        # Ordena pelo mais recente antes da transação atual
        anteriores = sorted(
            historico,
            key=lambda h: abs((_parse_dt(h["data_hora"]) - dt_atual).total_seconds()),
        )
        mais_recente = anteriores[0]
        dt_anterior  = _parse_dt(mais_recente["data_hora"])

        delta_horas = abs((dt_atual - dt_anterior).total_seconds()) / 3600

        local_mudou = (
            mais_recente["local"].strip().lower() != tx_atual["local"].strip().lower()
        )

        if local_mudou and delta_horas <= _JANELA_MUDANCA_LOCAL_H:
            self._fatores.append(
                f"Mudança rápida de localização: de '{mais_recente['local']}' para "
                f"'{tx_atual['local']}' em {delta_horas:.1f}h — fisicamente improvável."
            )
            self._pontos += 2

    def _regra_horario_madrugada(self) -> None:
        """Transações entre 0h e 5h recebem incremento de risco."""
        hora = _parse_dt(self.transacao["data_hora"]).hour
        if _HORA_INICIO_MADRUGADA <= hora < _HORA_FIM_MADRUGADA:
            self._fatores.append(
                f"Transação realizada às {hora:02d}h — horário de madrugada atípico."
            )
            self._pontos += 1

    def _regra_categoria_alto_risco(self) -> None:
        """Categorias com maior incidência histórica de fraude."""
        if self.transacao["categoria"] in _CATEGORIAS_ALTO_RISCO:
            self._fatores.append(
                f"Categoria '{self.transacao['categoria']}' apresenta alta incidência "
                "de fraudes no setor financeiro."
            )
            self._pontos += 1

    def _regra_primeira_transacao(self, historico: list[Transacao]) -> None:
        """Cliente sem histórico prévio merece atenção adicional."""
        if not historico:
            self._fatores.append(
                "Primeira transação registrada para este cliente — perfil ainda não estabelecido."
            )
            self._pontos += 1

    # ------------------------------------------------------------------
    # Consolidação do resultado
    # ------------------------------------------------------------------

    def _calcular_nivel(self) -> str:
        if self._pontos >= 4:
            return "alto"
        if self._pontos >= 2:
            return "medio"
        return "baixo"

    def _montar_justificativa(self, nivel: str) -> str:
        cabecalho = {
            "alto":  "🔴 Risco ALTO — múltiplos indicadores de fraude detectados.",
            "medio": "🟡 Risco MÉDIO — anomalias moderadas identificadas.",
            "baixo": "🟢 Risco BAIXO — transação dentro dos padrões esperados.",
        }[nivel]

        if not self._fatores:
            return f"{cabecalho}\n\nNenhum fator de risco adicional identificado."

        itens = "\n".join(f"  • {f}" for f in self._fatores)
        return f"{cabecalho}\n\nFatores identificados:\n{itens}"

    def _definir_acao(self, nivel: str) -> str:
        if nivel == "alto":
            return "bloquear_cartao"
        if nivel == "medio":
            return "enviar_mensagem"
        return "monitorar"


# ---------------------------------------------------------------------------
# Utilitário interno
# ---------------------------------------------------------------------------

def _parse_dt(data_hora: str) -> datetime:
    """Converte string ISO-8601 para datetime (suporta 'T' ou espaço como separador)."""
    return datetime.fromisoformat(data_hora.replace(" ", "T"))
