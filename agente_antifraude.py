"""
Agente de análise antifraude.

Combina o escore de probabilidade de fraude retornado pela API (predict_proba
do LightGBM) com regras heurísticas sobre variáveis temporais, de velocidade
e comportamentais para produzir justificativas detalhadas ao analista.

Uso:
    from agente_antifraude import AgenteAntifraude

    resultado = AgenteAntifraude(transacao, historico).analisar()
    # ResultadoAnalise(
    #   escore_modelo=0.73,
    #   nivel_risco="alto",
    #   justificativa="...",
    #   fatores=[...],
    #   acao_sugerida="bloquear_cartao",
    # )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import Optional, TypedDict


# ---------------------------------------------------------------------------
# Tipos de dados — alinhados ao schema retornado pela API (main.py)
# ---------------------------------------------------------------------------

class Transacao(TypedDict):
    id:           int
    client_id:    str
    card_id:      str
    valor:        float
    local:        Optional[str]
    categoria:    Optional[str]
    timestamp:    str            # ISO-8601, ex: "2024-07-15T02:34:00"
    escore_risco: float          # predict_proba do LightGBM (0.0 – 1.0)
    risco:        str            # "baixo" | "medio" | "alto"
    suspeita:     bool

    # Features de velocidade calculadas pelo pipeline (presentes quando disponíveis)
    card_txn_count_10min:   Optional[float]
    card_txn_count_1h:      Optional[float]
    seconds_since_last_txn: Optional[float]
    high_value_flag:        Optional[float]


class ResultadoAnalise(TypedDict):
    escore_modelo:  float   # escore bruto do LightGBM (0.0 – 1.0)
    nivel_risco:    str     # "baixo" | "medio" | "alto"
    justificativa:  str     # texto explicativo principal para o analista
    fatores:        list[str]   # lista de fatores individuais identificados
    acao_sugerida:  str     # "monitorar" | "enviar_mensagem" | "bloquear_compra" | "bloquear_cartao"


# ---------------------------------------------------------------------------
# Constantes de calibração heurística
# ---------------------------------------------------------------------------

# Thresholds do modelo LightGBM (espelham os definidos em main.py)
_THRESHOLD_ALTO  = 0.40
_THRESHOLD_MEDIO = 0.15

# Múltiplos da média histórica para flag de valor atípico
_FATOR_VALOR_MEDIO = 2.0
_FATOR_VALOR_ALTO  = 4.0

# Desvios-padrão para considerar valor estatisticamente atípico
_DESVIOS_ATIPICO = 2.0

# Janela de tempo (horas) para detectar mudança rápida de localização
_JANELA_MUDANCA_LOCAL_H = 2.0

# Faixa de madrugada (0h–5h)
_HORA_INICIO_MADRUGADA = 0
_HORA_FIM_MADRUGADA    = 5

# Categorias com maior incidência histórica de fraude
_CATEGORIAS_ALTO_RISCO = {"compra_internacional", "transferencia", "saque"}

# Thresholds de velocidade (transações em janela temporal)
_LIMIAR_VELOCIDADE_10MIN = 2   # ≥ 2 transações em 10 min → suspeito
_LIMIAR_VELOCIDADE_1H    = 4   # ≥ 4 transações em 1 hora → suspeito

# Intervalo mínimo entre transações (segundos) abaixo do qual é suspeito
_MIN_INTERVALO_SEG = 60        # < 60 s entre transações → suspeito

# Pontuação mínima para cada nível
_PONTOS_ALTO  = 4
_PONTOS_MEDIO = 2


# ---------------------------------------------------------------------------
# Agente principal
# ---------------------------------------------------------------------------

@dataclass
class AgenteAntifraude:
    """
    Analisa uma transação combinando o escore do modelo LightGBM com regras
    heurísticas sobre variáveis temporais, de velocidade e comportamentais.

    Args:
        transacao: dicionário com os campos de uma Transacao (schema da API).
        historico:  lista de transações anteriores do mesmo cliente
                    (pode incluir a transação atual — ela é filtrada pelo ID).
    """

    transacao: Transacao
    historico: list[Transacao]

    _fatores: list[str] = field(default_factory=list, init=False, repr=False)
    _pontos:  int       = field(default=0,            init=False, repr=False)

    # ------------------------------------------------------------------
    # Ponto de entrada público
    # ------------------------------------------------------------------

    def analisar(self) -> ResultadoAnalise:
        """Executa todas as regras e devolve o resultado consolidado."""
        historico_anterior = [
            h for h in self.historico if h["id"] != self.transacao["id"]
        ]

        escore = float(self.transacao.get("escore_risco", 0.0))

        # Regra principal: escore do modelo LightGBM
        self._regra_escore_modelo(escore)

        # Regras sobre features de velocidade calculadas pelo pipeline
        self._regra_velocidade_cartao()
        self._regra_intervalo_temporal()
        self._regra_high_value_flag()

        # Regras heurísticas comportamentais
        self._regra_valor_atipico(historico_anterior)
        self._regra_mudanca_local(historico_anterior)
        self._regra_horario_madrugada()
        self._regra_categoria_alto_risco()
        self._regra_primeira_transacao(historico_anterior)

        nivel     = self._calcular_nivel(escore)
        justifica = self._montar_justificativa(nivel, escore)
        acao      = self._definir_acao(nivel)

        return ResultadoAnalise(
            escore_modelo=round(escore, 6),
            nivel_risco=nivel,
            justificativa=justifica,
            fatores=list(self._fatores),
            acao_sugerida=acao,
        )

    # ------------------------------------------------------------------
    # Regra principal: escore do LightGBM
    # ------------------------------------------------------------------

    def _regra_escore_modelo(self, escore: float) -> None:
        """
        Traduz o escore de probabilidade de fraude do modelo em pontos e
        registra o fator correspondente com contexto quantitativo.
        """
        pct = escore * 100
        if escore >= _THRESHOLD_ALTO:
            self._fatores.append(
                f"Modelo LightGBM atribuiu probabilidade de fraude de {pct:.1f}% "
                f"(threshold alto: {_THRESHOLD_ALTO * 100:.0f}%) — padrão altamente atípico."
            )
            self._pontos += 3
        elif escore >= _THRESHOLD_MEDIO:
            self._fatores.append(
                f"Modelo LightGBM atribuiu probabilidade de fraude de {pct:.1f}% "
                f"(threshold médio: {_THRESHOLD_MEDIO * 100:.0f}%) — anomalia moderada detectada."
            )
            self._pontos += 1

    # ------------------------------------------------------------------
    # Regras sobre features de velocidade
    # ------------------------------------------------------------------

    def _regra_velocidade_cartao(self) -> None:
        """
        Avalia card_txn_count_10min e card_txn_count_1h calculados pelo pipeline
        de engenharia de features (build_velocity_features).
        """
        count_10min = self.transacao.get("card_txn_count_10min")
        count_1h    = self.transacao.get("card_txn_count_1h")

        if count_10min is not None and count_10min >= _LIMIAR_VELOCIDADE_10MIN:
            self._fatores.append(
                f"Velocidade elevada: {int(count_10min)} transação(ões) do cartão nos "
                f"últimos 10 minutos (limiar: {_LIMIAR_VELOCIDADE_10MIN}) — "
                "padrão típico de uso indevido por terceiro."
            )
            self._pontos += 2

        if count_1h is not None and count_1h >= _LIMIAR_VELOCIDADE_1H:
            self._fatores.append(
                f"Alta frequência: {int(count_1h)} transação(ões) do cartão na última "
                f"hora (limiar: {_LIMIAR_VELOCIDADE_1H}) — possível uso massivo suspeito."
            )
            self._pontos += 1

    def _regra_intervalo_temporal(self) -> None:
        """
        Avalia seconds_since_last_txn: intervalos muito curtos entre transações
        do mesmo cliente indicam automação ou uso simultâneo do cartão.
        """
        seg = self.transacao.get("seconds_since_last_txn")
        if seg is None:
            return   # primeira transação — sem histórico prévio

        if seg < _MIN_INTERVALO_SEG:
            self._fatores.append(
                f"Intervalo de apenas {seg:.0f}s desde a última transação do cliente "
                f"(mínimo esperado: {_MIN_INTERVALO_SEG}s) — possível uso simultâneo do cartão."
            )
            self._pontos += 2
        elif seg < 300:  # < 5 min
            self._fatores.append(
                f"Intervalo curto de {seg:.0f}s ({seg / 60:.1f} min) desde a última "
                "transação — merece atenção para padrão de uso."
            )
            self._pontos += 1

    def _regra_high_value_flag(self) -> None:
        """
        Avalia high_value_flag: calculada pelo pipeline como 1 quando o valor
        da transação supera N× a média histórica expandida do cliente.
        """
        flag = self.transacao.get("high_value_flag")
        if flag is not None and flag >= 1:
            media_est = self.transacao["valor"] / 2.0  # estimativa mínima
            self._fatores.append(
                f"Pipeline de features sinalizou valor atípico (high_value_flag=1): "
                f"R$ {self.transacao['valor']:,.2f} supera o dobro da média histórica "
                "expandida do cliente sem saque de caixa anterior."
            )
            self._pontos += 1

    # ------------------------------------------------------------------
    # Regras heurísticas comportamentais
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

        if len(valores) >= 3:
            dp = stdev(valores)
            if dp > 0 and (self.transacao["valor"] - media) / dp >= _DESVIOS_ATIPICO:
                sigmas = (self.transacao["valor"] - media) / dp
                self._fatores.append(
                    f"Valor R$ {self.transacao['valor']:,.2f} está {sigmas:.1f}σ acima "
                    f"da média histórica do cliente (R$ {media:,.2f})."
                )
                self._pontos += 2 if fator >= _FATOR_VALOR_ALTO else 1
                return

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
        """Detecta mudança de localização em menos de `_JANELA_MUDANCA_LOCAL_H` horas."""
        if not historico:
            return

        local_atual = (self.transacao.get("local") or "").strip().lower()
        if not local_atual:
            return

        dt_atual = _parse_dt(self.transacao["timestamp"])
        anteriores = sorted(
            historico,
            key=lambda h: abs((_parse_dt(h["timestamp"]) - dt_atual).total_seconds()),
        )
        mais_recente = anteriores[0]
        local_anterior = (mais_recente.get("local") or "").strip().lower()
        if not local_anterior:
            return

        dt_anterior  = _parse_dt(mais_recente["timestamp"])
        delta_horas  = abs((dt_atual - dt_anterior).total_seconds()) / 3600

        if local_anterior != local_atual and delta_horas <= _JANELA_MUDANCA_LOCAL_H:
            self._fatores.append(
                f"Mudança rápida de localização: '{mais_recente.get('local')}' para "
                f"'{self.transacao.get('local')}' em {delta_horas:.1f}h — fisicamente improvável."
            )
            self._pontos += 2

    def _regra_horario_madrugada(self) -> None:
        """Transações entre 0h e 5h recebem incremento de risco."""
        hora = _parse_dt(self.transacao["timestamp"]).hour
        if _HORA_INICIO_MADRUGADA <= hora < _HORA_FIM_MADRUGADA:
            self._fatores.append(
                f"Transação realizada às {hora:02d}h — horário de madrugada atípico."
            )
            self._pontos += 1

    def _regra_categoria_alto_risco(self) -> None:
        """Categorias com maior incidência histórica de fraude."""
        cat = self.transacao.get("categoria") or ""
        if cat in _CATEGORIAS_ALTO_RISCO:
            self._fatores.append(
                f"Categoria '{cat}' apresenta alta incidência de fraudes no setor financeiro."
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

    def _calcular_nivel(self, escore: float) -> str:
        """
        O nível é determinado primariamente pelo escore do modelo. A pontuação
        heurística pode elevar (mas não rebaixar) o nível do modelo.
        """
        # Nível base pelo escore do LightGBM
        if escore >= _THRESHOLD_ALTO:
            nivel_modelo = "alto"
        elif escore >= _THRESHOLD_MEDIO:
            nivel_modelo = "medio"
        else:
            nivel_modelo = "baixo"

        # Pontuação heurística pode elevar o nível
        if self._pontos >= _PONTOS_ALTO:
            nivel_heuristico = "alto"
        elif self._pontos >= _PONTOS_MEDIO:
            nivel_heuristico = "medio"
        else:
            nivel_heuristico = "baixo"

        ordem = {"baixo": 0, "medio": 1, "alto": 2}
        return max(nivel_modelo, nivel_heuristico, key=lambda n: ordem[n])

    def _montar_justificativa(self, nivel: str, escore: float) -> str:
        cabecalho = {
            "alto":  "🔴 **Risco ALTO** — múltiplos indicadores de fraude detectados.",
            "medio": "🟡 **Risco MÉDIO** — anomalias moderadas identificadas.",
            "baixo": "🟢 **Risco BAIXO** — transação dentro dos padrões esperados.",
        }[nivel]

        linhas = [
            cabecalho,
            f"\n**Escore do modelo:** `{escore:.4f}` "
            f"({'≥' if escore >= _THRESHOLD_ALTO else '≥' if escore >= _THRESHOLD_MEDIO else '<'}"
            f" {_THRESHOLD_ALTO if escore >= _THRESHOLD_ALTO else _THRESHOLD_MEDIO if escore >= _THRESHOLD_MEDIO else _THRESHOLD_MEDIO})",
        ]

        if self._fatores:
            linhas.append("\n**Fatores identificados:**")
            for f in self._fatores:
                linhas.append(f"  • {f}")
        else:
            linhas.append("\nNenhum fator de risco adicional identificado.")

        return "\n".join(linhas)

    def _definir_acao(self, nivel: str) -> str:
        if nivel == "alto":
            return "bloquear_cartao"
        if nivel == "medio":
            return "enviar_mensagem"
        return "monitorar"


# ---------------------------------------------------------------------------
# Utilitário interno
# ---------------------------------------------------------------------------

def _parse_dt(timestamp: str) -> datetime:
    """Converte string ISO-8601 para datetime (suporta 'T' ou espaço como separador)."""
    return datetime.fromisoformat(timestamp.replace(" ", "T"))
