"""
feature_engineering.py
=======================
Geração de features de velocidade, comportamento temporal e variáveis
preditivas avançadas para detecção de fraudes em transações de cartão.

Requisitos:
    pip install pandas numpy

Uso:
    from feature_engineering import build_velocity_features

    df_features = build_velocity_features(df)

Features geradas
----------------
Velocidade (por cartão):
    card_txn_count_10min        – qtd. de transações nos últimos 10 min
    card_txn_count_1h           – qtd. de transações na última 1 hora

Comportamento temporal (por cliente):
    seconds_since_last_txn      – segundos desde a transação anterior

Valor (por cliente):
    high_value_flag             – 1 se valor > N× média histórica expandida

Geográfica (por cartão) [NOVA]:
    haversine_dist_km           – distância em km entre o terminal da transação
                                  atual e o terminal imediatamente anterior do
                                  mesmo cartão (NaN na primeira transação)

Médias móveis ponderadas por categoria MCC (por cartão) [NOVA]:
    ewm_amount_by_mcc           – média exponencialmente ponderada (EWM) do
                                  valor das transações anteriores do cartão na
                                  mesma categoria MCC (sem data leakage)

Rejeições acumuladas (por cartão) [NOVA]:
    cumulative_rejected_txns    – contagem acumulada de transações rejeitadas
                                  anteriores do mesmo cartão
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── Constantes ────────────────────────────────────────────────────────────────

WINDOW_10MIN = "10min"
WINDOW_1H = "1h"
HIGH_VALUE_MULTIPLIER = 2.0   # N× acima da média histórica → flag ativa

# Raio médio da Terra em quilômetros (WGS-84)
_EARTH_RADIUS_KM = 6_371.0

# Fator de suavização para a média ponderada exponencial por MCC
_EWM_ALPHA = 0.3

# Códigos de decisão/status que representam uma transação rejeitada
_REJECTED_DECISION_VALUES = {"deny", "decline", "declined", "rejected", "block"}
_REJECTED_STATUS_VALUES   = {"declined", "rejected", "denied", "blocked"}


# ─── Funções auxiliares ────────────────────────────────────────────────────────

def _sort_and_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Garante ordenação por timestamp e índice limpo."""
    return df.sort_values(ts_col).reset_index(drop=True)


def _rolling_count(
    df: pd.DataFrame,
    group_col: str,
    ts_col: str,
    window: str,
) -> pd.Series:
    """
    Conta transações de cada grupo dentro de uma janela temporal rolante.

    Usa rolling baseado em tempo com closed='left' para excluir a própria
    transação corrente (contagem de eventos *anteriores* na janela).
    """
    counts = (
        df.groupby(group_col, sort=False)[ts_col]
        .transform(
            lambda s: (
                s.reset_index(drop=True)
                 .rename("ts")
                 .to_frame()
                 .set_index("ts")
                 .assign(_v=1)
                 ["_v"]
                 .rolling(window, closed="left")
                 .count()
                 .values
            )
        )
    )
    return counts.fillna(0).astype("int32")


def _time_since_last(df: pd.DataFrame, group_col: str, ts_col: str) -> pd.Series:
    """
    Diferença em segundos entre a transação atual e a anterior do mesmo grupo.
    Primeira transação do grupo recebe NaN.
    """
    return (
        df.groupby(group_col, sort=False)[ts_col]
        .transform(lambda s: s.diff().dt.total_seconds())
    )


def _high_value_flag(
    df: pd.DataFrame,
    group_col: str,
    amount_col: str,
    multiplier: float = HIGH_VALUE_MULTIPLIER,
) -> pd.Series:
    """
    Flag binária: 1 se o valor da transação supera `multiplier` × média
    histórica *expandida* do cliente (calculada apenas com transações
    anteriores para evitar data leakage).
    """
    def _expanding_flag(s: pd.Series) -> pd.Series:
        # shift(1): média considera apenas transações anteriores
        historical_mean = s.shift(1).expanding().mean()
        return (s > multiplier * historical_mean).astype("int8")

    return df.groupby(group_col, sort=False)[amount_col].transform(_expanding_flag)


def _haversine_dist_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Distância em quilômetros entre dois pontos geográficos usando a fórmula de
    Haversine.  Todos os inputs são séries com valores em graus decimais.
    Retorna NaN onde qualquer coordenada for NaN.
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat   = np.radians(lat2 - lat1)
    dlon   = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    return _EARTH_RADIUS_KM * c


def _haversine_consecutive(
    df: pd.DataFrame,
    card_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.Series:
    """
    Para cada transação, calcula a distância em km entre as coordenadas do
    terminal atual e as do terminal da transação imediatamente anterior do
    mesmo cartão.  A primeira transação de cada cartão recebe NaN.

    Pressupõe que ``df`` já está ordenado por timestamp.
    """
    prev_lat = df.groupby(card_col, sort=False)[lat_col].shift(1)
    prev_lon = df.groupby(card_col, sort=False)[lon_col].shift(1)

    return _haversine_dist_km(
        prev_lat, prev_lon,
        df[lat_col], df[lon_col],
    )


def _ewm_amount_by_mcc(
    df: pd.DataFrame,
    card_col: str,
    mcc_col: str,
    amount_col: str,
    alpha: float = _EWM_ALPHA,
) -> pd.Series:
    """
    Média móvel exponencialmente ponderada (EWM) do valor das transações do
    mesmo cartão na mesma categoria MCC, usando apenas transações *anteriores*
    (shift(1) antes do EWM para evitar data leakage).

    Retorna NaN para a primeira transação de cada (cartão, MCC).
    """
    def _ewm_shift(s: pd.Series) -> pd.Series:
        return s.shift(1).ewm(alpha=alpha, adjust=False).mean()

    return (
        df.groupby([card_col, mcc_col], sort=False)[amount_col]
        .transform(_ewm_shift)
    )


def _cumulative_rejected(
    df: pd.DataFrame,
    card_col: str,
    decision_col: str | None,
    status_col: str | None,
) -> pd.Series:
    """
    Contagem acumulada de transações *rejeitadas anteriores* para o mesmo
    cartão.  Usa ``decision_col`` (preferencial) ou ``status_col`` para
    identificar rejeições.  Aplica shift(1) para não incluir a transação atual.

    Pressupõe que ``df`` já está ordenado por timestamp.
    """
    if decision_col and decision_col in df.columns:
        is_rejected = df[decision_col].str.lower().isin(_REJECTED_DECISION_VALUES).astype("int8")
    elif status_col and status_col in df.columns:
        is_rejected = df[status_col].str.lower().isin(_REJECTED_STATUS_VALUES).astype("int8")
    else:
        # Colunas não disponíveis: retorna zeros
        return pd.Series(0, index=df.index, dtype="int32")

    return (
        df.assign(_rejected=is_rejected)
        .groupby(card_col, sort=False)["_rejected"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
        .astype("int32")
    )


# ─── Função principal ──────────────────────────────────────────────────────────

def build_velocity_features(
    df: pd.DataFrame,
    *,
    client_col: str = "client_id",
    card_col: str = "card_id",
    ts_col: str = "timestamp",
    amount_col: str = "amount",
    merchant_col: str = "merchant_id",
    lat_col: str = "terminal_latitude",
    lon_col: str = "terminal_longitude",
    mcc_col: str = "mcc",
    decision_col: str = "decision",
    status_col: str = "transaction_status",
    high_value_multiplier: float = HIGH_VALUE_MULTIPLIER,
) -> pd.DataFrame:
    """
    Adiciona features de velocidade, comportamento temporal e variáveis
    preditivas avançadas a um DataFrame de transações de cartões.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada. Deve conter pelo menos as colunas identificadas
        pelos demais parâmetros.
    client_col : str
        Coluna de identificação do cliente.
    card_col : str
        Coluna de identificação do cartão.
    ts_col : str
        Coluna de timestamp (será convertida para datetime se necessário).
    amount_col : str
        Coluna de valor da transação.
    merchant_col : str
        Coluna de identificação do estabelecimento.
    lat_col : str
        Coluna de latitude do terminal (graus decimais). Usada na distância
        Haversine; linhas sem coordenada recebem NaN na feature.
    lon_col : str
        Coluna de longitude do terminal (graus decimais).
    mcc_col : str
        Coluna de código de categoria do estabelecimento (MCC). Usada na
        média ponderada por categoria.
    decision_col : str
        Coluna de decisão da transação (p.ex. "approve"/"deny"). Usada para
        identificar rejeições; ignorada se ausente do DataFrame.
    status_col : str
        Coluna de status da transação (p.ex. "approved"/"declined"). Usada
        como fallback quando ``decision_col`` não estiver disponível.
    high_value_multiplier : float
        Fator multiplicador para a flag de valor alto.

    Retorna
    -------
    pd.DataFrame
        Cópia do DataFrame com as novas colunas adicionadas:

        Existentes:
        - ``card_txn_count_10min``   : qtd. de transações do cartão nos últimos 10 min
        - ``card_txn_count_1h``      : qtd. de transações do cartão na última 1 hora
        - ``seconds_since_last_txn`` : segundos desde a transação anterior do cliente
        - ``high_value_flag``        : 1 se valor > N× média histórica do cliente

        Novas:
        - ``haversine_dist_km``      : distância km entre terminais consecutivos do cartão
        - ``ewm_amount_by_mcc``      : média exponencial do valor por categoria MCC
        - ``cumulative_rejected_txns``: contagem acumulada de rejeições anteriores do cartão
    """
    # ── Validação de colunas obrigatórias ──────────────────────────────────
    required = {client_col, card_col, ts_col, amount_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no DataFrame: {missing}")

    result = _sort_and_index(df.copy(), ts_col)

    # ── Converte timestamp ─────────────────────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(result[ts_col]):
        result[ts_col] = pd.to_datetime(result[ts_col], utc=True, errors="coerce")

    # ── Features de velocidade (por cartão) ───────────────────────────────
    result["card_txn_count_10min"] = _rolling_count(
        result, card_col, ts_col, WINDOW_10MIN
    )
    result["card_txn_count_1h"] = _rolling_count(
        result, card_col, ts_col, WINDOW_1H
    )

    # ── Diferença temporal (por cliente) ──────────────────────────────────
    result["seconds_since_last_txn"] = _time_since_last(result, client_col, ts_col)

    # ── Flag de valor alto (por cliente) ──────────────────────────────────
    result["high_value_flag"] = _high_value_flag(
        result, client_col, amount_col, high_value_multiplier
    )

    # ── [NOVA] Distância Haversine entre terminais consecutivos ───────────
    if lat_col in result.columns and lon_col in result.columns:
        result[lat_col] = pd.to_numeric(result[lat_col], errors="coerce")
        result[lon_col] = pd.to_numeric(result[lon_col], errors="coerce")
        result["haversine_dist_km"] = _haversine_consecutive(
            result, card_col, lat_col, lon_col
        )
    else:
        result["haversine_dist_km"] = np.nan

    # ── [NOVA] Média exponencial ponderada do valor por categoria MCC ─────
    if mcc_col in result.columns:
        result[mcc_col] = result[mcc_col].astype(str)
        result["ewm_amount_by_mcc"] = _ewm_amount_by_mcc(
            result, card_col, mcc_col, amount_col
        )
    else:
        result["ewm_amount_by_mcc"] = np.nan

    # ── [NOVA] Contagem acumulada de rejeições anteriores por cartão ──────
    result["cumulative_rejected_txns"] = _cumulative_rejected(
        result,
        card_col,
        decision_col if decision_col in result.columns else None,
        status_col   if status_col   in result.columns else None,
    )

    return result


# ─── Execução autônoma para testes rápidos ────────────────────────────────────

if __name__ == "__main__":
    import os

    CSV_PATH = os.path.join(os.path.dirname(__file__),
                            "case_transacoes_cartoes_sintetico_900k.csv")
    CHUNK_SIZE = 100_000

    print(f"Processando '{CSV_PATH}' em chunks de {CHUNK_SIZE:,} linhas …\n")

    # Mapeamento para nomes canônicos do dataset real
    COL_MAP = {
        "customer_id":       "client_id",
        "card_id":           "card_id",
        "event_occurred_at": "timestamp",
        "amount":            "amount",
        "acceptor_id":       "merchant_id",
        # coordenadas, MCC e status já existem com nomes padrão no dataset
    }

    ALL_FEATURES = [
        "client_id", "card_id", "timestamp", "amount",
        "card_txn_count_10min", "card_txn_count_1h",
        "seconds_since_last_txn", "high_value_flag",
        "haversine_dist_km", "ewm_amount_by_mcc", "cumulative_rejected_txns",
    ]

    chunks_out: list[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
        # Renomeia apenas as colunas que existirem no chunk
        rename = {k: v for k, v in COL_MAP.items() if k in chunk.columns}
        chunk = chunk.rename(columns=rename)

        chunk_feat = build_velocity_features(chunk)
        chunks_out.append(chunk_feat[[c for c in ALL_FEATURES if c in chunk_feat.columns]])
        print(f"  chunk {i+1}: {len(chunk):,} linhas processadas")

    result_df = pd.concat(chunks_out, ignore_index=True)

    print(f"\nTotal processado: {len(result_df):,} linhas")
    print("\n── Amostra das features geradas ──")
    print(result_df.head(10).to_string(index=False))

    print("\n── Distribuição da flag de valor alto ──")
    print(result_df["high_value_flag"].value_counts().to_string())

    print("\n── Estatísticas de velocidade ──")
    print(result_df[["card_txn_count_10min", "card_txn_count_1h",
                     "seconds_since_last_txn"]].describe().to_string())

    print("\n── Estatísticas das features avançadas ──")
    adv_cols = ["haversine_dist_km", "ewm_amount_by_mcc", "cumulative_rejected_txns"]
    adv_present = [c for c in adv_cols if c in result_df.columns]
    if adv_present:
        print(result_df[adv_present].describe().to_string())

    print("\n── Distribuição de rejeições acumuladas ──")
    if "cumulative_rejected_txns" in result_df.columns:
        print(result_df["cumulative_rejected_txns"].value_counts().sort_index().head(10).to_string())
