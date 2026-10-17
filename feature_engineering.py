"""
feature_engineering.py
=======================
Geração de features de velocidade e comportamento temporal para
detecção de fraudes em transações de cartão.

Requisitos:
    pip install pandas numpy

Uso:
    from feature_engineering import build_velocity_features

    df_features = build_velocity_features(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── Constantes ────────────────────────────────────────────────────────────────

WINDOW_10MIN = "10min"
WINDOW_1H = "1h"
HIGH_VALUE_MULTIPLIER = 2.0   # N× acima da média histórica → flag ativa


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
    return counts.astype("int32")


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


# ─── Função principal ──────────────────────────────────────────────────────────

def build_velocity_features(
    df: pd.DataFrame,
    *,
    client_col: str = "client_id",
    card_col: str = "card_id",
    ts_col: str = "timestamp",
    amount_col: str = "amount",
    merchant_col: str = "merchant_id",
    high_value_multiplier: float = HIGH_VALUE_MULTIPLIER,
) -> pd.DataFrame:
    """
    Adiciona features de velocidade e comportamento temporal a um DataFrame
    de transações de cartões.

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
        Coluna de identificação do estabelecimento (não usada nas features
        atuais, mas mantida para extensões futuras).
    high_value_multiplier : float
        Fator multiplicador para a flag de valor alto.

    Retorna
    -------
    pd.DataFrame
        Cópia do DataFrame com as novas colunas adicionadas:
        - ``card_txn_count_10min``  : qtd. de transações do cartão nos últimos 10 min
        - ``card_txn_count_1h``     : qtd. de transações do cartão na última 1 hora
        - ``seconds_since_last_txn``: segundos desde a transação anterior do cliente
        - ``high_value_flag``       : 1 se valor > N× média histórica do cliente
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
        "customer_id":  "client_id",
        "card_id":      "card_id",
        "event_occurred_at": "timestamp",
        "amount":       "amount",
        "acceptor_id":  "merchant_id",
    }

    chunks_out: list[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
        # Renomeia apenas as colunas que existirem no chunk
        rename = {k: v for k, v in COL_MAP.items() if k in chunk.columns}
        chunk = chunk.rename(columns=rename)

        chunk_feat = build_velocity_features(chunk)
        chunks_out.append(chunk_feat[
            ["client_id", "card_id", "timestamp", "amount",
             "card_txn_count_10min", "card_txn_count_1h",
             "seconds_since_last_txn", "high_value_flag"]
        ])
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
