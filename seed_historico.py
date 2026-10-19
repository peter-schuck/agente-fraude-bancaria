"""
seed_historico.py
=================
Script utilitário de carga inicial para o sistema antifraude.

Lê o arquivo ``case_transacoes_cartoes_sintetico_900k.csv`` em blocos
(chunks), aplica o pipeline de ``feature_engineering.build_velocity_features``
e popula o estado global ``transacoes_db`` do ``main.py`` com uma amostra
representativa de transações históricas reais, preservando a proporção de
fraudes do dataset.

Pode ser usado de duas formas:

1. **Script standalone** — para pré-visualizar a amostra gerada ou depurar:

       python seed_historico.py

2. **Módulo importável** — chamado pelo lifespan do ``main.py`` quando a
   variável de ambiente ``SEED_HISTORICO=1`` estiver definida:

       SEED_HISTORICO=1 uvicorn main:app --reload

Variáveis de ambiente relevantes
---------------------------------
SEED_HISTORICO      "1" activa a carga histórica no lifespan (padrão: "0")
SEED_SAMPLE_SIZE    Número de transações na amostra final   (padrão: 500)
SEED_CHUNK_SIZE     Linhas lidas por bloco do CSV           (padrão: 100 000)
SEED_RANDOM_SEED    Semente aleatória para reproducibilidade (padrão: 42)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

from feature_engineering import build_velocity_features

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
CSV_PATH  = BASE_DIR / "case_transacoes_cartoes_sintetico_900k.csv"

# Colunas de identificação a descartar (PII / IDs internos)
_ID_COLS = {
    "fdz_uuid", "event_external_id", "lifecycle_id",
    "card_pan", "account_id",
    "customer_citizen_id_number", "customer_ssn", "customer_tax_number",
    "customer_passport_number", "customer_driver_license",
    "cardholder_name", "customer_name", "customer_email", "customer_phone",
    "event_received_at", "local_datetime", "transmission_datetime",
    "capture_date", "dt_partition", "timestamp1", "fdz_timestamp",
}

# Colunas que vazam informações pós-evento (não disponíveis na inferência)
_LEAKAGE_COLS = {
    "fraud", "feedback_label", "fraud_score",
    "dispute_reason", "dispute_start_date", "dispute_end_date",
    "dispute_feedback_notes", "dispute_notes", "dispute_refunded",
    "dispute_submitted_by", "fdz_feedbacks", "fdz_triggered_actions",
    "oob_response_type", "oob_start_date", "oob_end_date",
    "oob_feedback_notes", "oob_notes",
    "action_code", "approval_code", "decision",
}

# Mapeamento: nomes do CSV → nomes canônicos usados pelo pipeline
_COL_RENAME = {
    "customer_id":       "client_id",
    "event_occurred_at": "timestamp",
    "acceptor_id":       "merchant_id",
}

# Colunas obrigatórias do pipeline de feature_engineering
_REQUIRED_COLS = {"client_id", "card_id", "timestamp", "amount", "merchant_id"}

# Coluna de rótulo no CSV (usada apenas para amostragem estratificada)
_FRAUD_COL = "fraud"


# ---------------------------------------------------------------------------
# Funções privadas
# ---------------------------------------------------------------------------

def _ler_chunks_relevantes(
    csv_path: Path,
    chunk_size: int,
) -> pd.DataFrame:
    """
    Lê o CSV em chunks, mantendo apenas as linhas com rótulo de fraude
    definido (True/False) e descartando colunas de PII e leakage.

    Retorna um DataFrame consolidado com as colunas renomeadas para os
    nomes canônicos do pipeline.
    """
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        # Mantém somente linhas com rótulo definido; converte para int8
        if _FRAUD_COL in chunk.columns:
            chunk = chunk.dropna(subset=[_FRAUD_COL]).copy()
            chunk[_FRAUD_COL] = chunk[_FRAUD_COL].map(
                {True: 1, False: 0, 1: 1, 0: 0}
            ).astype("int8")

        # Renomeia para nomes canônicos
        rename = {k: v for k, v in _COL_RENAME.items() if k in chunk.columns}
        chunk = chunk.rename(columns=rename)

        # Remove colunas de PII / leakage — fraud é mantido para amostragem
        # estratificada e diagnóstico; será removido em _construir_transacoes.
        leakage_sem_fraud = _LEAKAGE_COLS - {_FRAUD_COL}
        drop = (_ID_COLS | leakage_sem_fraud) & set(chunk.columns)
        chunk = chunk.drop(columns=list(drop))

        # Descarta se faltar coluna obrigatória
        if not _REQUIRED_COLS.issubset(chunk.columns):
            continue

        chunks.append(chunk)

    if not chunks:
        raise RuntimeError(
            f"Nenhum dado válido encontrado em '{csv_path}'. "
            "Verifique o caminho e o formato do arquivo."
        )

    return pd.concat(chunks, ignore_index=True)


def _amostrar_estratificado(
    df: pd.DataFrame,
    n: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Retorna uma amostra de tamanho ``n`` preservando a proporção de fraudes
    do DataFrame original.  Se o DataFrame for menor que ``n``, retorna tudo.
    """
    if len(df) <= n:
        return df.reset_index(drop=True)

    if _FRAUD_COL in df.columns:
        # Amostragem estratificada: preserva proporção de fraudes.
        # Usa concat de amostras por estrato para evitar que groupby/apply
        # descarte a coluna de agrupamento em pandas >= 2.2.
        frac = n / len(df)
        parts = [
            grp.sample(frac=frac, random_state=random_seed)
            for _, grp in df.groupby(_FRAUD_COL, sort=False)
        ]
        return pd.concat(parts, ignore_index=True)

    return df.sample(n=n, random_state=random_seed).reset_index(drop=True)


def _aplicar_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica ``build_velocity_features`` sobre o DataFrame completo e adiciona
    as features de velocidade calculadas sobre o histórico real.
    """
    # Garante que timestamp seja datetime antes de passar ao pipeline
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df = build_velocity_features(
        df,
        client_col="client_id",
        card_col="card_id",
        ts_col="timestamp",
        amount_col="amount",
        merchant_col="merchant_id",
    )

    # Preenche NaN nas features de contagem (primeira transação do grupo)
    for col in ("card_txn_count_10min", "card_txn_count_1h", "high_value_flag"):
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def _construir_transacoes(df: pd.DataFrame, id_offset: int) -> list[dict]:
    """
    Converte o DataFrame processado para a lista de dicts compatível com
    o schema ``Transacao`` do ``main.py``.

    Campos derivados pelo modelo (``escore_risco``, ``risco``, ``suspeita``)
    são inicializados com valores neutros; o lifespan do main.py os recalcula
    via ``_pontuar_todas_transacoes()`` logo após a inserção.
    """
    transacoes: list[dict] = []

    # Timestamp de volta para string ISO 8601 (sem tz, compatível com o schema)
    ts_series = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    for i, (_, row) in enumerate(df.iterrows()):
        txn = row.to_dict()

        # Campos obrigatórios do schema Transacao
        txn["id"]           = id_offset + i
        txn["valor"]        = float(txn.get("amount", 0.0))
        txn["local"]        = str(txn.get("acceptor_city", "")) or None
        txn["categoria"]    = str(txn.get("tipo_transacao", "")) or None
        txn["timestamp"]    = ts_series.iloc[i]
        txn["escore_risco"] = 0.0
        txn["risco"]        = "baixo"
        txn["suspeita"]     = False

        # Remove colunas que não pertencem ao schema de transação ou que
        # já foram mapeadas acima, para não poluir o payload
        for col in (_FRAUD_COL, "merchant_id"):
            txn.pop(col, None)

        transacoes.append(txn)

    return transacoes


# ---------------------------------------------------------------------------
# Função pública — chamada pelo lifespan do main.py
# ---------------------------------------------------------------------------

def carregar_historico(
    transacoes_db: list[dict],
    id_offset: int = 1,
    *,
    sample_size: int | None = None,
    chunk_size: int | None = None,
    random_seed: int | None = None,
) -> int:
    """
    Lê o CSV histórico, aplica feature engineering e popula ``transacoes_db``
    com a amostra gerada.

    Parâmetros
    ----------
    transacoes_db : list[dict]
        Lista em memória do ``main.py`` onde as transações serão inseridas.
    id_offset : int
        Primeiro ID a ser atribuído (deve ser maior que o maior ID já existente).
    sample_size : int, optional
        Sobrescreve a variável de ambiente ``SEED_SAMPLE_SIZE``.
    chunk_size : int, optional
        Sobrescreve a variável de ambiente ``SEED_CHUNK_SIZE``.
    random_seed : int, optional
        Sobrescreve a variável de ambiente ``SEED_RANDOM_SEED``.

    Retorna
    -------
    int
        Novo valor para ``_proximo_id`` (id_offset + número de itens inseridos).
    """
    _sample_size  = sample_size  or int(os.getenv("SEED_SAMPLE_SIZE", "500"))
    _chunk_size   = chunk_size   or int(os.getenv("SEED_CHUNK_SIZE",  "100000"))
    _random_seed  = random_seed  or int(os.getenv("SEED_RANDOM_SEED", "42"))

    print(
        f"[seed_historico] Carregando historico do CSV "
        f"(amostra={_sample_size:,}, chunk={_chunk_size:,}) ..."
    )

    df = _ler_chunks_relevantes(CSV_PATH, _chunk_size)
    print(f"[seed_historico] {len(df):,} linhas com rotulo carregadas.")

    df = _amostrar_estratificado(df, _sample_size, _random_seed)
    print(f"[seed_historico] Amostra estratificada: {len(df):,} transacoes.")

    df = _aplicar_feature_engineering(df)
    print("[seed_historico] Feature engineering aplicado.")

    transacoes = _construir_transacoes(df, id_offset)
    transacoes_db.extend(transacoes)

    novo_proximo_id = id_offset + len(transacoes)
    print(
        f"[seed_historico] {len(transacoes):,} transacoes historicas inseridas "
        f"(IDs {id_offset}-{novo_proximo_id - 1})."
    )
    return novo_proximo_id


# ---------------------------------------------------------------------------
# Execucao standalone -- visualizacao e diagnostico
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    chunk_size  = 100_000
    random_seed = 42

    print(f"Lendo CSV em chunks de {chunk_size:,} linhas ...")
    df = _ler_chunks_relevantes(CSV_PATH, chunk_size)
    print(f"   Total com rotulo: {len(df):,} linhas\n")

    if _FRAUD_COL in df.columns:
        dist = df[_FRAUD_COL].value_counts(normalize=True).mul(100).round(2)
        print(f"Distribuicao de fraudes (dataset completo):\n{dist.to_string()}\n")

    df_sample = _amostrar_estratificado(df, sample_size, random_seed)

    if _FRAUD_COL in df_sample.columns:
        dist_s = df_sample[_FRAUD_COL].value_counts(normalize=True).mul(100).round(2)
        print(f"Distribuicao de fraudes (amostra de {sample_size}):\n{dist_s.to_string()}\n")

    df_feat = _aplicar_feature_engineering(df_sample)

    vel_cols = [
        "card_txn_count_10min", "card_txn_count_1h",
        "seconds_since_last_txn", "high_value_flag",
    ]
    print("Estatisticas das features de velocidade:")
    print(df_feat[vel_cols].describe().to_string())

    print(f"\nAmostra das primeiras 5 transacoes:")
    print(
        df_feat[
            ["client_id", "card_id", "timestamp", "amount"] + vel_cols
        ].head(5).to_string(index=False)
    )
