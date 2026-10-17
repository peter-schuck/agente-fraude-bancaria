import pandas as pd
import numpy as np

FILE_PATH = "antifraude\case_transacoes_cartoes_sintetico_900k.csv"
CHUNK_SIZE = 50_000
NULL_THRESHOLD = 50.0  # % para considerar alta ausência

# ── 1. Leitura em chunks para acumular estatísticas ──────────────────────────
total_rows = 0
null_counts = None
dtypes_sample = None
numeric_sum = None
numeric_sum_sq = None
numeric_n = None

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    total_rows += len(chunk)

    if null_counts is None:
        null_counts = chunk.isnull().sum()
        dtypes_sample = chunk.dtypes
    else:
        null_counts = null_counts.add(chunk.isnull().sum(), fill_value=0)

    num_cols = chunk.select_dtypes(include=[np.number]).columns
    if numeric_sum is None:
        numeric_sum = chunk[num_cols].sum()
        numeric_sum_sq = (chunk[num_cols] ** 2).sum()
        numeric_n = chunk[num_cols].count()
    else:
        numeric_sum = numeric_sum.add(chunk[num_cols].sum(), fill_value=0)
        numeric_sum_sq = numeric_sum_sq.add((chunk[num_cols] ** 2).sum(), fill_value=0)
        numeric_n = numeric_n.add(chunk[num_cols].count(), fill_value=0)

print(f"\nTotal de linhas: {total_rows:,}")
print(f"Total de colunas: {len(null_counts)}")

# ── 2. Porcentagem de nulos por coluna ───────────────────────────────────────
null_pct = (null_counts / total_rows * 100).sort_values(ascending=False)

print("\n── Porcentagem de valores nulos por coluna (top 20) ──")
print(null_pct.head(20).to_string())

high_null_cols = null_pct[null_pct >= NULL_THRESHOLD]
print(f"\n── Colunas com ≥{NULL_THRESHOLD}% de valores nulos ({len(high_null_cols)}) ──")
print(high_null_cols.to_string())

# ── 3. Classificação de colunas categóricas e numéricas ─────────────────────
numeric_cols = dtypes_sample[dtypes_sample.apply(pd.api.types.is_numeric_dtype)].index.tolist()
categorical_cols = dtypes_sample[dtypes_sample == "object"].index.tolist()

print(f"\n── Colunas numéricas ({len(numeric_cols)}) ──")
print(numeric_cols)

print(f"\n── Colunas categóricas / object ({len(categorical_cols)}) ──")
print(categorical_cols)

# ── 4. Estatísticas descritivas das variáveis numéricas ─────────────────────
# Média e desvio-padrão calculados online (sem carregar tudo na memória)
mean = numeric_sum / numeric_n
variance = (numeric_sum_sq / numeric_n) - (mean ** 2)
std = variance.apply(lambda v: v ** 0.5 if v >= 0 else float("nan"))

desc = pd.DataFrame({
    "count": numeric_n,
    "mean": mean,
    "std": std,
    "null_%": null_pct.reindex(numeric_n.index),
})

print("\n── Estatísticas descritivas das colunas numéricas ──")
print(desc.to_string())

# ── 5. Matriz de correlação (amostra de 10k linhas) ──────────────────────────
sample = pd.read_csv(FILE_PATH, nrows=10_000, low_memory=False)
num_sample = sample[numeric_cols].dropna(axis=1, how="all")

# Filtra apenas colunas com pelo menos 30% de dados presentes na amostra
valid_cols = num_sample.columns[num_sample.isnull().mean() < 0.70]
corr_matrix = num_sample[valid_cols].corr()

print("\n── Matriz de correlação (amostra 10k linhas, colunas com <70% nulos) ──")
print(corr_matrix.to_string())
