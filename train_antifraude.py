"""
train_antifraude.py
===================
Treina um modelo LightGBM para detecção de fraudes em transações de cartão.

Estratégia:
  - Divisão temporal estrita: 80% passado → treino, 20% futuro → teste.
  - Desbalanceamento tratado via scale_pos_weight (razão negativo/positivo).
  - Avaliação: PR-AUC, Precision, Recall, F1, matriz de confusão e curva P-R.
  - Modelo salvo com joblib.

Uso:
    python train_antifraude.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ─── Configurações ─────────────────────────────────────────────────────────────

FILE_PATH   = os.path.join(os.path.dirname(__file__),
                           "case_transacoes_cartoes_sintetico_900k.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "modelo_antifraude.pkl")
CHUNK_SIZE  = 100_000
TRAIN_RATIO = 0.80          # 80% mais antigos para treino
RANDOM_SEED = 42

# Colunas de identificação — não entram como features
ID_COLS = [
    "fdz_uuid", "event_external_id", "lifecycle_id",
    "card_pan", "card_id", "account_id", "customer_id",
    "customer_citizen_id_number", "customer_ssn", "customer_tax_number",
    "customer_passport_number", "customer_driver_license",
    "cardholder_name", "customer_name", "customer_email", "customer_phone",
    "event_occurred_at", "event_received_at", "local_datetime",
    "transmission_datetime", "capture_date", "dt_partition",
    "timestamp1", "fdz_timestamp",
]

# Colunas que representam o alvo ou vazam informações pós-evento
LEAKAGE_COLS = [
    "fraud", "feedback_label", "fraud_score",
    "dispute_reason", "dispute_start_date", "dispute_end_date",
    "dispute_feedback_notes", "dispute_notes", "dispute_refunded",
    "dispute_submitted_by", "fdz_feedbacks", "fdz_triggered_actions",
    "oob_response_type", "oob_start_date", "oob_end_date",
    "oob_feedback_notes", "oob_notes",
    "action_code", "approval_code", "decision",
]

TIMESTAMP_COL = "event_occurred_at"
TARGET_COL    = "fraud"

# ─── 1. Leitura do dataset ─────────────────────────────────────────────────────

print("Lendo dataset ...")
chunks: list[pd.DataFrame] = []
for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"  Total de linhas: {len(df):,}  |  colunas: {df.shape[1]}")

# ─── 2. Preparação do alvo ─────────────────────────────────────────────────────

# 'fraud' é bool/NaN; NaN indica transações sem rótulo definitivo → descartamos
df[TARGET_COL] = df[TARGET_COL].map({True: 1, False: 0, 1: 1, 0: 0})
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

print(f"\nApos remover sem rotulo: {len(df):,} linhas")
print(f"  Fraude=1: {df[TARGET_COL].sum():,}  "
      f"({df[TARGET_COL].mean()*100:.4f}%)")

# ─── 3. Divisão temporal ────────────────────────────────────────────────────────

df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

cutoff_idx = int(len(df) * TRAIN_RATIO)
cutoff_time = df[TIMESTAMP_COL].iloc[cutoff_idx]

train_df = df.iloc[:cutoff_idx].copy()
test_df  = df.iloc[cutoff_idx:].copy()

print(f"\nDivisao temporal (corte: {cutoff_time})")
print(f"  Treino : {len(train_df):,} linhas  "
      f"({train_df[TARGET_COL].sum():,} fraudes, "
      f"{train_df[TARGET_COL].mean()*100:.4f}%)")
print(f"  Teste  : {len(test_df):,} linhas  "
      f"({test_df[TARGET_COL].sum():,} fraudes, "
      f"{test_df[TARGET_COL].mean()*100:.4f}%)")

# ─── 4. Seleção de features ─────────────────────────────────────────────────────

drop_cols = set(ID_COLS + LEAKAGE_COLS + [TARGET_COL])

# Colunas numéricas disponíveis
num_features = [
    c for c in df.select_dtypes(include="number").columns
    if c not in drop_cols
]

# Colunas categóricas com cardinalidade razoável (≤ 200 valores únicos)
cat_features = [
    c for c in df.select_dtypes(include=["object", "str", "category"]).columns
    if c not in drop_cols
    and df[c].nunique() <= 200
]

feature_cols = num_features + cat_features
print(f"\nFeatures selecionadas: {len(feature_cols)} "
      f"({len(num_features)} numéricas + {len(cat_features)} categóricas)")

# ─── 5. Matrizes X / y ─────────────────────────────────────────────────────────

X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET_COL].values

X_test  = test_df[feature_cols].copy()
y_test  = test_df[TARGET_COL].values

# Converte colunas categóricas para o tipo que o LightGBM espera
for c in cat_features:
    X_train[c] = X_train[c].astype("category")
    X_test[c]  = X_test[c].astype("category")

# ─── 6. Tratamento do desbalanceamento ─────────────────────────────────────────

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

print(f"\nDesbalanceamento no treino -> scale_pos_weight = {scale_pos_weight:.1f}")

# ─── 7. Treinamento LightGBM ───────────────────────────────────────────────────

lgb_params = {
    "objective":         "binary",
    "metric":            "average_precision",
    "scale_pos_weight":  scale_pos_weight,
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      RANDOM_SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

print("\nIniciando treinamento ...")
model = lgb.LGBMClassifier(**lgb_params)
model.fit(
    X_train, y_train,
    categorical_feature=cat_features,
    callbacks=[lgb.log_evaluation(period=50)],
)
print("Treinamento concluido.")

# ─── 8. Avaliação ──────────────────────────────────────────────────────────────

y_prob  = model.predict_proba(X_test)[:, 1]
pr_auc  = average_precision_score(y_test, y_prob)

# Threshold padrão (0.5) — ajuste conforme a curva P-R abaixo
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "="*55)
print("  METRICAS DE AVALIACAO")
print("="*55)
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  Precision : {precision:.4f}  (threshold={threshold})")
print(f"  Recall    : {recall:.4f}  (threshold={threshold})")
print(f"  F1-Score  : {f1:.4f}  (threshold={threshold})")
print("="*55)

# Matriz de Confusao
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusao:")
print(f"  TN={cm[0,0]:>6,}  FP={cm[0,1]:>6,}")
print(f"  FN={cm[1,0]:>6,}  TP={cm[1,1]:>6,}")

# ─── 9. Curva Precision-Recall ─────────────────────────────────────────────────

prec_curve, rec_curve, thresholds_curve = precision_recall_curve(y_test, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curva P-R
axes[0].plot(rec_curve, prec_curve, color="#3b82d4", lw=2,
             label=f"PR-AUC = {pr_auc:.4f}")
axes[0].axhline(y_test.mean(), linestyle="--", color="#7c5cd8", lw=1,
                label=f"Baseline (prevalência = {y_test.mean():.4f})")
axes[0].scatter([recall], [precision], color="red", zorder=5,
                label=f"Threshold={threshold}: P={precision:.3f}, R={recall:.3f}")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Curva Precision-Recall")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(alpha=0.3)

# Precision e Recall por threshold
axes[1].plot(thresholds_curve, prec_curve[:-1], label="Precision", color="#3b82d4", lw=2)
axes[1].plot(thresholds_curve, rec_curve[:-1],  label="Recall",    color="#e07b39", lw=2)
axes[1].axvline(threshold, linestyle="--", color="red", lw=1,
                label=f"Threshold={threshold}")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title("Precision e Recall por Threshold")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "curva_precision_recall.png")
plt.savefig(plot_path, dpi=120)
print(f"\nCurva P-R salva em: {plot_path}")
plt.show()

# Matriz de confusão visual
fig2, ax2 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Legítima", "Fraude"]).plot(
    ax=ax2, colorbar=False, cmap="Blues"
)
ax2.set_title("Matriz de Confusão")
plt.tight_layout()
cm_path = os.path.join(os.path.dirname(__file__), "matriz_confusao.png")
plt.savefig(cm_path, dpi=120)
print(f"Matriz de confusao salva em: {cm_path}")
plt.show()

# ─── 10. Importância das features (top-20) ─────────────────────────────────────

importance = pd.Series(
    model.feature_importances_,
    index=feature_cols,
).sort_values(ascending=False)

print("\nTop-20 features por importancia (gain):")
print(importance.head(20).to_string())

# ─── 11. Salvamento do modelo ──────────────────────────────────────────────────

joblib.dump(model, MODEL_PATH)
print(f"\nModelo salvo em: {MODEL_PATH}")

# ─── 12. Dica de ajuste de threshold ──────────────────────────────────────────

# Encontra threshold que maximiza F1
f1_scores = []
for t in thresholds_curve:
    preds = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds, zero_division=0))

best_idx       = int(np.argmax(f1_scores))
best_threshold = float(thresholds_curve[best_idx])
best_f1        = float(f1_scores[best_idx])
best_prec      = float(prec_curve[best_idx])
best_rec       = float(rec_curve[best_idx])

print("\n-- Threshold otimo para F1 --")
print(f"  Threshold : {best_threshold:.4f}")
print(f"  Precision : {best_prec:.4f}")
print(f"  Recall    : {best_rec:.4f}")
print(f"  F1-Score  : {best_f1:.4f}")
