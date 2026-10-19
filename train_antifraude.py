"""
train_antifraude.py
===================
Treina um modelo LightGBM para detecção de fraudes em transações de cartão.

Estratégia:
  - Divisão temporal estrita: 80% passado → treino, 20% futuro → teste.
  - Desbalanceamento tratado via scale_pos_weight (razão negativo/positivo).
  - Busca de hiperparâmetros com Optuna (maximizando PR-AUC em validação interna).
  - Avaliação final: PR-AUC, Precision, Recall, F1, matriz de confusão e curva P-R.
  - Modelo salvo com joblib.

Uso:
    python train_antifraude.py [--trials N]   # padrão: 50 trials
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Treina modelo antifraude com Optuna")
parser.add_argument(
    "--trials", type=int, default=50,
    help="Número de trials Optuna (padrão: 50)",
)
args, _ = parser.parse_known_args()
N_TRIALS = args.trials

# ─── Configurações ─────────────────────────────────────────────────────────────

FILE_PATH   = os.path.join(os.path.dirname(__file__),
                           "case_transacoes_cartoes_sintetico_900k.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "modelo_antifraude.pkl")
CHUNK_SIZE  = 100_000
TRAIN_RATIO = 0.80          # 80% mais antigos para treino
VAL_RATIO   = 0.20          # 20% do treino como validação interna p/ Optuna
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

X_train_full = train_df[feature_cols].copy()
y_train_full = train_df[TARGET_COL].values

X_test  = test_df[feature_cols].copy()
y_test  = test_df[TARGET_COL].values

# Converte colunas categóricas para o tipo que o LightGBM espera
for c in cat_features:
    X_train_full[c] = X_train_full[c].astype("category")
    X_test[c]       = X_test[c].astype("category")

# ─── 6. Divisão interna para validação Optuna (temporal) ───────────────────────

val_split_idx   = int(len(X_train_full) * (1 - VAL_RATIO))
X_tr  = X_train_full.iloc[:val_split_idx]
y_tr  = y_train_full[:val_split_idx]
X_val = X_train_full.iloc[val_split_idx:]
y_val = y_train_full[val_split_idx:]

n_neg_tr = int((y_tr == 0).sum())
n_pos_tr = int((y_tr == 1).sum())
base_scale_pos_weight = n_neg_tr / n_pos_tr if n_pos_tr > 0 else 1.0

print(f"\nValidacao interna Optuna: {len(X_tr):,} treino / {len(X_val):,} val")
print(f"  scale_pos_weight base (neg/pos no sub-treino): "
      f"{base_scale_pos_weight:.1f}")

# ─── 7. Busca de hiperparâmetros com Optuna ────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    """Retorna PR-AUC no conjunto de validação interna."""

    params = {
        "objective":        "binary",
        "metric":           "average_precision",
        "verbosity":        -1,
        "boosting_type":    "gbdt",
        "random_state":     RANDOM_SEED,
        "n_jobs":           -1,
        # ── Hiperparâmetros otimizados ──────────────────────────────────────
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3,
                                                  log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "scale_pos_weight":  trial.suggest_float(
                                 "scale_pos_weight",
                                 max(1.0, base_scale_pos_weight * 0.5),
                                 base_scale_pos_weight * 2.0,
                             ),
        # ── Regularização (espaço de busca moderado) ────────────────────────
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "max_depth":         trial.suggest_int("max_depth", -1, 12),
        # ── n_estimators alto; early stopping garante o número certo ────────
        "n_estimators":      1000,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        categorical_feature=cat_features,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    y_prob_val = model.predict_proba(X_val)[:, 1]
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    return pr_auc_val


print(f"\n{'='*55}")
print(f"  OPTUNA — busca de hiperparâmetros ({N_TRIALS} trials)")
print(f"{'='*55}")

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_SEED),
    study_name="antifraude_lgbm",
)
study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
)

best_params = study.best_params
best_val_pr_auc = study.best_value

print(f"\nMelhor PR-AUC (validação interna): {best_val_pr_auc:.4f}")
print("Melhores hiperparâmetros:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ─── 8. Treinamento final com os melhores parâmetros ──────────────────────────

# Calcula scale_pos_weight sobre o conjunto de treino completo
n_neg_full = int((y_train_full == 0).sum())
n_pos_full = int((y_train_full == 1).sum())

final_params = {
    "objective":        "binary",
    "metric":           "average_precision",
    "boosting_type":    "gbdt",
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbose":          -1,
    # Valores encontrados pelo Optuna
    "num_leaves":        best_params["num_leaves"],
    "learning_rate":     best_params["learning_rate"],
    "min_child_samples": best_params["min_child_samples"],
    # scale_pos_weight: recalculado para o treino completo mas respeitando a
    # proporção ótima encontrada pelo Optuna em relação ao base_scale_pos_weight
    "scale_pos_weight":  best_params["scale_pos_weight"]
                         / base_scale_pos_weight
                         * (n_neg_full / n_pos_full if n_pos_full > 0 else 1.0),
    "feature_fraction":  best_params["feature_fraction"],
    "bagging_fraction":  best_params["bagging_fraction"],
    "bagging_freq":      best_params["bagging_freq"],
    "reg_alpha":         best_params["reg_alpha"],
    "reg_lambda":        best_params["reg_lambda"],
    "max_depth":         best_params["max_depth"],
    # n_estimators com early stopping no conjunto de validação total
    "n_estimators":      1000,
}

print(f"\nTreinamento final (treino completo + early stopping na val) ...")
model = lgb.LGBMClassifier(**final_params)
model.fit(
    X_train_full, y_train_full,
    categorical_feature=cat_features,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50),
    ],
)
print("Treinamento final concluido.")

# ─── 9. Avaliação ──────────────────────────────────────────────────────────────

y_prob  = model.predict_proba(X_test)[:, 1]
pr_auc  = average_precision_score(y_test, y_prob)

# Threshold padrão (0.5) — ajuste conforme a curva P-R abaixo
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "="*55)
print("  METRICAS DE AVALIACAO (conjunto de teste)")
print("="*55)
print(f"  PR-AUC (val interna, melhor trial) : {best_val_pr_auc:.4f}")
print(f"  PR-AUC (teste final)               : {pr_auc:.4f}")
print(f"  Precision : {precision:.4f}  (threshold={threshold})")
print(f"  Recall    : {recall:.4f}  (threshold={threshold})")
print(f"  F1-Score  : {f1:.4f}  (threshold={threshold})")
print("="*55)

# Matriz de Confusao
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusao:")
print(f"  TN={cm[0,0]:>6,}  FP={cm[0,1]:>6,}")
print(f"  FN={cm[1,0]:>6,}  TP={cm[1,1]:>6,}")

# ─── 10. Curva Precision-Recall ─────────────────────────────────────────────────

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

# ─── 11. Importância das features (top-20) ─────────────────────────────────────

importance = pd.Series(
    model.feature_importances_,
    index=feature_cols,
).sort_values(ascending=False)

print("\nTop-20 features por importancia (gain):")
print(importance.head(20).to_string())

# ─── 12. Histórico de otimização Optuna ────────────────────────────────────────

trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
trials_df = trials_df.sort_values("value", ascending=False)
print("\nTop-10 trials Optuna (PR-AUC val):")
print(trials_df.head(10).to_string(index=False))

# ─── 13. Salvamento do modelo ──────────────────────────────────────────────────

joblib.dump(model, MODEL_PATH)
print(f"\nModelo salvo em: {MODEL_PATH}")

# ─── 14. Dica de ajuste de threshold ──────────────────────────────────────────

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
