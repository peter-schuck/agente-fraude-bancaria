"""
Sistema Antifraude Bancário — Backend com FastAPI + LightGBM.

O modelo serializado modelo_antifraude.pkl é carregado na inicialização via
lifespan. Os endpoints de transações aplicam o pipeline de engenharia de
features (feature_engineering.build_velocity_features) e usam predict_proba
do LightGBM para calcular o escore e nível de risco de cada transação.

A cada predição, valores SHAP (TreeExplainer) são calculados em tempo real
e retornados no payload como `shap_values` — dicionário {feature: shap_value}
ordenado por magnitude decrescente, limitado às 15 features mais influentes.

Execute com: uvicorn main:app --reload
Documentação interativa disponível em: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import re
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from feature_engineering import build_velocity_features
from seed_historico import carregar_historico

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configurações do modelo
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_antifraude.pkl")

# Thresholds calibrados a partir da curva Precision-Recall do treinamento.
THRESHOLD_ALTO  = 0.40   # score >= ALTO  → risco alto  (bloquear / acionar equipe)
THRESHOLD_MEDIO = 0.15   # score >= MEDIO → risco médio (revisar manualmente)
# score < MEDIO            → risco baixo  (aprovar)

# Número máximo de features retornadas em shap_values (ordenadas por |SHAP|)
SHAP_TOP_N = 15

# ---------------------------------------------------------------------------
# Estado global do modelo (preenchido no lifespan)
# ---------------------------------------------------------------------------

_model_state: dict[str, Any] = {}


def _carregar_modelo() -> None:
    """
    Carrega o modelo LightGBM serializado com joblib e extrai os metadados
    necessários para montagem correta do DataFrame de inferência:
    - lista ordenada de feature names
    - nomes e categorias das features categóricas (via booster serializado)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{MODEL_PATH}'. "
            "Execute train_antifraude.py primeiro."
        )

    model = joblib.load(MODEL_PATH)
    feature_names: list[str] = model.feature_name_

    booster = model.booster_
    model_str = booster.model_to_string()
    cat_idx_match = re.search(r"\[categorical_feature: ([^\]]+)\]", model_str)

    cat_feature_names: list[str] = []
    if cat_idx_match:
        cat_indices = [int(x) for x in cat_idx_match.group(1).split(",")]
        cat_feature_names = [feature_names[i] for i in cat_indices]

    pandas_categorical: list[list] = getattr(booster, "pandas_categorical", [])
    cat_map: dict[str, list] = dict(zip(cat_feature_names, pandas_categorical))

    # Inicializa o TreeExplainer SHAP uma única vez durante o startup.
    # model_output="raw" retorna contribuições na escala log-odds (padrão e mais
    # eficiente para LightGBM). Os sinais são diretamente interpretáveis:
    # valores positivos aumentam o risco de fraude, negativos reduzem.
    explainer = shap.TreeExplainer(model, model_output="raw")

    _model_state.update(
        model=model,
        feature_names=feature_names,
        cat_feature_names=cat_feature_names,
        cat_feature_set=set(cat_feature_names),
        cat_map=cat_map,
        shap_explainer=explainer,
    )


# ---------------------------------------------------------------------------
# Lifespan — carrega o modelo antes de receber requisições
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _carregar_modelo()
    _seed_transacoes_db()

    # Carga histórica opcional — activada por SEED_HISTORICO=1
    # Popula transacoes_db com uma amostra representativa do CSV real antes
    # de aceitar requisições.  Os escores são calculados logo abaixo via
    # _pontuar_todas_transacoes(), que re-processa todo o banco em memória.
    if os.getenv("SEED_HISTORICO", "0") == "1":
        global _proximo_id
        novo_id = carregar_historico(
            transacoes_db,
            id_offset=_proximo_id,
        )
        _proximo_id = novo_id
        _pontuar_todas_transacoes()

    yield


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sistema Antifraude Bancário",
    description=(
        "API para análise e gestão de transações suspeitas. "
        "O risco de fraude é calculado em tempo real pelo modelo LightGBM "
        "serializado em modelo_antifraude.pkl."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class TransacaoEntrada(BaseModel):
    """
    Payload para cadastro/análise de uma nova transação.

    Os campos abaixo mapeiam diretamente para features do modelo LightGBM.
    Campos ausentes ficam como NaN — o modelo trata valores faltantes
    internamente, portanto payloads parciais são aceitos.

    Campos obrigatórios para a engenharia de features de velocidade:
        client_id, card_id, timestamp, amount
    """

    # -- Identificadores (usados pela engenharia de features) -----------------
    client_id:  str   = Field(..., description="Identificador do cliente")
    card_id:    str   = Field(..., description="Identificador do cartão")
    timestamp:  str   = Field(..., description="Timestamp da transação (ISO 8601)")
    amount:     float = Field(..., gt=0, description="Valor em R$ (deve ser positivo)")

    # -- Campos descritivos ---------------------------------------------------
    local:      Optional[str] = Field(None, description="Local da transação")
    categoria:  Optional[str] = Field(None, description="Categoria da transação")

    # -- Features numéricas do modelo -----------------------------------------
    amount_fees:                  Optional[float] = Field(None)
    account_balance:              Optional[float] = Field(None)
    account_available_credit:     Optional[float] = Field(None)
    account_credit_limit:         Optional[float] = Field(None)
    mc_fraud_score:               Optional[float] = Field(None)
    visa_fraud_score:             Optional[float] = Field(None)
    acceptor_risk:                Optional[float] = Field(None)
    acceptor_latitude:            Optional[float] = Field(None)
    acceptor_longitude:           Optional[float] = Field(None)
    cp_card_prev_term_dist_1h:    Optional[float] = Field(None)
    cp_card_term_vel_1h:          Optional[float] = Field(None)
    cardid_count_1m:              Optional[float] = Field(None)
    mcc:                          Optional[float] = Field(None)
    card_days_to_expire:          Optional[float] = Field(None)
    card_credit_score:            Optional[float] = Field(None)

    # -- Features categóricas do modelo ---------------------------------------
    acceptor_region:      Optional[str] = Field(None)
    customer_region:      Optional[str] = Field(None)
    acceptor_country_code: Optional[str] = Field(None)
    card_brand:           Optional[str] = Field(None)
    card_type:            Optional[str] = Field(None)
    pos_environment:      Optional[str] = Field(None)
    auth_type:            Optional[str] = Field(None)
    currency_code:        Optional[str] = Field(None)
    transaction_category_code: Optional[str] = Field(None)
    event_type:           Optional[str] = Field(None)

    class Config:
        extra = "allow"   # campos adicionais são aceitos e repassados ao modelo


class Transacao(BaseModel):
    """Representa uma transação enriquecida com predição de risco."""

    id:           int
    client_id:    str
    card_id:      str
    valor:        float
    local:        Optional[str]
    categoria:    Optional[str]
    timestamp:    str
    escore_risco: float = Field(..., description="Probabilidade de fraude (0.0–1.0)")
    risco:        str   = Field(..., description="baixo | medio | alto")
    suspeita:     bool
    shap_values:  Optional[dict[str, float]] = Field(
        None,
        description=(
            "Top-15 contribuições SHAP (TreeExplainer, escala log-odds). "
            "Chave = nome da feature; valor positivo aumenta P(fraude), "
            "valor negativo reduz. Ordenado por |contribuição| decrescente."
        ),
    )


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

# Transações são persistidas como dicts com os campos do schema Transacao.
transacoes_db: list[dict] = []

# Log de ações executadas pelos analistas (auditoria)
acoes_executadas: list[dict] = []

# Contador auto-incremento para IDs
_proximo_id: int = 1

# ---------------------------------------------------------------------------
# Seed de transações — carregadas no lifespan após o modelo
# ---------------------------------------------------------------------------

# Cada entrada foi extraída do dataset de treino real e verificada para
# produzir a faixa de risco indicada pelo modelo LightGBM treinado.
# Campos de identificação (client_id, card_id, local, categoria, timestamp)
# foram substituídos por valores fictícios; todas as demais features
# são idênticas às linhas originais do dataset.
#
# Escores verificados (predict_proba):
#   IDs 1–3  → baixo  (< 0.15)
#   IDs 4–5  → médio  (0.15–0.40)
#   IDs 6–8  → alto   (> 0.40)
_TRANSACOES_SEED: list[dict] = [
    # ── 1 · baixo risco (score ≈ 0.035) ─────────────────────────────────────
    {
        "id": 1, "client_id": "CLI-001", "card_id": "CARD-001",
        "valor": 1.66, "local": "São Paulo, SP",
        "categoria": "alimentacao", "timestamp": "2024-07-15T12:10:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 1.66, "amount_fees": 0.04,
        "acceptor_country_code": "BR", "acceptor_region": "MG",
        "acceptor_risk": 11.0, "acceptor_risky_flag": 106.66,
        "acceptor_street_addr": "SYN_acceptor_street__08",
        "acceptor_url": "SYN_acceptor_url_08", "acceptor_visa_risk": 59.48,
        "account_available_credit": 3207.09, "account_balance": 937.29,
        "account_branch": 11.0, "account_core": 4.0, "account_currency": "BRL",
        "account_debt_payment_method": 21.0, "account_digital_signature": 8.0,
        "account_iban": 20.0, "account_number_of_cards": 17.0,
        "account_rewards_used_count": 0.0, "account_subsegment": 11.0,
        "add_wallet_onboarding": "SYN_add_wallet_onboa_08",
        "allow_contactless": True, "allow_crossborder": True, "allow_ecommerce": True,
        "avs_code": "SYN_avs_code_08",
        "business_app_transfer_purpose": "SYN_business_app_tra_08",
        "card_available_balance": 25.84, "card_brand": "BANDEIRA_A",
        "card_country_code": "BR", "card_credit_limit": 2877.65,
        "card_format": "SYN_card_format_08", "card_issuer": "SYN_card_issuer_08",
        "card_last4": 3953.0, "card_presence_flag": "Y",
        "card_program": "SYN_card_program_08",
        "card_transaction_amount_limit": 53.46, "card_type": "debito",
        "cardholder_authentication_veri": "SYN_cardholder_authe_08",
        "cardholder_bill_zip": "SYN_cardholder_bill__08",
        "cardholder_compress_bill_addr": "SYN_cardholder_compr_08",
        "cardholder_presence_indicator": "SYN_cardholder_prese_08",
        "cardholder_verification_rule": "SYN_cardholder_verif_08",
        "cardid_count_1m": 1.0, "cp_card_term_vel_1h": 1.0,
        "currency_code": "BRL", "customer_country_code": "BR",
        "customer_currency": "BRL", "customer_dob": "SYN_customer_dob_08",
        "customer_number_of_cards": 6.0, "customer_presumed_income": 1467.3,
        "customer_region": "MS", "customer_reported_income": 2401.12,
        "customer_risk_rating": "medio",
        "customer_segment": "SYN_customer_segment_08",
        "customer_street_addr": "SYN_customer_street__08",
        "cvv2_presence_indicator": "SYN_cvv2_presence_in_08",
        "cvv2_present_indicator": "SYN_cvv2_present_ind_08",
        "device_ip": "SYN_device_ip_08", "device_ip_country_code": "No",
        "digitalwallettoken_count_3m": 5.0, "event_type": "authorization",
        "f3ds_secure_protocol_version": "SYN_f3ds_secure_prot_08",
        "fdz_exposed_schema_diffb": "SYN_fdz_exposed_sche_08",
        "malware_flag": False, "malware_indicator": "SYN_malware_indicato_08",
        "mc_fraud_score": 226.0, "mcc": 4722.0,
        "mcc_cardbin_cntdist_cardexpdat": "SYN_mcc_cardbin_cntd_08",
        "modalidade_transacao": "cartao_presente",
        "original_amount_data": 17.08, "original_rec_amount_data": 44.57,
        "p2p_name": "SYN_p2p_name_08",
        "pos_cardholder_ver_cap": "SYN_pos_cardholder_v_08",
        "pos_environment": "POS", "pos_pin_len_cap": "SYN_pos_pin_len_cap_08",
        "pos_read_cap": "SYN_pos_read_cap_08",
        "sca_required": False, "secure_3d_val_status": "SYN_secure_3d_val_st_08",
        "service_code": "SYN_service_code_08",
        "terminal_ip": "SYN_terminal_ip_08", "terminal_ip_latitude": -18.2437,
        "terminal_ip_region": "MS", "tipo_transacao": "debito",
        "transaction_status": "approved", "visa_fraud_score": 127.0,
    },
    # ── 2 · baixo risco (score ≈ 0.043) ─────────────────────────────────────
    {
        "id": 2, "client_id": "CLI-002", "card_id": "CARD-002",
        "valor": 16.44, "local": "Rio de Janeiro, RJ",
        "categoria": "servicos", "timestamp": "2024-07-15T10:30:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 16.44, "amount_fees": 0.21,
        "acceptor_add_contact_info": "SYN_acceptor_add_con_30",
        "acceptor_country_code": "BR", "acceptor_region": "MG",
        "account_associated_cards": 12.0, "account_available_credit": 2464.69,
        "account_balance": 11614.57, "account_bic": 11.0,
        "account_currency": "BRL", "account_debt_balance": 40.27,
        "account_debt_payment_method": 10.0,
        "account_flag_adquirencia_activ": 12.0, "account_iban": 6.0,
        "account_last_payment_amount": 22.31,
        "account_trx_bank_acc_number": 5.0,
        "allow_contactless": True, "allow_crossborder": True, "allow_ecommerce": True,
        "allow_magstripe": False, "allow_wallet": False,
        "application_transaction_counte": 9.0,
        "card_available_balance": 66.47, "card_brand": "BANDEIRA_B",
        "card_country_code": "BR", "card_credit_limit": 1084.99,
        "card_credit_score": 20.83,
        "card_expected_payments_frequen": "SYN_card_expected_pa_30",
        "card_last4": 2219.0, "card_presence_flag": "Y", "card_seq_number": 11.0,
        "card_status": "SYN_card_status_30",
        "cardholder_authentication_veri": "SYN_cardholder_authe_30",
        "cardholder_bill_amount": 70.77,
        "cardholder_presence_indicator": "SYN_cardholder_prese_30",
        "cardholder_verification_rule": "SYN_cardholder_verif_30",
        "cardid_count_1m": 1.0, "cp_card_prev_term_dist_1h": 25.81,
        "cp_card_term_vel_1h": 1.0, "currency_code": "BRL",
        "customer_accounts": 12.0, "customer_country_code": "BR",
        "customer_country_code_change_f": 14.0, "customer_number_of_accounts": 5.0,
        "customer_presumed_income": 1779.24, "customer_region": "SP",
        "customer_risk_rating": "baixo", "customer_total_credit_limit": 60.6,
        "device_ip_country_code": "BR", "digitalwallettoken_count_3m": 4.0,
        "event_type": "authorization",
        "f3ds_secure_indicator": "SYN_f3ds_secure_indi_30",
        "fdz_exposed_schema_diffb": "SYN_fdz_exposed_sche_30",
        "lists_state": "SYN_lists_state_30",
        "malware_flag": False, "malware_indicator": "SYN_malware_indicato_30",
        "mc_fraud_score": 147.0, "mcc": 4111.0,
        "mcc_cardbin_cntdist_cardexpdat": "SYN_mcc_cardbin_cntd_30",
        "modalidade_transacao": "cartao_presente",
        "original_amount_data": 116.8, "original_rec_amount_data": 55.81,
        "pos_environment": "POS",
        "segment_channel_type": "SYN_segment_channel__30",
        "service_code": "SYN_service_code_30",
        "terminal_ip_country_code": 0.0, "terminal_ip_longitude": -64.1564,
        "terminal_ip_zip": "SYN_terminal_ip_zip_30", "terminal_longitude": -68.3929,
        "terminal_zip": "SYN_terminal_zip_30", "tipo_transacao": "debito",
        "transaction_status": "approved", "transstain": "SYN_transstain_30",
        "trusted_acceptor": False, "visa_fraud_score": 107.0,
    },
    # ── 3 · baixo risco (score ≈ 0.046) ─────────────────────────────────────
    {
        "id": 3, "client_id": "CLI-003", "card_id": "CARD-003",
        "valor": 57.73, "local": "Curitiba, PR",
        "categoria": "saude", "timestamp": "2024-07-15T09:05:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 57.73, "amount_fees": 1.26,
        "acceptor_country_code": "BR", "acceptor_latitude": -24.9722,
        "acceptor_region": "PE", "acceptor_risk": 20.0,
        "acceptor_street_addr": "SYN_acceptor_street__05",
        "acceptor_zip": "SYN_acceptor_zip_05",
        "account_associated_cards": 15.0, "account_available_credit": 2972.27,
        "account_balance": 7123.09, "account_bic": 3.0,
        "account_credit_limit": 47.85, "account_currency": "BRL",
        "account_debt_balance": 31.64, "account_digital_signature": 1.0,
        "account_iban": 1.0, "account_number_of_cards": 14.0,
        "account_open_mode": 2.0, "account_pw_change_flag": 21.0,
        "account_rewards_used_count": 10.0, "account_trx_bank_acc_number": 22.0,
        "allow_contactless": False, "allow_crossborder": True,
        "card_brand": "BANDEIRA_B", "card_country_code": "BR",
        "card_credit_limit": 1954.77, "card_last4": 4824.0,
        "card_presence_flag": "N", "card_program": "SYN_card_program_05",
        "card_status": "SYN_card_status_05",
        "cardholder_authentication_veri": "SYN_cardholder_authe_05",
        "cardholder_bill_amount": 108.77,
        "cardholder_verification_rule": "SYN_cardholder_verif_05",
        "cardid_count_1m": 1.0, "cp_card_prev_term_dist_1h": 22.74,
        "cp_card_term_vel_1h": 1.0, "currency_code": "BRL",
        "customer_country_code": "BR",
        "customer_occupation_status": "SYN_customer_occupat_05",
        "customer_presumed_income": 3938.08, "customer_region": "SP",
        "customer_reported_income": 1365.08, "customer_risk_rating": "medio",
        "customer_segment": "SYN_customer_segment_05",
        "customer_zip": "SYN_customer_zip_05",
        "device_ip_country_code": "BR", "device_ip_longitude": -47.6143,
        "device_language": "SYN_device_language_05",
        "digitalwallettoken_count_3m": 6.0, "event_type": "authorization",
        "f3ds_secure_protocol_version": "SYN_f3ds_secure_prot_05",
        "fdz_partition_index": "SYN_fdz_partition_in_05",
        "malware_flag": True, "mc_fraud_score": 195.0, "mcc": 5541.0,
        "modalidade_transacao": "cartao_presente",
        "original_rec_amount_data": 6.21, "p2p_account": 1.0,
        "p2p_source_funds": "SYN_p2p_source_funds_05",
        "p2p_state": "SYN_p2p_state_05",
        "pos_cardholder_disp_data_len": "SYN_pos_cardholder_d_05",
        "pos_cardholder_ver_cap": "SYN_pos_cardholder_v_05",
        "pos_environment": "POS",
        "pos_icc_data_len": "SYN_pos_icc_data_len_05",
        "pos_pin_len_cap": "SYN_pos_pin_len_cap_05", "proxy_flag": False,
        "sca_required": False, "secure_3d_indicator": "SYN_secure_3d_indica_05",
        "tipo_transacao": "debito", "transaction_status": "approved",
        "tvr": "SYN_tvr_05", "visa_fraud_score": 0.0,
    },
    # ── 4 · médio risco (score ≈ 0.200) ─────────────────────────────────────
    {
        "id": 4, "client_id": "CLI-001", "card_id": "CARD-004",
        "valor": 81.91, "local": "Salvador, BA",
        "categoria": "eletronicos", "timestamp": "2024-07-14T23:45:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 81.91, "amount_fees": 0.92,
        "acceptor_country_code": "BR", "acceptor_latitude": -22.5277,
        "acceptor_mastercard_risk": 1.36, "acceptor_region": "BA",
        "acceptor_risk": 41.0, "acceptor_risky_flag": 36.2,
        "account_associated_cards": 14.0, "account_balance": 5975.4,
        "account_core": 19.0, "account_currency": "BRL",
        "account_flag_adquirencia_activ": 24.0, "account_iban": 24.0,
        "account_open_mode": 7.0,
        "allow_contactless": True, "allow_crossborder": False,
        "allow_ecommerce": True, "allow_withdraw": False,
        "application_transaction_counte": 19.0,
        "card_brand": "BANDEIRA_B", "card_country_code": "BR",
        "card_credit_limit": 33642.64, "card_last4": 7740.0,
        "card_presence_flag": "N", "card_type": "debito",
        "cardholder_bill_conv_rate": 41.89, "cardid_count_1m": 1.0,
        "cp_card_prev_term_dist_1h": 14.9, "cp_card_term_vel_1h": 2.0,
        "currency_code": "BRL", "customer_country_code_change_f": 19.0,
        "customer_currency": "BRL",
        "customer_occupation_status": "SYN_customer_occupat_14",
        "customer_presumed_income": 7197.55, "customer_region": "RS",
        "customer_reported_income": 2839.56, "customer_risk_rating": "baixo",
        "customer_total_credit_limit": 30.29,
        "cvv_val_code": "SYN_cvv_val_code_14",
        "device_ip": "SYN_device_ip_14", "device_ip_country_code": "BR",
        "digitalwallettoken_count_3m": 11.0,
        "info_type": "SYN_info_type_14",
        "mc_fraud_score": 206.0, "mc_merchant_fraud_code": False, "mcc": 5411.0,
        "modalidade_transacao": "cartao_nao_presente",
        "original_amount_data": 20.91,
        "p2p_address": "SYN_p2p_address_14",
        "p2p_source_funds": "SYN_p2p_source_funds_14",
        "pos_cardholder_rec_data_len": "SYN_pos_cardholder_r_14",
        "pos_environment": "ECOMMERCE", "terminal_ip_longitude": -69.4012,
        "tipo_transacao": "debito", "transaction_status": "approved",
        "visa_fraud_score": 455.0,
    },
    # ── 5 · médio risco (score ≈ 0.152) ─────────────────────────────────────
    {
        "id": 5, "client_id": "CLI-002", "card_id": "CARD-005",
        "valor": 15.16, "local": "Porto Alegre, RS",
        "categoria": "transferencia", "timestamp": "2024-07-14T22:18:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 15.16, "amount_fees": 0.35,
        "acceptor_country_code": "BR", "acceptor_longitude": -56.2869,
        "acceptor_region": "SC", "acceptor_risk": 16.0,
        "acceptor_street_addr": "SYN_acceptor_street__13",
        "account_available_credit": 1580.43, "account_balance": -1256.06,
        "account_bank_branch_id": 3.0, "account_currency": "BRL",
        "account_digital_signature": 10.0, "account_last_payment_amount": 32.16,
        "account_limits": 20.21, "account_ownership_type": 3.0,
        "account_pw_change_flag": 17.0, "account_rewards_used_count": 13.0,
        "account_segment": 16.0, "account_status": 23.0, "account_type": 8.0,
        "acquirer_seq_number": 22.0,
        "allow_contactless": True, "allow_crossborder": True, "allow_ecommerce": True,
        "avs_code": "SYN_avs_code_13", "card_brand": "BANDEIRA_A",
        "card_category": "SYN_card_category_13", "card_country_code": "BR",
        "card_days_to_expire": 22.0,
        "card_expected_payments_frequen": "SYN_card_expected_pa_13",
        "card_last4": 2206.0, "card_presence_flag": "N",
        "card_transaction_amount_limit": 14.83, "card_type": "debito",
        "card_ver_data": "SYN_card_ver_data_13",
        "cardholder_authentication_veri": "SYN_cardholder_authe_13",
        "cardholder_bill_currency_code": "BRL",
        "cardholder_bill_street_addr": "SYN_cardholder_bill__13",
        "cardholder_verification_rule": "SYN_cardholder_verif_13",
        "cardid_count_1m": 1.0, "cp_card_prev_term_dist_1h": 16.77,
        "currency_code": "BRL", "customer_country_code": "BR",
        "customer_currency": "BRL",
        "customer_kyc_lvl": "SYN_customer_kyc_lvl_13",
        "customer_presumed_income": 6937.84, "customer_region": "SP",
        "customer_reported_income": 3825.05, "customer_risk_rating": "baixo",
        "device_ip_country_code": "BR", "device_ip_latitude": -16.8485,
        "digitalwallettoken_count_3m": 11.0, "event_type": "authorization",
        "f3ds_secure_indicator": "SYN_f3ds_secure_indi_13",
        "fdz_partition_index": "SYN_fdz_partition_in_13",
        "lifecycle_seq_number": 21.0,
        "malware_indicator": "SYN_malware_indicato_13",
        "mc_fraud_score": 302.0, "mcc": 4111.0,
        "modalidade_transacao": "cartao_presente",
        "pos_acceptor_rec_data_len": "SYN_pos_acceptor_rec_13",
        "pos_approval_code_len": "SYN_pos_approval_cod_13",
        "pos_environment": "POS", "pos_read_cap": "SYN_pos_read_cap_13",
        "service_code": "SYN_service_code_13",
        "terminal_ip_latitude": -11.1524, "terminal_ip_region": "SP",
        "terminal_ip_zip": "SYN_terminal_ip_zip_13", "terminal_longitude": -50.1285,
        "terminal_timezone": "SYN_terminal_timezon_13",
        "tipo_transacao": "debito",
        "transaction_category_code": "SYN_transaction_cate_13",
        "transaction_status": "approved", "visa_fraud_score": 64.0,
    },
    # ── 6 · alto risco (score ≈ 0.868) ──────────────────────────────────────
    {
        "id": 6, "client_id": "CLI-003", "card_id": "CARD-006",
        "valor": 202.91, "local": "Belo Horizonte, MG",
        "categoria": "compra_internacional", "timestamp": "2024-07-15T02:34:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 202.91, "amount_fees": 1.74,
        "acceptor_country_code": "BR", "acceptor_longitude": -44.5785,
        "acceptor_mastercard_risk": 44.9, "acceptor_region": "BA",
        "acceptor_risk": 23.0, "acceptor_risky_flag": 12.93,
        "acceptor_visa_risk": 9.36, "account_associated_cards": 17.0,
        "account_balance": 6330.1, "account_bic": 12.0, "account_core": 12.0,
        "account_open_mode": 0.0, "account_otb_limit": 117.93,
        "account_rewards_used_count": 9.0, "account_segment": 23.0,
        "account_trx_bank_acc_number": 3.0, "acquirer_seq_number": 4.0,
        "allow_contactless": False, "allow_crossborder": False,
        "allow_ecommerce": True, "allow_withdraw": True,
        "application_transaction_counte": 7.0,
        "card_brand": "BANDEIRA_C", "card_country_code": "BR",
        "card_credit_limit": 5474.11, "card_credit_score": 38.01,
        "card_expected_payments_frequen": "SYN_card_expected_pa_08",
        "card_format": "SYN_card_format_08", "card_last4": 5768.0,
        "card_presence_flag": "Y",
        "card_transaction_amount_limit": 12.3, "card_type": "debito",
        "cardholder_presence_indicator": "SYN_cardholder_prese_08",
        "cardid_count_1m": 1.0, "cp_card_prev_term_dist_1h": 0.56,
        "cp_card_term_vel_1h": 3.0, "currency_code": "BRL",
        "customer_accounts": 6.0, "customer_cards": "SYN_customer_cards_08",
        "customer_country_code": "BR", "customer_dob": "SYN_customer_dob_08",
        "customer_gender": "SYN_customer_gender_08",
        "customer_kyc_lvl": "SYN_customer_kyc_lvl_08",
        "customer_occupation_status": "SYN_customer_occupat_08",
        "customer_onboarding_score": 81.12, "customer_presumed_income": 18360.24,
        "customer_region": "SP", "customer_reported_income": 3267.6,
        "customer_risk_rating": "medio",
        "customer_street_addr": "SYN_customer_street__08",
        "cvv_val_code": "SYN_cvv_val_code_08",
        "device_ip_country_code": "BR", "device_ip_latitude": -8.5419,
        "device_language": "SYN_device_language_08",
        "digitalwallettoken_count_3m": 7.0,
        "f3ds_secure_protocol_version": "SYN_f3ds_secure_prot_08",
        "lifecycle_indicator": "SYN_lifecycle_indica_08",
        "mc_fraud_score": 227.0, "mcc": 5732.0,
        "modalidade_transacao": "cartao_presente",
        "p2p_account": 4.0, "p2p_country": 12.0,
        "pos_approval_code_len": "SYN_pos_approval_cod_08",
        "pos_card_capture_cap": "SYN_pos_card_capture_08",
        "pos_environment": "POS", "pos_pin_len_cap": "SYN_pos_pin_len_cap_08",
        "terminal_ip": "SYN_terminal_ip_08", "terminal_latitude": -21.0665,
        "terminal_longitude": -43.3944,
        "terminal_street_addr": "SYN_terminal_street__08",
        "tipo_transacao": "debito",
        "transaction_category_code": "SYN_transaction_cate_08",
        "transaction_status": "approved", "transstain": "SYN_transstain_08",
        "visa_fraud_score": 125.0,
    },
    # ── 7 · alto risco (score ≈ 0.869) ──────────────────────────────────────
    {
        "id": 7, "client_id": "CLI-001", "card_id": "CARD-001",
        "valor": 70.28, "local": "Campo Grande, MS",
        "categoria": "saque", "timestamp": "2024-07-15T03:05:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 70.28, "amount_fees": 1.51,
        "acceptor_add_contact_info": "SYN_acceptor_add_con_26",
        "acceptor_country_code": "BR", "acceptor_region": "MT",
        "acceptor_risk": 10.0, "acceptor_risky_flag": 35.09,
        "acceptor_street_addr": "SYN_acceptor_street__26",
        "acceptor_url": "SYN_acceptor_url_26",
        "account_available_credit": 1814.1, "account_balance": 8509.5,
        "account_currency": "BRL", "account_debt_payment_method": 15.0,
        "account_flag_adquirencia_activ": 9.0,
        "account_last_payment_amount": 22.59, "account_ownership_type": 23.0,
        "account_pw_change_flag": 14.0, "account_segment": 21.0,
        "account_status": 23.0, "account_subsegment": 11.0, "account_type": 24.0,
        "allow_contactless": True, "allow_crossborder": True, "allow_ecommerce": True,
        "allow_wallet": False,
        "card_available_balance": 11.32, "card_brand": "BANDEIRA_A",
        "card_country_code": "BR", "card_credit_limit": 9116.15,
        "card_last4": 9993.0, "card_presence_flag": "Y", "card_type": "debito",
        "cardholder_bill_zip": "SYN_cardholder_bill__26",
        "cardholder_presence_indicator": "SYN_cardholder_prese_26",
        "cardid_count_1m": 1.0, "cp_card_prev_term_dist_1h": 3.51,
        "cp_card_term_vel_1h": 1.0, "currency_code": "BRL",
        "currency_unit": "BRL", "customer_cards": "SYN_customer_cards_26",
        "customer_country_code": "BR", "customer_currency": "BRL",
        "customer_gender": "SYN_customer_gender_26",
        "customer_number_of_accounts": 20.0,
        "customer_occupation_status": "SYN_customer_occupat_26",
        "customer_onboarding_score": 26.84, "customer_presumed_income": 19757.65,
        "customer_region": "SC", "customer_reported_income": 1200.0,
        "customer_risk_rating": "baixo",
        "cvv_val_code": "SYN_cvv_val_code_26",
        "device_ip_country_code": "BR", "digitalwallettoken_count_3m": 7.0,
        "event_type": "authorization",
        "fdz_exposed_schema_diffb": "SYN_fdz_exposed_sche_26",
        "malware_indicator": "SYN_malware_indicato_26",
        "mc_fraud_score": 373.0, "mc_merchant_fraud_code": False, "mcc": 5999.0,
        "modalidade_transacao": "cartao_presente",
        "p2p_name": "SYN_p2p_name_26",
        "pos_cardholder_disp_data_len": "SYN_pos_cardholder_d_26",
        "pos_environment": "POS",
        "pos_icc_data_len": "SYN_pos_icc_data_len_26",
        "pos_read_cap": "SYN_pos_read_cap_26",
        "secure_3d_val_code": "SYN_secure_3d_val_co_26",
        "segment_channel_type": "SYN_segment_channel__26",
        "tipo_transacao": "debito", "transaction_status": "approved",
        "transstain": "SYN_transstain_26", "trusted_acceptor": False,
        "tvr": "SYN_tvr_26", "visa_fraud_score": 215.0,
    },
    # ── 8 · alto risco (score ≈ 0.864) ──────────────────────────────────────
    {
        "id": 8, "client_id": "CLI-002", "card_id": "CARD-002",
        "valor": 52.75, "local": "Recife, PE",
        "categoria": "compra_internacional", "timestamp": "2024-07-13T04:17:00",
        "escore_risco": 0.0, "risco": "baixo", "suspeita": False,
        "amount": 52.75, "amount_fees": 1.12,
        "acceptor_country_code": "BR", "acceptor_region": "PE",
        "acceptor_risk": 25.0, "acceptor_risky_flag": 73.38,
        "acceptor_url": "SYN_acceptor_url_27",
        "account_balance": 13900.13, "account_core": 19.0,
        "account_currency": "BRL", "account_flag_adquirencia_activ": 20.0,
        "account_last_payment_amount": 40.41, "account_subsegment": 20.0,
        "acquirer_seq_number": 20.0,
        "allow_contactless": True, "allow_crossborder": True,
        "atc2": "SYN_atc2_27", "avs_code": "SYN_avs_code_27",
        "card_brand": "BANDEIRA_B", "card_country_code": "BR",
        "card_credit_limit": 1114.84, "card_credit_score": 305.66,
        "card_last4": 9114.0, "card_presence_flag": "N",
        "card_program": "SYN_card_program_27", "card_seq_number": 8.0,
        "card_status": "SYN_card_status_27", "card_type": "credito",
        "cardid_count_1m": 1.0, "cp_card_prev_term_dist_1h": 11.36,
        "cp_card_term_vel_1h": 2.0, "currency_code": "BRL",
        "customer_cards": "SYN_customer_cards_27", "customer_country_code": "BR",
        "customer_country_code_change_f": 23.0, "customer_currency": "BRL",
        "customer_occupation_status": "SYN_customer_occupat_27",
        "customer_onboarding_score": 38.77, "customer_presumed_income": 4187.68,
        "customer_region": "BA", "customer_reported_income": 8161.98,
        "customer_risk_rating": "baixo",
        "customer_street_addr": "SYN_customer_street__27",
        "cvv2_present_indicator": "SYN_cvv2_present_ind_27",
        "device_ip": "SYN_device_ip_27", "device_ip_country_code": "BR",
        "device_language": "SYN_device_language_27",
        "digitalwallettoken_count_3m": 8.0, "event_type": "clearing",
        "lifecycle_indicator": "SYN_lifecycle_indica_27",
        "mc_fraud_score": 360.0, "mcc": 4111.0,
        "mcc_cardbin_cntdist_cardexpdat": "SYN_mcc_cardbin_cntd_27",
        "modalidade_transacao": "cartao_nao_presente",
        "p2p_account": 12.0, "p2p_country": 15.0,
        "p2p_source_funds": "SYN_p2p_source_funds_27",
        "pos_card_capture_cap": "SYN_pos_card_capture_27",
        "pos_cardholder_rec_data_len": "SYN_pos_cardholder_r_27",
        "pos_environment": "ECOMMERCE", "pos_read_cap": "SYN_pos_read_cap_27",
        "proxy_flag": False,
        "segment_channel_type": "SYN_segment_channel__27",
        "terminal_ip_region": "BA", "terminal_longitude": -35.4774,
        "tipo_transacao": "credito", "transaction_status": "approved",
        "transstain": "SYN_transstain_27", "visa_fraud_score": 174.0,
    },
]


def _seed_transacoes_db() -> None:
    """
    Preenche transacoes_db com as transações seed e ajusta o contador de IDs,
    depois executa o pipeline completo para que cada entrada já contenha
    escore_risco, risco, suspeita e as features de velocidade reais.

    Deve ser chamada no lifespan, após _carregar_modelo().
    """
    global _proximo_id
    transacoes_db.extend(_TRANSACOES_SEED)
    _proximo_id = max(t["id"] for t in _TRANSACOES_SEED) + 1
    _pontuar_todas_transacoes()

# ---------------------------------------------------------------------------
# Funções auxiliares — inferência
# ---------------------------------------------------------------------------

def _build_dataframe_para_modelo(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Constrói um DataFrame com exatamente as features esperadas pelo modelo.
    Campos ausentes no payload ficam como NaN.
    - Colunas numéricas → float64
    - Colunas categóricas → dtype 'category' com vocabulário do treinamento
    """
    feature_names: list[str]   = _model_state["feature_names"]
    cat_feature_names: list[str] = _model_state["cat_feature_names"]
    cat_feature_set: set[str]  = _model_state["cat_feature_set"]
    cat_map: dict[str, list]   = _model_state["cat_map"]

    row: dict[str, Any] = {feat: payload.get(feat) for feat in feature_names}
    df = pd.DataFrame([row], columns=feature_names)

    for col in cat_feature_names:
        cats = cat_map.get(col)
        if cats is not None:
            df[col] = pd.Categorical(df[col], categories=cats)
        else:
            df[col] = df[col].astype("category")

    num_cols = [c for c in feature_names if c not in cat_feature_set]
    df[num_cols] = df[num_cols].astype("float64")

    return df


def _pipeline_engenharia_features(
    transacoes: list[dict],
) -> dict[int, dict[str, Any]]:
    """
    Aplica build_velocity_features sobre todas as transações do banco e
    retorna um mapa {id → dict_com_features_de_velocidade}.

    As features de velocidade (card_txn_count_10min, card_txn_count_1h,
    seconds_since_last_txn, high_value_flag) são calculadas de forma
    histórica sobre o conjunto completo de transações.
    """
    if not transacoes:
        return {}

    df = pd.DataFrame(transacoes)[["id", "client_id", "card_id", "timestamp", "valor"]].copy()
    df = df.rename(columns={"valor": "amount"})

    # build_velocity_features exige merchant_id; criamos coluna placeholder
    df["merchant_id"] = "UNKNOWN"

    df_feat = build_velocity_features(
        df,
        client_col="client_id",
        card_col="card_id",
        ts_col="timestamp",
        amount_col="amount",
        merchant_col="merchant_id",
    )

    vel_cols = ["card_txn_count_10min", "card_txn_count_1h",
                "seconds_since_last_txn", "high_value_flag"]
    result = df_feat.set_index("id")[vel_cols].copy()

    # _rolling_count pode retornar NaN nas primeiras linhas de cada grupo
    # (janela vazia). Preenchemos com 0 para evitar erros de cast no modelo.
    result[["card_txn_count_10min", "card_txn_count_1h", "high_value_flag"]] = (
        result[["card_txn_count_10min", "card_txn_count_1h", "high_value_flag"]]
        .fillna(0)
    )

    return result.to_dict(orient="index")


def _classificar_risco(score: float) -> tuple[str, bool]:
    """Retorna (nível_risco, suspeita) a partir do escore do modelo."""
    if score >= THRESHOLD_ALTO:
        return "alto", True
    if score >= THRESHOLD_MEDIO:
        return "medio", True
    return "baixo", False


def _calcular_shap(df: pd.DataFrame) -> dict[str, float]:
    """
    Calcula os valores SHAP para uma única transação usando o TreeExplainer
    inicializado no startup.

    Retorna um dicionário {feature_name: shap_value} com as SHAP_TOP_N features
    de maior magnitude absoluta, ordenadas de forma decrescente por |shap_value|.
    Os valores estão na escala log-odds: positivos aumentam P(fraude), negativos
    reduzem.
    """
    explainer = _model_state["shap_explainer"]
    feature_names: list[str] = _model_state["feature_names"]

    # shap_values retorna shape (n_samples, n_features) para TreeExplainer com
    # model_output="probability". Usamos [0] para a linha única enviada.
    raw = explainer.shap_values(df)

    # Alguns builds retornam lista [class0, class1]; extraímos a classe positiva.
    if isinstance(raw, list):
        shap_row = raw[1][0]
    else:
        shap_row = raw[0]

    # Monta dicionário e ordena pelas contribuições de maior magnitude
    shap_dict = dict(zip(feature_names, (float(v) for v in shap_row)))
    top = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {k: round(v, 8) for k, v in top[:SHAP_TOP_N]}


def _pontuar_transacao(
    entrada: dict[str, Any],
    features_velocidade: dict[str, Any],
) -> tuple[float, str, bool, dict[str, float]]:
    """
    Executa o pipeline completo para uma transação:
      1. Mescla campos do payload com features de velocidade calculadas
      2. Monta o DataFrame com as features do modelo
      3. Chama predict_proba do LightGBM
      4. Calcula valores SHAP via TreeExplainer
      5. Classifica o risco

    Retorna (escore_risco, nivel_risco, suspeita, shap_values).
    """
    # Mescla payload original com features de velocidade
    payload = {**entrada, **features_velocidade}

    # O modelo usa 'amount' como nome canônico; 'valor' é o nome interno da API
    if "valor" in payload and "amount" not in payload:
        payload["amount"] = payload["valor"]

    df = _build_dataframe_para_modelo(payload)
    score = float(_model_state["model"].predict_proba(df)[0, 1])
    nivel, suspeita = _classificar_risco(score)
    shap_vals = _calcular_shap(df)
    return round(score, 6), nivel, suspeita, shap_vals


def _pontuar_todas_transacoes() -> None:
    """
    Re-executa o pipeline de engenharia de features sobre todas as transações
    em memória e atualiza escore_risco, risco, suspeita e shap_values de cada
    entrada.

    Chamado ao inserir novas transações para manter as features de velocidade
    (que dependem do histórico completo) sempre atualizadas.
    """
    features_por_id = _pipeline_engenharia_features(transacoes_db)

    for txn in transacoes_db:
        vel = features_por_id.get(txn["id"], {})
        score, nivel, suspeita, shap_vals = _pontuar_transacao(txn, vel)
        txn["escore_risco"] = score
        txn["risco"]        = nivel
        txn["suspeita"]     = suspeita
        txn["shap_values"]  = shap_vals


# ---------------------------------------------------------------------------
# Funções auxiliares — operações no banco
# ---------------------------------------------------------------------------

def _proximo_id_auto() -> int:
    global _proximo_id
    novo = _proximo_id
    _proximo_id += 1
    return novo


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
    cliente = transacao["client_id"]
    tid     = transacao["id"]

    if tipo_acao == "bloquear_compra":
        transacao["suspeita"] = False
        return f"Compra #{tid} do cliente {cliente} bloqueada com sucesso."

    if tipo_acao == "bloquear_cartao":
        transacao["suspeita"] = False
        return (
            f"Cartão do cliente {cliente} bloqueado. "
            f"Transação #{tid} encerrada e conta sinalizada para revisão."
        )

    if tipo_acao == "enviar_mensagem":
        return (
            f"Mensagem de alerta enviada ao cliente {cliente} "
            f"referente à transação #{tid}. Aguardando confirmação."
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/transacoes",
    response_model=Transacao,
    summary="Registra uma nova transação e aplica o modelo de risco",
    tags=["Transações"],
    status_code=201,
)
def criar_transacao(entrada: TransacaoEntrada) -> Transacao:
    """
    Registra uma nova transação, aplica o pipeline de engenharia de features
    de velocidade (build_velocity_features) sobre todo o histórico e calcula
    o escore de fraude via predict_proba do LightGBM.

    Além do escore, o campo `shap_values` é preenchido com as 15 features de
    maior contribuição (SHAP TreeExplainer) na escala de probabilidade.

    Campos `escore_risco`, `risco`, `suspeita` e `shap_values` são derivados
    do modelo — não são aceitos no payload de entrada.
    """
    global _proximo_id

    novo_id = _proximo_id_auto()
    txn = {
        "id":           novo_id,
        "client_id":    entrada.client_id,
        "card_id":      entrada.card_id,
        "valor":        entrada.amount,
        "local":        entrada.local,
        "categoria":    entrada.categoria,
        "timestamp":    entrada.timestamp,
        "escore_risco": 0.0,
        "risco":        "baixo",
        "suspeita":     False,
        # Campos extras do payload são preservados para uso pelo modelo
        **{
            k: v
            for k, v in entrada.model_dump().items()
            if k not in {"client_id", "card_id", "amount", "local",
                         "categoria", "timestamp"}
            and v is not None
        },
    }
    transacoes_db.append(txn)

    # Re-pontua todas as transações para atualizar features de velocidade
    _pontuar_todas_transacoes()

    return transacoes_db[-1]


@app.get(
    "/transacoes-suspeitas",
    response_model=list[Transacao],
    summary="Lista todas as transações suspeitas",
    tags=["Transações"],
)
def listar_transacoes_suspeitas():
    """
    Retorna todas as transações cujo escore de risco calculado pelo modelo
    LightGBM resultou em classificação `medio` ou `alto` (suspeita = true).
    """
    suspeitas = [t for t in transacoes_db if t.get("suspeita")]

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
    Retorna todas as transações (suspeitas ou não) associadas ao
    `cliente_id` informado, com os escores de risco do modelo.
    """
    historico = [t for t in transacoes_db if t["client_id"] == cliente_id]

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
    resultado = simular_acao(acao.tipo_acao, transacao)

    registro = {
        "transacao_id":  id,
        "client_id":     transacao["client_id"],
        "tipo_acao":     acao.tipo_acao,
        "justificativa": acao.justificativa,
        "resultado":     resultado,
        "executado_em":  datetime.now().isoformat(),
    }
    acoes_executadas.append(registro)

    return {
        "status":        "sucesso",
        "acao_executada": acao.tipo_acao,
        "resultado":     resultado,
        "auditoria":     registro,
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


@app.get(
    "/",
    summary="Status da API e informações do modelo carregado",
    tags=["Status"],
)
def root() -> dict:
    """Retorna o status da API e metadados do modelo LightGBM em uso."""
    return {
        "status":           "online",
        "modelo":           os.path.basename(MODEL_PATH),
        "features":         len(_model_state.get("feature_names", [])),
        "threshold_medio":  THRESHOLD_MEDIO,
        "threshold_alto":   THRESHOLD_ALTO,
        "transacoes_em_memoria": len(transacoes_db),
    }
