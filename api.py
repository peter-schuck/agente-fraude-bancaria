"""
api.py
======
FastAPI para inferencia em tempo real do modelo antifraude LightGBM.

Endpoint principal:
    POST /analisar-transacao
        Recebe os dados da transacao, calcula o escore de risco e retorna
        nivel de risco (Baixo / Medio / Alto) e recomendacao do agente.

Outros endpoints utilitarios:
    GET  /               -> status da API e modelo carregado
    GET  /features       -> lista de features esperadas pelo modelo
    GET  /health         -> health-check para monitoramento

Uso:
    pip install fastapi uvicorn lightgbm joblib pandas
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_antifraude.pkl")

# Thresholds definidos a partir da curva P-R gerada no treinamento.
# Ajuste conforme a tolerancia a falsos positivos/negativos do negocio.
THRESHOLD_ALTO  = 0.40   # score >= ALTO  -> risco Alto  (bloquear / acionar equipe)
THRESHOLD_MEDIO = 0.15   # score >= MEDIO -> risco Medio (revisar manualmente)
# score < MEDIO           -> risco Baixo  (aprovar)

# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo nao encontrado em '{MODEL_PATH}'. "
        "Execute train_antifraude.py primeiro."
    )

import re as _re

_model = joblib.load(MODEL_PATH)
_feature_names: list[str] = _model.feature_name_

# Extrai indices das features categoricas a partir do modelo serializado
_booster         = _model.booster_
_model_str       = _booster.model_to_string()
_cat_idx_match   = _re.search(r"\[categorical_feature: ([^\]]+)\]", _model_str)
_cat_feature_set: set[str] = set()
_cat_feature_names: list[str] = []   # ordem exata do treinamento
if _cat_idx_match:
    _cat_indices = [int(x) for x in _cat_idx_match.group(1).split(",")]
    _cat_feature_names = [_feature_names[i] for i in _cat_indices]
    _cat_feature_set   = set(_cat_feature_names)

# pandas_categorical: lista de arrays de categorias, uma por feature cat (mesma ordem)
_pandas_categorical: list[list] = getattr(_booster, "pandas_categorical", [])
# Mapa feature_name -> lista de categorias do treino
_cat_map: dict[str, list] = dict(zip(_cat_feature_names, _pandas_categorical))

# ---------------------------------------------------------------------------
# Schema Pydantic — todos os campos sao opcionais para maxima flexibilidade
# ---------------------------------------------------------------------------

class TransacaoInput(BaseModel):
    """
    Dados de uma transacao de cartao para analise de risco.

    Todos os campos sao opcionais: campos ausentes recebem None e o modelo
    LightGBM trata internamente como valor faltante (NaN), comportamento
    correto para arvores de decisao.

    Os campos abaixo refletem as features usadas no treinamento.
    Envie apenas os campos disponiveis — o endpoint aceita payloads parciais.
    """

    # -- Valores e saldo ------------------------------------------------------
    amount:                       Optional[float] = Field(None, description="Valor da transacao")
    amount_fees:                  Optional[float] = Field(None, description="Taxas da transacao")
    account_balance:              Optional[float] = Field(None, description="Saldo atual da conta")
    account_available_credit:     Optional[float] = Field(None, description="Credito disponivel")
    account_credit_limit:         Optional[float] = Field(None, description="Limite de credito da conta")
    account_cash_advance_balance: Optional[float] = Field(None, description="Saldo de saque")
    account_debt_balance:         Optional[float] = Field(None, description="Saldo devedor")
    account_last_payment_amount:  Optional[float] = Field(None, description="Ultimo pagamento")
    card_credit_limit:            Optional[float] = Field(None, description="Limite do cartao")
    card_available_balance:       Optional[float] = Field(None, description="Saldo disponivel no cartao")
    card_credit_score:            Optional[float] = Field(None, description="Score de credito do cartao")
    card_transaction_amount_limit:Optional[float] = Field(None, description="Limite por transacao")
    cardholder_bill_amount:       Optional[float] = Field(None, description="Valor de fatura do portador")
    cardholder_bill_conv_rate:    Optional[float] = Field(None, description="Taxa de conversao da fatura")
    customer_total_balance:       Optional[float] = Field(None, description="Saldo total do cliente")
    customer_total_credit_limit:  Optional[float] = Field(None, description="Limite total do cliente")
    customer_presumed_income:     Optional[float] = Field(None, description="Renda presumida do cliente")
    customer_reported_income:     Optional[float] = Field(None, description="Renda declarada do cliente")
    customer_onboarding_score:    Optional[float] = Field(None, description="Score de onboarding")
    original_amount_data:         Optional[float] = Field(None, description="Valor original da transacao")
    original_rec_amount_data:     Optional[float] = Field(None, description="Valor original de recorrencia")

    # -- Scores de risco externos ---------------------------------------------
    mc_fraud_score:               Optional[float] = Field(None, description="Score de fraude Mastercard")
    visa_fraud_score:             Optional[float] = Field(None, description="Score de fraude Visa")
    acceptor_risk:                Optional[float] = Field(None, description="Risco do estabelecimento")
    acceptor_mastercard_risk:     Optional[float] = Field(None, description="Risco Mastercard do estabelecimento")
    acceptor_visa_risk:           Optional[float] = Field(None, description="Risco Visa do estabelecimento")
    acceptor_risky_flag:          Optional[float] = Field(None, description="Flag de risco do estabelecimento")

    # -- Localizacao e geolocalizacao -----------------------------------------
    acceptor_latitude:            Optional[float] = Field(None, description="Latitude do estabelecimento")
    acceptor_longitude:           Optional[float] = Field(None, description="Longitude do estabelecimento")
    terminal_latitude:            Optional[float] = Field(None, description="Latitude do terminal")
    terminal_longitude:           Optional[float] = Field(None, description="Longitude do terminal")
    terminal_ip_latitude:         Optional[float] = Field(None, description="Latitude do IP do terminal")
    terminal_ip_longitude:        Optional[float] = Field(None, description="Longitude do IP do terminal")
    device_ip_latitude:           Optional[float] = Field(None, description="Latitude do IP do dispositivo")
    device_ip_longitude:          Optional[float] = Field(None, description="Longitude do IP do dispositivo")
    cp_card_prev_term_dist_1h:    Optional[float] = Field(None, description="Distancia do terminal anterior (1h)")
    cp_card_term_vel_1h:          Optional[float] = Field(None, description="Velocidade geografica do cartao (1h)")

    # -- Velocidade e contagens -----------------------------------------------
    cardid_count_1m:              Optional[float] = Field(None, description="Contagem de transacoes do cartao (1 mes)")
    digitalwallettoken_count_3m:  Optional[float] = Field(None, description="Contagem tokens de carteira digital (3 meses)")

    # -- Identificadores de cartao e conta ------------------------------------
    mcc:                          Optional[float] = Field(None, description="Merchant Category Code")
    card_last4:                   Optional[float] = Field(None, description="Ultimos 4 digitos do cartao")
    card_days_to_expire:          Optional[float] = Field(None, description="Dias para expiracao do cartao")
    card_seq_number:              Optional[float] = Field(None, description="Numero sequencial do cartao")
    card_number_of_cardholders:   Optional[float] = Field(None, description="Numero de portadores do cartao")

    # -- Contadores de contas e cartoes do cliente ----------------------------
    account_associated_cards:     Optional[float] = Field(None, description="Cartoes associados a conta")
    account_number_of_cards:      Optional[float] = Field(None, description="Numero de cartoes na conta")
    account_rewards_used_count:   Optional[float] = Field(None, description="Recompensas utilizadas")
    customer_accounts:            Optional[float] = Field(None, description="Numero de contas do cliente")
    customer_number_of_accounts:  Optional[float] = Field(None, description="Total de contas do cliente")
    customer_number_of_cards:     Optional[float] = Field(None, description="Total de cartoes do cliente")
    customer_country_code_change_f: Optional[float] = Field(None, description="Flag de mudanca de pais do cliente")

    # -- Flags e indicadores numericos ----------------------------------------
    account_bic:                  Optional[float] = Field(None, description="BIC da conta")
    account_branch:               Optional[float] = Field(None, description="Agencia da conta")
    account_debt_payment_method:  Optional[float] = Field(None, description="Metodo de pagamento de debito")
    account_digital_signature:    Optional[float] = Field(None, description="Assinatura digital da conta")
    account_iban:                 Optional[float] = Field(None, description="IBAN da conta")
    account_last_activity:        Optional[float] = Field(None, description="Ultima atividade da conta")
    account_limits:               Optional[float] = Field(None, description="Limites da conta")
    account_otb_limit:            Optional[float] = Field(None, description="Limite OTB da conta")
    account_pw_change_flag:       Optional[float] = Field(None, description="Flag de mudanca de senha")
    account_status:               Optional[float] = Field(None, description="Status da conta")
    account_type:                 Optional[float] = Field(None, description="Tipo de conta")
    account_bank_branch_id:       Optional[float] = Field(None, description="ID da agencia bancaria")
    account_flag_adquirencia_activ: Optional[float] = Field(None, description="Flag adquirencia ativa")
    account_open_mode:            Optional[float] = Field(None, description="Modo de abertura da conta")
    account_ownership_type:       Optional[float] = Field(None, description="Tipo de titularidade da conta")
    acquirer_seq_number:          Optional[float] = Field(None, description="Numero sequencial do adquirente")
    application_transaction_counte: Optional[float] = Field(None, description="Contador de transacoes do aplicativo")
    lifecycle_seq_number:         Optional[float] = Field(None, description="Numero sequencial do ciclo de vida")
    mcc_cardbin_cntdist_cardexpdat: Optional[float] = Field(None, description="Contagem MCC/BIN por data de expiracao")
    p2p_account:                  Optional[float] = Field(None, description="Conta P2P")
    p2p_country:                  Optional[float] = Field(None, description="Pais P2P")
    pos_acceptor_disp_data_len:   Optional[float] = Field(None, description="Tamanho dados display POS")
    pos_acceptor_rec_data_len:    Optional[float] = Field(None, description="Tamanho dados recibo POS")
    pos_approval_code_len:        Optional[float] = Field(None, description="Tamanho do codigo de aprovacao POS")
    pos_cardholder_disp_data_len: Optional[float] = Field(None, description="Tamanho dados display portador")
    pos_cardholder_rec_data_len:  Optional[float] = Field(None, description="Tamanho dados recibo portador")
    pos_icc_data_len:             Optional[float] = Field(None, description="Tamanho dados ICC")
    pos_pin_len_cap:              Optional[float] = Field(None, description="Capacidade tamanho PIN")
    terminal_ip_country_code:     Optional[float] = Field(None, description="Codigo de pais do IP do terminal")
    transaction_seq_counter:      Optional[float] = Field(None, description="Contador sequencial da transacao")

    # -- Features categoricas -------------------------------------------------
    acceptor_region:              Optional[str] = Field(None, description="Regiao do estabelecimento (UF)")
    customer_region:              Optional[str] = Field(None, description="Regiao do cliente (UF)")
    acceptor_country_code:        Optional[str] = Field(None, description="Pais do estabelecimento")
    acceptor_add_contact_info:    Optional[str] = Field(None, description="Info de contato do estabelecimento")
    acceptor_street_addr:         Optional[str] = Field(None, description="Endereco do estabelecimento")
    acceptor_url:                 Optional[str] = Field(None, description="URL do estabelecimento")
    acceptor_zip:                 Optional[str] = Field(None, description="CEP do estabelecimento")
    account_currency:             Optional[str] = Field(None, description="Moeda da conta")
    account_bank_name:            Optional[str] = Field(None, description="Nome do banco")
    account_core:                 Optional[str] = Field(None, description="Core bancario")
    account_segment:              Optional[str] = Field(None, description="Segmento da conta")
    account_subsegment:           Optional[str] = Field(None, description="Subsegmento da conta")
    account_trx_bank_acc_number:  Optional[str] = Field(None, description="Numero conta bancaria de transacao")
    add_wallet_onboarding:        Optional[str] = Field(None, description="Onboarding de carteira digital")
    allow_contactless:            Optional[str] = Field(None, description="Permite contactless")
    allow_crossborder:            Optional[str] = Field(None, description="Permite transacoes cross-border")
    allow_ecommerce:              Optional[str] = Field(None, description="Permite e-commerce")
    allow_magstripe:              Optional[str] = Field(None, description="Permite tarja magnetica")
    allow_wallet:                 Optional[str] = Field(None, description="Permite carteira digital")
    allow_withdraw:               Optional[str] = Field(None, description="Permite saque")
    atc1:                         Optional[str] = Field(None, description="ATC 1")
    atc2:                         Optional[str] = Field(None, description="ATC 2")
    auth_type:                    Optional[str] = Field(None, description="Tipo de autorizacao")
    avs_code:                     Optional[str] = Field(None, description="Codigo AVS")
    business_app_transfer_purpose:Optional[str] = Field(None, description="Finalidade da transferencia")
    card_bin:                     Optional[str] = Field(None, description="BIN do cartao")
    card_brand:                   Optional[str] = Field(None, description="Bandeira do cartao")
    card_category:                Optional[str] = Field(None, description="Categoria do cartao")
    card_country_code:            Optional[str] = Field(None, description="Pais de emissao do cartao")
    card_expected_payments_frequen: Optional[str] = Field(None, description="Frequencia esperada de pagamentos")
    card_format:                  Optional[str] = Field(None, description="Formato do cartao")
    card_issuer:                  Optional[str] = Field(None, description="Emissor do cartao")
    card_presence_flag:           Optional[str] = Field(None, description="Flag de presenca do cartao")
    card_program:                 Optional[str] = Field(None, description="Programa do cartao")
    card_status:                  Optional[str] = Field(None, description="Status do cartao")
    card_type:                    Optional[str] = Field(None, description="Tipo do cartao")
    card_ver_data:                Optional[str] = Field(None, description="Dados de verificacao do cartao")
    cardholder_authentication_veri: Optional[str] = Field(None, description="Verificacao de autenticacao do portador")
    cardholder_bill_currency_code:Optional[str] = Field(None, description="Codigo de moeda da fatura")
    cardholder_bill_street_addr:  Optional[str] = Field(None, description="Endereco de fatura do portador")
    cardholder_bill_zip:          Optional[str] = Field(None, description="CEP de fatura do portador")
    cardholder_compress_bill_addr:Optional[str] = Field(None, description="Endereco comprimido de fatura")
    cardholder_presence_indicator:Optional[str] = Field(None, description="Indicador de presenca do portador")
    cardholder_verification_rule: Optional[str] = Field(None, description="Regra de verificacao do portador")
    currency_code:                Optional[str] = Field(None, description="Codigo de moeda da transacao")
    currency_unit:                Optional[str] = Field(None, description="Unidade monetaria")
    customer_cards:               Optional[str] = Field(None, description="Cartoes do cliente")
    customer_country_code:        Optional[str] = Field(None, description="Pais do cliente")
    customer_currency:            Optional[str] = Field(None, description="Moeda do cliente")
    customer_dob:                 Optional[str] = Field(None, description="Data de nascimento do cliente")
    customer_email_change_flag:   Optional[str] = Field(None, description="Flag de mudanca de e-mail")
    customer_kyc_lvl:             Optional[str] = Field(None, description="Nivel KYC do cliente")
    customer_occupation:          Optional[str] = Field(None, description="Ocupacao do cliente")
    customer_occupation_status:   Optional[str] = Field(None, description="Status de ocupacao")
    customer_phone_change_flag:   Optional[str] = Field(None, description="Flag de mudanca de telefone")
    customer_risk_rating:         Optional[str] = Field(None, description="Rating de risco do cliente")
    customer_segment:             Optional[str] = Field(None, description="Segmento do cliente")
    customer_street_addr:         Optional[str] = Field(None, description="Endereco do cliente")
    customer_zip:                 Optional[str] = Field(None, description="CEP do cliente")
    cvv2_presence_indicator:      Optional[str] = Field(None, description="Indicador de presenca do CVV2")
    cvv2_present_indicator:       Optional[str] = Field(None, description="CVV2 presente")
    cvv_val_code:                 Optional[str] = Field(None, description="Codigo de validacao CVV")
    device_ip:                    Optional[str] = Field(None, description="IP do dispositivo")
    device_ip_country_code:       Optional[str] = Field(None, description="Pais do IP do dispositivo")
    device_language:              Optional[str] = Field(None, description="Idioma do dispositivo")
    event_type:                   Optional[str] = Field(None, description="Tipo de evento")
    f3ds_secure_indicator:        Optional[str] = Field(None, description="Indicador 3DS seguro")
    f3ds_secure_protocol_version: Optional[str] = Field(None, description="Versao protocolo 3DS")
    fdz_exposed_schema_diffb:     Optional[str] = Field(None, description="Schema exposto FDZ")
    fdz_partition_index:          Optional[str] = Field(None, description="Indice de particao FDZ")
    first_recurring_payment:      Optional[str] = Field(None, description="Primeiro pagamento recorrente")
    info_type:                    Optional[str] = Field(None, description="Tipo de informacao")
    lifecycle_indicator:          Optional[str] = Field(None, description="Indicador de ciclo de vida")
    lists_state:                  Optional[str] = Field(None, description="Estado de listas (blacklist/whitelist)")
    malware_flag:                 Optional[str] = Field(None, description="Flag de malware")
    malware_indicator:            Optional[str] = Field(None, description="Indicador de malware")
    mc_merchant_fraud_code:       Optional[str] = Field(None, description="Codigo de fraude do comerciante Mastercard")
    merchant_initiated:           Optional[str] = Field(None, description="Transacao iniciada pelo comerciante")
    mti:                          Optional[str] = Field(None, description="Message Type Indicator")
    oob_comm_type:                Optional[str] = Field(None, description="Tipo de comunicacao OOB")
    p2p_address:                  Optional[str] = Field(None, description="Endereco P2P")
    p2p_name:                     Optional[str] = Field(None, description="Nome P2P")
    p2p_source_funds:             Optional[str] = Field(None, description="Fonte de fundos P2P")
    p2p_state:                    Optional[str] = Field(None, description="Estado P2P")
    pos_card_capture_cap:         Optional[str] = Field(None, description="Capacidade de captura do cartao no POS")
    pos_card_read_method:         Optional[str] = Field(None, description="Metodo de leitura do cartao")
    pos_cardholder_ver_cap:       Optional[str] = Field(None, description="Capacidade de verificacao do portador")
    pos_cardholder_ver_method:    Optional[str] = Field(None, description="Metodo de verificacao do portador")
    pos_environment:              Optional[str] = Field(None, description="Ambiente POS (ECOMMERCE, POS, etc.)")
    pos_read_cap:                 Optional[str] = Field(None, description="Capacidade de leitura do POS")
    proxy_flag:                   Optional[str] = Field(None, description="Flag de proxy detectado")
    reason_code:                  Optional[str] = Field(None, description="Codigo de razao")
    sca_required:                 Optional[str] = Field(None, description="SCA requerida")
    secure_3d_indicator:          Optional[str] = Field(None, description="Indicador 3D Secure")
    secure_3d_protocol_version:   Optional[str] = Field(None, description="Versao protocolo 3D Secure")
    secure_3d_val_code:           Optional[str] = Field(None, description="Codigo de validacao 3DS")
    secure_3d_val_status:         Optional[str] = Field(None, description="Status de validacao 3DS")
    segment_channel_type:         Optional[str] = Field(None, description="Tipo de canal do segmento")
    service_code:                 Optional[str] = Field(None, description="Codigo de servico")
    terminal_ip:                  Optional[str] = Field(None, description="IP do terminal")
    terminal_ip_region:           Optional[str] = Field(None, description="Regiao do IP do terminal")
    terminal_ip_zip:              Optional[str] = Field(None, description="CEP do IP do terminal")
    terminal_poc_flag:            Optional[str] = Field(None, description="Flag POC do terminal")
    terminal_region:              Optional[str] = Field(None, description="Regiao do terminal")
    terminal_street_addr:         Optional[str] = Field(None, description="Endereco do terminal")
    terminal_timezone:            Optional[str] = Field(None, description="Fuso horario do terminal")
    terminal_zip:                 Optional[str] = Field(None, description="CEP do terminal")
    transaction_category_code:    Optional[str] = Field(None, description="Codigo de categoria da transacao")
    transaction_status:           Optional[str] = Field(None, description="Status da transacao")
    transstain:                   Optional[str] = Field(None, description="Stain da transacao")
    trusted_acceptor:             Optional[str] = Field(None, description="Estabelecimento confiavel")
    ttc:                          Optional[str] = Field(None, description="TTC")
    tvr:                          Optional[str] = Field(None, description="TVR")
    customer_gender:              Optional[str] = Field(None, description="Genero do cliente")
    tipo_transacao:               Optional[str] = Field(None, description="Tipo da transacao")
    modalidade_transacao:         Optional[str] = Field(None, description="Modalidade da transacao")

    class Config:
        extra = "allow"   # campos adicionais sao aceitos e ignorados silenciosamente


# ---------------------------------------------------------------------------
# Schema de resposta
# ---------------------------------------------------------------------------

class RiscoNivel(str):
    BAIXO = "Baixo"
    MEDIO = "Medio"
    ALTO  = "Alto"


class TransacaoOutput(BaseModel):
    escore_risco:    float = Field(..., description="Probabilidade de fraude (0.0 – 1.0)")
    nivel_risco:     str   = Field(..., description="Baixo | Medio | Alto")
    recomendacao:    str   = Field(..., description="Acao recomendada pelo agente")
    threshold_alto:  float = Field(..., description="Threshold para nivel Alto")
    threshold_medio: float = Field(..., description="Threshold para nivel Medio")
    analisado_em:    str   = Field(..., description="Timestamp UTC da analise")


# ---------------------------------------------------------------------------
# Aplicacao FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API Antifraude",
    description=(
        "Analise de risco de transacoes de cartao em tempo real "
        "usando modelo LightGBM treinado com divisao temporal estrita."
    ),
    version="1.0.0",
)


def _classificar_risco(score: float) -> tuple[str, str]:
    """Retorna (nivel_risco, recomendacao) a partir do escore."""
    if score >= THRESHOLD_ALTO:
        return (
            "Alto",
            (
                "BLOQUEAR transacao e acionar equipe de investigacao. "
                f"Escore {score:.4f} >= threshold alto {THRESHOLD_ALTO}. "
                "Contatar o portador do cartao imediatamente para confirmar a operacao."
            ),
        )
    if score >= THRESHOLD_MEDIO:
        return (
            "Medio",
            (
                "REVISAR manualmente antes de autorizar. "
                f"Escore {score:.4f} esta entre {THRESHOLD_MEDIO} e {THRESHOLD_ALTO}. "
                "Considerar autorizacao com validacao adicional (OTP / biometria)."
            ),
        )
    return (
        "Baixo",
        (
            "APROVAR transacao. "
            f"Escore {score:.4f} < threshold medio {THRESHOLD_MEDIO}. "
            "Padrao de comportamento compativel com historico do cliente."
        ),
    )


def _build_dataframe(payload: TransacaoInput) -> pd.DataFrame:
    """
    Constroi um DataFrame com exatamente as features esperadas pelo modelo.
    Campos ausentes no payload ficam como NaN.
    - Colunas numericas sao forcadas para float64.
    - Colunas categoricas recebem dtype 'category' com as mesmas categorias
      usadas no treinamento (via _cat_map extraido do booster serializado).
    """
    raw: dict[str, Any] = payload.model_dump()

    # Monta linha com todas as features na ordem certa
    row: dict[str, Any] = {feat: raw.get(feat, None) for feat in _feature_names}
    df = pd.DataFrame([row], columns=_feature_names)

    # Colunas categoricas: dtype 'category' com categorias do treinamento
    # Valores fora do vocabulario ficam como NaN (LightGBM trata como missing)
    for col in _cat_feature_names:
        cats = _cat_map.get(col)
        if cats is not None:
            df[col] = pd.Categorical(df[col], categories=cats)
        else:
            df[col] = df[col].astype("category")

    # Colunas numericas: float64 explicito para evitar dtype object quando
    # todos os valores sao None (pandas padrao seria object)
    num_cols = [c for c in _feature_names if c not in _cat_feature_set]
    df[num_cols] = df[num_cols].astype("float64")

    return df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Status da API")
def root() -> dict:
    return {
        "status": "online",
        "modelo": os.path.basename(MODEL_PATH),
        "features": len(_feature_names),
        "threshold_medio": THRESHOLD_MEDIO,
        "threshold_alto": THRESHOLD_ALTO,
    }


@app.get("/health", summary="Health-check")
def health() -> dict:
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/features", summary="Features esperadas pelo modelo")
def features() -> dict:
    return {
        "total": len(_feature_names),
        "features": _feature_names,
    }


@app.post(
    "/analisar-transacao",
    response_model=TransacaoOutput,
    summary="Analisa uma transacao e retorna o nivel de risco",
)
def analisar_transacao(transacao: TransacaoInput) -> TransacaoOutput:
    """
    Recebe os dados de uma transacao em tempo real, calcula o escore de fraude
    usando o modelo LightGBM e retorna:

    - **escore_risco**: probabilidade de fraude (0.0 a 1.0)
    - **nivel_risco**: Baixo / Medio / Alto
    - **recomendacao**: acao recomendada pelo agente antifraude
    """
    try:
        df = _build_dataframe(transacao)
        score = float(_model.predict_proba(df)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Erro ao processar a transacao: {exc}",
        )

    nivel, recomendacao = _classificar_risco(score)

    return TransacaoOutput(
        escore_risco=round(score, 6),
        nivel_risco=nivel,
        recomendacao=recomendacao,
        threshold_alto=THRESHOLD_ALTO,
        threshold_medio=THRESHOLD_MEDIO,
        analisado_em=datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Entrada direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
