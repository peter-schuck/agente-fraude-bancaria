# 🛡️ Sistema Antifraude Bancário — Sicredi

Sistema completo de detecção de fraudes em transações de cartão de crédito e débito, composto por um modelo de Machine Learning (LightGBM), uma API REST (FastAPI), um dashboard interativo (Streamlit) e um agente de análise inteligente.

---

## 📋 Visão geral

O projeto foi desenvolvido como um **case avançado de Ciência de Dados e IA** sobre um dataset sintético de 900.000 transações de cartões, com distribuição controlada de fraudes (~0,10%) e 297 features cobrindo valores, comportamento, geolocalização, velocidade e perfil de cliente.

O fluxo completo do sistema é:

```
Dataset CSV (900k linhas)
       │
       ▼
train_antifraude.py          ← Treina o modelo LightGBM com Optuna
       │  gera
       ▼
modelo_antifraude.pkl        ← Modelo serializado
       │  carregado por
       ▼
main.py (FastAPI)            ← Backend REST com predição em tempo real + SHAP
       │  consumido por
       ▼
dashboard.py (Streamlit)     ← Frontend interativo para analistas
       │  usa
       ▼
agente_antifraude.py         ← Agente de IA com regras heurísticas e chat
```

---

## 📁 Estrutura de arquivos

| Arquivo | Descrição |
|---|---|
| [`train_antifraude.py`](#treinamento) | Treina o modelo LightGBM com busca de hiperparâmetros via Optuna. Gera `modelo_antifraude.pkl`, `curva_precision_recall.png` e `matriz_confusao.png`. |
| [`main.py`](#backend-principal) | Backend principal (FastAPI v2). Gerencia transações em memória, aplica o pipeline de features de velocidade e retorna escore de fraude + valores SHAP por transação. |
| [`api.py`](#api-simplificada) | API simplificada (FastAPI v1). Endpoint único `POST /analisar-transacao` para inferência pontual, sem gerenciamento de histórico. |
| [`dashboard.py`](#dashboard) | Frontend Streamlit. Interface visual para analistas com lista de transações suspeitas, gauge de escore, gráfico SHAP, histórico do cliente e chat com agente IA. |
| [`agente_antifraude.py`](#agente-de-ia) | Agente de análise que combina o escore LightGBM com regras heurísticas (velocidade, madrugada, mudança de local, valor atípico) e sugere ações ao analista. |
| [`feature_engineering.py`](#feature-engineering) | Pipeline de engenharia de features: velocidade do cartão (janelas 10 min / 1 h), intervalo temporal, flag de valor alto, distância Haversine entre terminais e média EWM por MCC. |
| [`seed_historico.py`](#seed-histórico) | Utilitário de carga histórica. Lê o CSV em chunks, amostra de forma estratificada (preservando proporção de fraudes) e popula o banco em memória do backend. |
| [`explorar_dataset.py`](#exploração-do-dataset) | Script de EDA: estatísticas de nulos, distribuição de tipos, médias/desvios por coluna e matriz de correlação de uma amostra de 10k linhas. |
| `modelo_antifraude.pkl` | Modelo LightGBM serializado com `joblib`. Carregado pelo backend no startup. |
| `case_transacoes_cartoes_sintetico_900k.csv` | Dataset sintético com 900.000 transações e 297 colunas. **Necessário para treinar o modelo e para a carga histórica opcional.** |
| `dicionario_dados_case_prevencao_sicredi.xlsx` | Dicionário de dados com a descrição de cada uma das 297 colunas do dataset. |
| `Orientacoes_case_prevencao_sicredi.txt` | Orientações gerais do case: descrição do dataset, distribuições controladas e sugestões pedagógicas. |
| `curva_precision_recall.png` | Gráfico da curva Precision-Recall gerado após o treinamento. |
| `matriz_confusao.png` | Matriz de confusão visual gerada após o treinamento. |

---

## 🔧 Pré-requisitos

- Python **3.10+**
- `pip` atualizado (`pip install --upgrade pip`)

Instale as dependências:

```bash
pip install fastapi uvicorn[standard] streamlit requests \
            lightgbm optuna joblib pandas numpy scikit-learn \
            matplotlib shap
```

> **Nota:** O arquivo `case_transacoes_cartoes_sintetico_900k.csv` deve estar presente na pasta `antifraude/` para que o treinamento e a carga histórica funcionem.

---

## 🚀 Como rodar

### 1. Treinamento do modelo

> Pule este passo se `modelo_antifraude.pkl` já estiver presente.

```bash
# Na pasta antifraude/
python train_antifraude.py

# Para controlar o número de trials do Optuna (padrão: 50):
python train_antifraude.py --trials 30
```

O script:
1. Lê o CSV em chunks de 100.000 linhas
2. Faz divisão temporal estrita (80% treino / 20% teste)
3. Busca hiperparâmetros com Optuna (maximizando PR-AUC)
4. Treina o modelo final e exibe métricas no console
5. Salva `modelo_antifraude.pkl`, `curva_precision_recall.png` e `matriz_confusao.png`

---

### 2. Backend (API FastAPI)

O backend precisa ser iniciado antes do dashboard.

```bash
# Na pasta antifraude/
uvicorn main:app --reload
```

A API estará disponível em `http://127.0.0.1:8000`.

**Documentação interativa (Swagger UI):** `http://127.0.0.1:8000/docs`

#### Carga histórica opcional

Para popular o backend com uma amostra de transações reais do CSV no startup:

```bash
# Windows (PowerShell)
$env:SEED_HISTORICO="1"; uvicorn main:app --reload

# Linux / macOS
SEED_HISTORICO=1 uvicorn main:app --reload
```

Variáveis de ambiente da carga histórica:

| Variável | Padrão | Descrição |
|---|---|---|
| `SEED_HISTORICO` | `0` | `1` para ativar a carga histórica no startup |
| `SEED_SAMPLE_SIZE` | `500` | Número de transações a amostrar do CSV |
| `SEED_CHUNK_SIZE` | `100000` | Linhas lidas por bloco do CSV |
| `SEED_RANDOM_SEED` | `42` | Semente para reproducibilidade |

---

### 3. Dashboard (Streamlit)

> O backend deve estar rodando antes de iniciar o dashboard.

```bash
# Em outro terminal, na pasta antifraude/
streamlit run dashboard.py
```

O dashboard abrirá automaticamente no navegador em `http://localhost:8501`.

---

### 4. API simplificada (opcional)

Para usar a API de inferência pontual sem gerenciamento de histórico:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints disponíveis:

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Status da API e número de features carregadas |
| `GET` | `/health` | Health-check |
| `GET` | `/features` | Lista de todas as features esperadas pelo modelo |
| `POST` | `/analisar-transacao` | Recebe dados de uma transação e retorna escore + nível de risco |

---

## 🔌 Endpoints do Backend Principal (`main.py`)

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/transacoes` | Registra nova transação e calcula escore de fraude + SHAP |
| `GET` | `/transacoes-suspeitas` | Lista transações com `suspeita=True` (risco médio ou alto) |
| `GET` | `/transacoes` | Lista todas as transações |
| `GET` | `/transacao/{id}` | Detalhe de uma transação específica |
| `GET` | `/cliente/{client_id}/historico` | Histórico de transações de um cliente |
| `POST` | `/transacao/{id}/executar-acao` | Executa ação de analista (bloquear compra/cartão, enviar mensagem) |

### Exemplo de uso (curl)

```bash
# Registrar uma transação
curl -X POST http://127.0.0.1:8000/transacoes \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "CLI-010",
    "card_id": "CARD-010",
    "timestamp": "2024-07-16T03:15:00",
    "amount": 1500.00,
    "local": "São Paulo, SP",
    "categoria": "compra_internacional",
    "card_type": "credito",
    "pos_environment": "ECOMMERCE",
    "mc_fraud_score": 450.0,
    "visa_fraud_score": 380.0
  }'
```

### Níveis de risco e thresholds

| Nível | Threshold | Ação recomendada |
|---|---|---|
| 🟢 **Baixo** | escore < 0.15 | Aprovar automaticamente |
| 🟡 **Médio** | 0.15 ≤ escore < 0.40 | Revisar manualmente / pedir validação adicional |
| 🔴 **Alto** | escore ≥ 0.40 | Bloquear e acionar equipe de investigação |

---

## 🤖 Descrição detalhada dos módulos

### Treinamento

`train_antifraude.py` implementa uma pipeline de ML com as seguintes etapas:

- **Divisão temporal estrita:** os 80% cronologicamente mais antigos são usados para treino e os 20% mais recentes para teste. Isso evita data leakage por data.
- **Desbalanceamento:** tratado via `scale_pos_weight` (razão negativo/positivo), ajustado pelo Optuna.
- **Busca de hiperparâmetros (Optuna):** maximiza PR-AUC em validação interna com TPE Sampler. Parâmetros buscados: `num_leaves`, `learning_rate`, `min_child_samples`, `scale_pos_weight`, `feature_fraction`, `bagging_fraction`, `reg_alpha`, `reg_lambda`, `max_depth`.
- **Métricas de avaliação:** PR-AUC, Precision, Recall, F1-Score, Matriz de Confusão.
- **Feature selection:** colunas numéricas + categóricas com cardinalidade ≤ 200, excluindo IDs e colunas com data leakage.

### Backend principal

`main.py` é a API FastAPI que gerencia o ciclo de vida completo das transações:

- No **startup (lifespan)**, carrega o modelo, inicializa o `TreeExplainer` do SHAP e popula o banco com 8 transações seed pré-configuradas (3 baixo risco, 2 médio, 3 alto).
- A cada nova transação recebida, aplica `build_velocity_features` sobre **todo o histórico** para recalcular features de velocidade contextuais e então chama `predict_proba` do LightGBM.
- Retorna `escore_risco`, `risco`, `suspeita` e `shap_values` (top-15 features por magnitude SHAP na escala log-odds).

### API simplificada

`api.py` é uma versão mais leve sem estado em memória, voltada a integrações externas que precisam apenas da inferência pontual. Aceita payloads parciais (campos ausentes = NaN, tratado internamente pelo LightGBM).

### Dashboard

`dashboard.py` consome a API do `main.py` e exibe:

- **Sidebar:** lista de transações suspeitas com escore, categoria e valor. Botão de atualização.
- **Coluna de detalhes (60%):** informações da transação selecionada, gauge de escore colorido por risco, métricas de velocidade do pipeline (txn/10min, txn/1h, intervalo, flag de valor alto), gráfico SHAP interativo e histórico recente do cliente.
- **Coluna de chat (40%):** chat livre com o Agente de IA, inicializado automaticamente com a análise da transação selecionada. Suporta perguntas sobre escore, velocidade, intervalo, valor, histórico do cliente, localização e recomendação de ação.
- **Botões de ação:** Bloquear Compra, Bloquear Cartão e Enviar Mensagem ao cliente.

### Agente de IA

`agente_antifraude.py` implementa a classe `AgenteAntifraude` que combina o escore do modelo com regras heurísticas:

| Regra | Gatilho | Pontos |
|---|---|---|
| Escore alto | ≥ 0.40 | +3 |
| Escore médio | ≥ 0.15 | +1 |
| Velocidade do cartão (10 min) | ≥ 2 transações | +2 |
| Velocidade do cartão (1 hora) | ≥ 4 transações | +1 |
| Intervalo crítico | < 60 s | +2 |
| Intervalo curto | < 5 min | +1 |
| Flag de valor alto | `high_value_flag = 1` | +1 |
| Valor atípico (histórico) | > 2σ ou > 4× média | +1 a +2 |
| Mudança rápida de local | diferente em < 2 h | +2 |
| Horário de madrugada | 0h–5h | +1 |
| Categoria de alto risco | internacional/saque/transferência | +1 |
| Primeira transação | sem histórico | +1 |

O nível final é o máximo entre o nível do modelo e o nível heurístico (nunca rebaixado). A ação sugerida é: `bloquear_cartao` (alto), `enviar_mensagem` (médio), `monitorar` (baixo).

### Feature Engineering

`feature_engineering.py` — função principal `build_velocity_features(df)`:

| Feature gerada | Descrição |
|---|---|
| `card_txn_count_10min` | Transações do mesmo cartão nos últimos 10 minutos (janela rolante, sem incluir a atual) |
| `card_txn_count_1h` | Transações do mesmo cartão na última hora |
| `seconds_since_last_txn` | Segundos desde a transação anterior do mesmo cliente |
| `high_value_flag` | `1` se o valor supera 2× a média histórica expandida do cliente |
| `haversine_dist_km` | Distância em km entre o terminal atual e o anterior do mesmo cartão |
| `ewm_amount_by_mcc` | Média ponderada exponencialmente (α=0.3) do valor por cartão+MCC |
| `cumulative_rejected_txns` | Contagem acumulada de transações rejeitadas anteriores do cartão |

Todas as features são calculadas **sem data leakage** (usando `shift(1)` e `closed='left'`).

### Seed histórico

`seed_historico.py` — função `carregar_historico(transacoes_db, id_offset)`:

- Lê o CSV em chunks para economizar memória
- Faz amostragem estratificada (preserva a proporção de fraudes ~0,10%)
- Aplica `build_velocity_features` sobre a amostra completa
- Retorna o próximo ID disponível para o contador automático do backend

### Exploração do dataset

`explorar_dataset.py` — script de EDA que calcula, sem carregar todo o CSV na memória:
- Total de linhas e colunas
- Percentual de nulos por coluna (top-20 e colunas com ≥ 50% nulos)
- Classificação de colunas numéricas e categóricas
- Médias e desvios-padrão online (via soma acumulada)
- Matriz de correlação de uma amostra de 10.000 linhas

---

## 📊 Sobre o dataset

O arquivo `case_transacoes_cartoes_sintetico_900k.csv` é **totalmente sintético** — todos os IDs, nomes, documentos e endereços são placeholders fictícios. Os padrões estatísticos simulam um ambiente bancário real:

| Característica | Detalhe |
|---|---|
| Total de linhas | 900.000 |
| Total de colunas | 297 |
| Taxa de fraude | 0,10% (900 fraudes) |
| Tipo débito | 60% (540.000) |
| Tipo crédito | 40% (360.000) |
| Fraudes concentradas em | Crédito + cartão não presente |
| Sinais de fraude | Rajadas sequenciais, alta velocidade, horários concentrados, divergência geográfica, valores anômalos |

---

## 🗂️ Fluxo de dados no backend

```
POST /transacoes
    │
    ├─ Armazena o payload em transacoes_db
    │
    ├─ build_velocity_features(transacoes_db completo)
    │       ↳ card_txn_count_10min, card_txn_count_1h,
    │         seconds_since_last_txn, high_value_flag
    │
    ├─ _build_dataframe_para_modelo(payload + features de velocidade)
    │       ↳ DataFrame com exatamente as features do treino
    │         (categóricas com dtype correto, numéricas float64)
    │
    ├─ LightGBM.predict_proba(df)[0, 1]  → escore_risco
    │
    ├─ TreeExplainer.shap_values(df)     → top-15 SHAP (log-odds)
    │
    └─ _classificar_risco(escore)        → risco + suspeita
```

---

## ⚠️ Notas importantes

- O backend usa **armazenamento em memória**. Todos os dados são perdidos ao reiniciar o processo.
- O modelo deve ser **retreinado** caso o dataset mude, pois as categorias das features categóricas são serializadas junto com o modelo.
- Os thresholds (0.15 para médio, 0.40 para alto) foram calibrados a partir da curva Precision-Recall do treinamento e podem ser ajustados em `main.py` e `api.py`.
- Para ambientes de produção, considere substituir o banco em memória por um banco de dados persistente e adicionar autenticação aos endpoints.
