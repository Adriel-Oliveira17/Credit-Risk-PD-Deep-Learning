# Modelagem de Risco de Crédito: Previsão de Probabilidade de Default (PD) com Rede Neural (MLP)

Este projeto implementa um pipeline completo de *Credit Scoring* para estimar a **Probabilidade de Default em 12 meses (`default_12m`)** utilizando uma **Rede Neural Artificial (Perceptron Multicamadas - MLP)** desenvolvida em TensorFlow/Keras.

O fluxo contempla engenharia de atributos financeiros, validação temporal (*In-Time* vs. *Out-of-Time*), seleção de variáveis via *Information Value* (IV), treinamento regularizado e auditoria de performance e estabilidade populacional (PSI).

---

## Estrutura do Projeto

    ├── pd com rede.ipynb   # Notebook principal com todo o pipeline de modelagem e avaliação
    ├── Dados.csv           # Base de dados de crédito (safras de 2022 a 2025)
    └── README.md           # Documentação do projeto

---

## Tecnologias e Bibliotecas

* **Python 3.13+**
* **Manipulação de Dados:** `pandas`, `numpy`
* **Visualização:** `matplotlib`, `seaborn`
* **Pré-processamento e Métricas:** `scikit-learn`
* **Deep Learning:** `tensorflow` / `keras`

---

## Pipeline de Modelagem

### 1. Engenharia de Atributos (*Feature Engineering*)
A partir das variáveis brutas da carteira (`renda_mensal`, `limite`, `saldo`, `variacao_saldo_3m`), são criadas novas métricas financeiras e aplicados tratamentos de escala:
* **`limite_usado_%`**: Razão entre o saldo devedor e o limite total concedido (`saldo / limite`).
* **`comprometimento_renda_%`**: Relação entre a renda mensal e o saldo da dívida (`renda_mensal / saldo`).
* **Tratamento de Outliers (`variacao_saldo_3m`)**: Aplicação de *clipping* no intervalo `[-1, 1]` para conter distorções extremas na variação trimestral do saldo.

### 2. Validação Temporal (*In-Time* vs. *Out-of-Time*)
Para simular o comportamento real de produção e garantir a capacidade de generalização temporal do modelo:
* **Treino (*In-Time*):** Safras de observação anteriores a 2025 (`safra_obs < 2025`).
* **Teste (*Out-of-Time* - OOT):** Safra de 2025 (`safra_obs == 2025`).

### 3. Seleção de Variáveis (*Information Value* - IV)
Calcula-se o IV de todas as variáveis candidatas na base de treino (discretizadas em 5 quantis). São selecionadas apenas as variáveis com poder preditivo relevante (`IV >= 0.02`), evitando ruído e *data leakage* (como a exclusão natural de `saldo_default`):
1. `qtd_atrasos_curtos_6m`
2. `limite_usado_%`
3. `variacao_saldo_3m`
4. `comprometimento_renda_%`
5. `saldo`
6. `pagou_minimo_ult_fatura`

### 4. Pré-processamento e Arquitetura da Rede Neural
* **Padronização:** `StandardScaler` ajustado exclusivamente no conjunto de treino e aplicado no conjunto de teste.
* **Arquitetura MLP (`6 -> 32 -> 16 -> 8 -> 1`):**
  * Camadas densas com ativação `ReLU` e **regularização L2** (`1e-4`).
  * `BatchNormalization` e `Dropout(0.3)` para estabilização do aprendizado e prevenção contra *overfitting*.
  * Camada de saída com ativação `sigmoid` para estimativa direta da probabilidade (`[0, 1]`).
* **Otimização e Callbacks:** Otimizador `Adam`, função de perda `binary_crossentropy`, `EarlyStopping` (paciência de 15 épocas monitorando `val_auc`) e `ReduceLROnPlateau` (redução pela metade da taxa de aprendizado após 5 épocas sem ganho na `val_auc`).

---

## Resultados e Métricas (Safra *Out-of-Time* - 2025)

O modelo apresentou forte calibração entre a inadimplência real e a probabilidade média prevista, além de alta estabilidade temporal entre as safras:

| Métrica | Valor Obtido | Interpretação |
| :--- | :---: | :--- |
| **AUC-ROC** | `0.7043` | Capacidade de ordenação e discriminação entre bons e maus pagadores |
| **Índice de Gini** | `0.4085` | Métrica padrão de mercado (`2 * AUC - 1`) |
| **KS (Kolmogorov-Smirnov)** | `0.2983` | Separação máxima entre as distribuições acumuladas de adimplentes e inadimplentes |
| **Brier Score** | `0.1382` | Erro quadrático médio das probabilidades previstas (calibração) |
| **Average Precision (PR-AUC)** | `0.3778` | Área sob a curva de Precisão-Recall |
| **Taxa de Inadimplência Observada** | `18.75%` | Proporção real de *default* na safra de 2025 |
| **PD Média Prevista** | `18.70%` | Probabilidade média estimada pelo modelo (aderência de calibração) |
| **PSI do Score** | `0.0010` | **Estável** (`PSI < 0.10` indica ausência de *drift* relevante na distribuição do score) |

### Estabilidade das Variáveis (PSI por Feature)
Todas as variáveis selecionadas mantiveram-se dentro da faixa de estabilidade (`PSI < 0.10`) na transição para a safra de 2025:

| Variável | PSI | Status |
| :--- | :---: | :--- |
| `saldo` | `0.001223` | Estável |
| `variacao_saldo_3m` | `0.001213` | Estável |
| `comprometimento_renda_%` | `0.000533` | Estável |
| `limite_usado_%` | `0.000520` | Estável |
| `qtd_atrasos_curtos_6m` | `0.000222` | Estável |
| `pagou_minimo_ult_fatura` | `0.000000` | Estável |

---

## Como Executar

1. Clone este repositório:

       git clone https://github.com/seu-usuario/seu-repositorio.git
       cd seu-repositorio

2. Instale as dependências necessárias:

       pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter

3. Certifique-se de que o arquivo `Dados.csv` esteja no mesmo diretório do notebook e execute todas as células de `pd com rede.ipynb`.
