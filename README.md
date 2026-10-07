# Atividade Prática — Modelagem de Risco de Crédito

Previsão de inadimplência com Random Forest e análise crítica de métricas.

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python risco_credito.py
```

O script imprime todos os resultados no terminal e salva a comparação de recall em
`resultados_recall.csv`.

## Estrutura

```
├── risco_credito.py        # modelo, métricas e análise (Partes 1 a 4)
├── resultados_recall.csv   # recall da classe 1 por modelo e threshold
├── requirements.txt
└── outputs/
    ├── Relatorio_Risco_Credito_Pedro_Zanette.pdf   # relatório de entrega
    └── figures/            # gráficos usados no relatório
```

---

## Parte 1 — Modelo Base

| Métrica | Valor |
|---|---|
| Acurácia OOB (treino) | **0,9344** |
| ROC-AUC (teste) | **0,8566** |

## Parte 2 — Importância das Variáveis

| Variável | Importância Gini | Permutation (Δ ROC-AUC) |
|---|---|---|
| RendaMensal | 0,4121 | 0,1196 |
| ScoreCredito | 0,3555 | 0,1150 |
| EmprestimosAtivos | 0,2324 | 0,1040 |

A Permutation Importance (no teste) confirma a mesma ordem, com diferenças menores.

**4. Qual variável teve maior impacto?**
A **RendaMensal** (≈41% da importância total), seguida do ScoreCredito. Quanto menor a
renda, maior a probabilidade de inadimplência no modelo gerador dos dados.

## Parte 3 — Desbalanceamento e Análise Crítica

**5. Proporção de inadimplentes:** 135 de 2.000 clientes = **6,75%** (93,25% adimplentes).

**6. Matriz de Confusão (teste, threshold 0,5):**

|  | Previsto 0 | Previsto 1 |
|---|---|---|
| **Real 0** | 370 | 3 |
| **Real 1** | 25 | 2 |

**Relatório de Classificação:**

| Classe | Precision | Recall | F1-score | Suporte |
|---|---|---|---|---|
| 0 (adimplente) | 0,9367 | 0,9920 | 0,9635 | 373 |
| 1 (inadimplente) | 0,4000 | **0,0741** | 0,1250 | 27 |
| Acurácia | | | 0,9300 | 400 |

**7. Reflexão Crítica — a acurácia de 93,4% é confiável?**

Não. Como 93,25% dos clientes são adimplentes, um modelo "burro" que respondesse
**sempre "não vai inadimplir"** já teria ≈93% de acurácia — praticamente o mesmo valor
do Random Forest. Ou seja, a acurácia alta reflete o desbalanceamento, não a capacidade
de identificar o risco.

A matriz de confusão confirma isso: dos 27 inadimplentes do teste, o modelo detectou
apenas **2** (recall de 7,4%) e deixou passar **25**. Para a fintech, cada falso negativo
é crédito concedido a quem não vai pagar — justamente o prejuízo que o modelo deveria
evitar.

O risco de usar acurácia como métrica principal é aprovar um modelo que parece
excelente no papel mas não cumpre o objetivo de negócio. Neste cenário, métricas mais
adequadas são **Recall e F1 da classe 1**, **Precision-Recall** e **ROC-AUC** (0,857
mostra que o modelo *ordena* bem o risco — o problema está no threshold de 0,5).

## Parte 4 — Desafio Extra: `class_weight='balanced'` e threshold

| Modelo | Threshold | Recall classe 1 | Detectados (TP) | Perdidos (FN) | Falsos alarmes (FP) |
|---|---|---|---|---|---|
| Original | 0,50 | 0,07 | 2 | 25 | 3 |
| Original | 0,30 | 0,26 | 7 | 20 | 13 |
| Original | 0,20 | 0,56 | 15 | 12 | 27 |
| Original | 0,10 | 0,70 | 19 | 8 | 63 |
| Balanced | 0,50 | 0,70 | 19 | 8 | 52 |
| Balanced | 0,30 | 0,81 | 22 | 5 | 88 |
| Balanced | 0,20 | 0,81 | 22 | 5 | 108 |
| Balanced | 0,10 | 0,93 | 25 | 2 | 154 |

**8. Impacto na taxa de detecção dos inadimplentes:**

- Só com `class_weight='balanced'` (threshold 0,5), o recall da classe 1 sobe de
  **7% para 70%**: o modelo passa a detectar 19 dos 27 inadimplentes, em vez de 2.
- Reduzir o threshold tem efeito parecido: no modelo original, baixar para 0,10 também
  leva o recall a 70%. Combinando as duas técnicas, chega a **93%** (threshold 0,10).
- O ganho tem custo: os **falsos alarmes aumentam** (de 3 para 52 no balanced 0,5, e
  até 154 no threshold 0,10), ou seja, bons clientes seriam recusados ou enviados para
  análise manual.
- A ROC-AUC quase não muda (0,857 → 0,861): o modelo não ficou "melhor" em separar as
  classes, apenas mudou o ponto de corte de decisão.

**Conclusão:** a escolha do threshold é uma decisão de negócio. Se o custo de um
calote for muito maior que o de recusar um bom cliente, vale priorizar recall
(ex.: balanced com threshold 0,3–0,5). O ideal é definir o limiar a partir do custo
real de cada tipo de erro.
